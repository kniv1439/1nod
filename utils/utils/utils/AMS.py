import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils.Layer import Transformer_Layer
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP


class AMS(nn.Module):
    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1, batch_norm=False):
        super(AMS, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.start_linear = nn.Linear(in_features=input_size, out_features=input_size)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(Transformer_Layer(
                device=device,
                d_model=d_model,
                d_ff=d_ff,
                dynamic=dynamic,
                num_nodes=num_nodes,
                patch_nums=patch_nums,
                patch_size=patch,
                factorized=True,
                layer_number=layer_number,
                batch_norm=batch_norm,
                nhead=self.num_experts  # 将 num_experts 传递给 nhead
            ))

        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        # x 必须是 (batch, seq_len, feat_dim)
        if x.dim() == 4:
            x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # x: (batch*seq_len, feat_dim)
        # 不要再用 start_linear
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        orig_shape = x.shape  # (batch, seq_len, feat_dim)
        assert x.dim() == 3, f"AMS输入必须是三维，当前shape: {x.shape}"

        # 1. 分解
        new_x = self.seasonality_and_trend_decompose(x)  # (batch, seq_len, feat_dim)

        # 2. gating和专家分配需要二维
        x_2d = x.view(-1, x.size(-1))  # (batch*seq_len, feat_dim)
        new_x_2d = new_x.view(-1, new_x.size(-1))

        gates, load = self.noisy_top_k_gating(new_x_2d, self.training)
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_2d)  # 每个专家输入 shape: (num_tokens, feat_dim)

        # 3. 专家输入如果需要三维，恢复三维
        expert_outputs = []
        for i in range(self.num_experts):
            inp = expert_inputs[i]
            # 如果你的 Transformer_Layer 只支持三维 (batch, seq_len, feat_dim)，则需要如下处理：
            if inp.dim() == 2:
                inp = inp.unsqueeze(1)  # (num_tokens, 1, feat_dim)
            out = self.experts[i](inp)
            if isinstance(out, tuple):
                out = out[0]
            # 如果输出多一维，去掉
            if out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)
            expert_outputs.append(out)

        output_2d = dispatcher.combine(expert_outputs)  # (batch*seq_len, feat_dim) 或 4维
        # 如果专家输出是4维 (batch*seq_len,1,1,feat_dim) 则压缩回2维
        if output_2d.dim() == 4 and output_2d.size(1) == 1 and output_2d.size(2) == 1:
            output_2d = output_2d.squeeze(2).squeeze(1)  # -> (batch*seq_len, feat_dim)

        if self.residual_connection:
            # 保证形状一致再做残差
            if output_2d.shape == x_2d.shape:
                output_2d = output_2d + x_2d
        output = output_2d.view(orig_shape)  # 恢复三维
        return output, balance_loss





