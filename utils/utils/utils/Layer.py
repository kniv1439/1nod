import math
import torch
import torch.nn as nn
from torch.nn import init
import time
import torch.nn.functional as F
from utils.Embedding import *



class Transformer_Layer(nn.Module):
    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size, dynamic, factorized, layer_number, batch_norm,nhead):
        super(Transformer_Layer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.batch_norm = batch_norm


        ##intra_patch_attention
        self.intra_embeddings = nn.Parameter(torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16),
                                                requires_grad=True)
        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(16, self.d_model)]) for _ in range(self.patch_nums)])
        self.intra_d_model = self.d_model
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16, num_nodes=num_nodes,
                                                          factorized=factorized, number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None, num_nodes=num_nodes,
                                                        factorized=False, number_of_weights=2)
        self.intra_Linear = nn.Linear(self.d_model, self.d_model)



        ##inter_patch_attention
        self.stride = patch_size
        # patch_num = int((context_window - cut_size) / self.stride + 1)

        self.inter_d_model = self.d_model * self.patch_size
        ##inter_embedding
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        # Positional encoding
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model)
        self.nhead = nhead 
        d_k = self.inter_d_model // nhead
        d_v = self.inter_d_model // nhead
        self.inter_patch_attention = Inter_Patch_Attention(self.inter_d_model, self.inter_d_model, nhead, d_k, d_v, attn_dropout=0,
                                          proj_dropout=0.1, res_attention=False)


        ##Normalization
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))
        self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))

        ##FFN
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_ff, self.d_model, bias=True))

    def forward(self, x):
        # 如果序列长度(seq_len 取 x 的 dim=1) 小于 patch_size，则直接跳过 Patch-Attention 逻辑，
        # 只做一次线性 + FFN 作为退化路径，避免 unfold 触发报错。
        if x.size(1) < self.patch_size:
            # 简单两层：Linear -> GELU -> Linear，与 FFN 保持一致维度
            shortcut = x
            x = self.intra_Linear(x)  # 保证维度与 d_model 对齐
            x = self.ff(x) + x        # 轻量 FFN
            out = x + shortcut        # 残差
            # 将 (B,S,Dim) -> (B, S, 1, Dim) 保证 4 维，满足 einsum i j k h
            if out.dim() == 3:
                out = out.unsqueeze(2)
            return out, None

        # 全局状态保存原始输入，用于残差连接
        original_input = x

        # 计算当前输入的实际序列长度和特征维度
        batch_size, seq_len, feat_dim = x.shape
        actual_patch_num = seq_len
        
        # 获取共享和独立的权重
        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        ###### 1. Intra-Patch Attention ######
        # 收集所有patch的特征，跨patch内部处理
        intra_patches = []
        
        # 动态计算patch划分，不使用固定数量 
        for i in range(actual_patch_num):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.patch_size, seq_len)  # 确保不越界
            
            if end_idx <= start_idx:  # 无效的patch
                continue
                
            # 提取当前patch的特征
            patch = x[:, start_idx:end_idx, :]  # (batch, patch_size, feat_dim)
            intra_patches.append(patch)
        
        # 如果没有收集到任何patch（极端情况），使用原始输入
        if not intra_patches:
            intra_out = self.intra_Linear(x)
        else:
            # 拼接所有patch特征并应用线性变换
            intra_out = torch.cat(intra_patches, dim=1)  # (batch, total_patch_len, feat_dim)
            intra_out = self.intra_Linear(intra_out)
            
            # 确保输出与原始输入形状一致
            if intra_out.size(1) != seq_len:
                # 如果长度不一致，进行插值或填充
                intra_out = torch.nn.functional.interpolate(
                    intra_out.transpose(1, 2),  # [B, D, L]
                    size=seq_len,
                    mode='linear'
                ).transpose(1, 2)  # [B, L, D]

        ###### 2. Inter-Patch Attention ######
        # 处理patch间的关系
        if seq_len >= self.patch_size and self.stride > 0:
            # 执行unfold操作分割patch
            unfolded = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
            # 统一处理为3维 [batch, patch_num, patch_size*feat_dim]
            if unfolded.dim() == 4:  # [batch, patch_num, feat_dim, patch_size]
                inter_input = unfolded.reshape(
                    batch_size, 
                    unfolded.size(1),  # patch_num
                    unfolded.size(2) * unfolded.size(3)  # feat_dim * patch_size
                )
            else:
                # 处理其他可能的维度情况
                inter_input = unfolded.reshape(batch_size, actual_patch_num, -1)
            
            # 应用线性投影
            inter_input = self.emb_linear(inter_input)
            
            # 处理位置编码
            current_patch_num = inter_input.size(1)
            current_dim = inter_input.size(2)
            
            # 动态更新位置编码
            if self.W_pos is None or self.W_pos.shape[1] != current_patch_num or self.W_pos.shape[2] != current_dim:
                pos_encoding = positional_encoding(
                    pe='zeros', 
                    learn_pe=True, 
                    q_len=current_patch_num, 
                    d_model=current_dim
                ).to(x.device)
                self.W_pos = nn.Parameter(pos_encoding, requires_grad=True)
            
            # 添加位置编码并应用dropout
            inter_input = self.dropout(inter_input + self.W_pos)
            
            # 执行注意力操作
            inter_out_attn, attention = self.inter_patch_attention(Q=inter_input, K=inter_input, V=inter_input)
            
            # 重塑到与原始输入兼容的形状以便残差连接
            # 先展平再插值到原始序列长度
            inter_out = inter_out_attn.reshape(batch_size, -1, self.d_model)
            
            # 如果长度不匹配，使用插值调整
            if inter_out.size(1) != seq_len:
                inter_out = torch.nn.functional.interpolate(
                    inter_out.transpose(1, 2),  # [B, D, L]
                    size=seq_len,
                    mode='linear'
                ).transpose(1, 2)  # [B, L, D]
        else:
            # 如果序列太短，跳过Inter Attention
            # 创建一个全零tensor作为placeholder
            inter_out = torch.zeros_like(x)
            attention = None

        ###### 3. 残差连接和最终处理 ######
        # 确保所有tensor形状一致再做残差
        assert original_input.shape == intra_out.shape == inter_out.shape, \
               f"形状不一致: original={original_input.shape}, intra={intra_out.shape}, inter={inter_out.shape}"
        
        # 残差连接
        out = original_input + intra_out + inter_out
            
        # 批量归一化
        if self.batch_norm:
            # 调整归一化层的输入形状
            reshaped_for_norm = out.reshape(-1, out.size(-2) if out.dim() > 2 else 1, self.d_model)
            out = self.norm_attn(reshaped_for_norm)
            # 恢复原始形状
            out = out.reshape(batch_size, seq_len, -1)
            
        # FFN
        shortcut = out  # 保存用于另一个残差连接
        out = self.dropout(out)
        out = self.ff(out) + shortcut  # FFN输出 + 残差
        
        # 最终归一化
        if self.batch_norm:
            reshaped_for_norm = out.reshape(-1, out.size(-2) if out.dim() > 2 else 1, self.d_model)
            out = self.norm_ffn(reshaped_for_norm)
            # 恢复与输入一致的形状
            out = out.reshape(batch_size, seq_len, -1)

        return out, attention



class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))



        attention = torch.matmul(query, key)
        attention /= (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x, attention


class Inter_Patch_Attention(nn.Module):
    def __init__(self, d_model, out_dim, nhead, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // nhead if d_k is None else d_k
        d_v = d_model // nhead if d_v is None else d_v

        self.nhead, self.d_k, self.d_v = nhead, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * nhead, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * nhead, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * nhead, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, nhead, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(nhead * d_v, out_dim), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.nhead, self.d_k).transpose(1,
                                                                                 2)  # q_s    : [bs x nhead x q_len x d_k]  此处的q_len为patch_num
        k_s = self.W_K(K).view(bs, K.shape[1], self.nhead, self.d_k).permute(0, 2, 3,
                                                                               1)  # k_s    : [bs x nhead x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, V.shape[1], self.nhead, self.d_v).transpose(1,
                                                                                 2)  # v_s    : [bs x nhead x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1],
                                                          self.nhead * self.d_v)  # output: [bs x q_len x nhead * d_v]
        output = self.to_out(output)

        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, nhead, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // nhead
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x nhead x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x nhead x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x nhead x max_q_len x d_v]

        return output, attn_weights


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        #print('FACTORIZED {}'.format(factorized))
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cpu')
            # self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True).to('cuda:0')
            self.generator = self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
