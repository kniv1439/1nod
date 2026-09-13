import math
import torch
import torch.nn.functional as F
from torch import nn

from utils.rotary import apply_rotary_emb
# 尝试导入flash_attn，导入失败时设置标志
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: flash_attn not available, falling back to standard attention")

from utils.rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadFlashDiff2(nn.Module):
    """
    DiffAttn implemented with FlashAttention, for packages that does not support different qk/v dimensions
    e.g., flash-attention (https://github.com/Dao-AILab/flash-attention)
    """
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
        head_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2 if head_dim is None else head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # 只有在flash_attn可用时才转换为fp16
        if FLASH_ATTN_AVAILABLE:
            self.half()
    
    # 标准注意力的替代实现
    def standard_attention(self, q, k, v, causal=True):
        """标准注意力机制的实现，作为flash_attn_func的替代，但保持相同的输入输出格式和功能"""
        # q: [batch_size, seq_len, num_heads, head_dim]
        # k: [batch_size, seq_len, num_heads, head_dim]
        # v: [batch_size, seq_len, num_heads, head_dim]
        
        bsz, q_len, num_heads, head_dim = q.shape
        k_len = k.shape[1]
        
        # 将q,k,v重塑为标准注意力的形状
        q = q.permute(0, 2, 1, 3)  # [bsz, num_heads, q_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [bsz, num_heads, k_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [bsz, num_heads, k_len, head_dim]
        
        # 使用分块计算来节省内存（简化，只对q进行分块）
        chunk_size = 32  # 减小每次处理的序列长度
        attn_output = torch.zeros(bsz, num_heads, q_len, head_dim, device=q.device, dtype=q.dtype)
        
        # 分块计算注意力
        for i in range(0, q_len, chunk_size):
            end_i = min(i + chunk_size, q_len)
            q_chunk = q[:, :, i:end_i]  # [bsz, num_heads, chunk_size, head_dim]
            
            # 计算注意力分数
            attn_scores = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scaling  # [bsz, num_heads, chunk_size, k_len]
            
            # 应用因果掩码（如果需要）
            if causal:
                causal_mask = torch.ones(end_i-i, k_len, device=q.device, dtype=torch.bool)
                for idx in range(end_i-i):
                    causal_mask[idx, :(i+idx+1)] = False  # 保留有效注意力区域
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, k_len]
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            # 应用softmax获取注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # 计算输出
            attn_output[:, :, i:end_i] = torch.matmul(attn_weights, v)
        
        # 恢复为与flash_attn_func相同的输出形状 [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3)
        
        return attn_output
    
    def forward(
        self,
        x,
        rel_pos,  # rel_pos 包含 cos 和 sin
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        # 只有在flash_attn可用时才转换为fp16
        if FLASH_ATTN_AVAILABLE:
            x = x.half()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, 2*self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, 2*self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        # 确保张量在 GPU 上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        # 将 cos 和 sin 转换为 fp16（如果使用flash_attn）
        cos, sin = rel_pos
        if FLASH_ATTN_AVAILABLE:
            cos = cos.half()
            sin = sin.half()

        # 应用旋转嵌入
        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # 选择使用flash_attn或标准注意力
        if FLASH_ATTN_AVAILABLE:
            q1 = q1.half()  # 转换为 fp16
            k1 = k1.half()
            v1 = v1.half()
            q2 = q2.half()
            k2 = k2.half()
            v2 = v2.half()
            
            attn11 = flash_attn_func(q1, k1, v1, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, causal=True)
            attn21 = flash_attn_func(q2, k2, v1, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, causal=True)
        else:
            # 使用标准注意力机制代替
            attn11 = self.standard_attention(q1, k1, v1, causal=True)
            attn12 = self.standard_attention(q1, k1, v2, causal=True)
            attn21 = self.standard_attention(q2, k2, v1, causal=True)
            attn22 = self.standard_attention(q2, k2, v2, causal=True)
        
        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn


class MultiheadFlashDiff2FullAttn(nn.Module):
    """
    DiffAttn with RoPE, but using FULL (non-causal) attention instead of causal.
    Suitable for link prediction where src and dst need to mutually attend to each other.
    Same interface as MultiheadFlashDiff2, but causal=False in all attention calls.
    """
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        num_kv_heads=None,
        head_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2 if head_dim is None else head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        if FLASH_ATTN_AVAILABLE:
            self.half()

    def _standard_attention_full(self, q, k, v):
        """Full (non-causal) attention, as fallback when flash_attn unavailable."""
        bsz, q_len, num_heads, head_dim = q.shape
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output.permute(0, 2, 1, 3)

    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if FLASH_ATTN_AVAILABLE:
            x = x.half()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        cos, sin = rel_pos
        if FLASH_ATTN_AVAILABLE:
            cos = cos.half()
            sin = sin.half()

        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        if FLASH_ATTN_AVAILABLE:
            q1, k1, v1 = q1.half(), k1.half(), v1.half()
            q2, k2, v2 = q2.half(), k2.half(), v2.half()
            attn11 = flash_attn_func(q1, k1, v1, causal=False)
            attn12 = flash_attn_func(q1, k1, v2, causal=False)
            attn21 = flash_attn_func(q2, k2, v1, causal=False)
            attn22 = flash_attn_func(q2, k2, v2, causal=False)
        else:
            attn11 = self._standard_attention_full(q1, k1, v1)
            attn12 = self._standard_attention_full(q1, k1, v2)
            attn21 = self._standard_attention_full(q2, k2, v1)
            attn22 = self._standard_attention_full(q2, k2, v2)

        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)
        return attn


class MultiheadFlashDiff2NoRoPE(nn.Module):
    """
    DiffAttn implemented with FlashAttention, but without RoPE (Rotary Position Embedding)
    This is a version for models that don't use RoPE
    """
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
        head_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2 if head_dim is None else head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # 不将整个模块转换为 FP16，保持权重在 FP32 以提高数值稳定性
        # 只在调用 flash_attn_func 时临时转换输入为 FP16
        # if FLASH_ATTN_AVAILABLE:
        #     self.half()
    
    # 标准注意力的替代实现
    def standard_attention(self, q, k, v, causal=True):
        """标准注意力机制的实现，作为flash_attn_func的替代，但保持相同的输入输出格式和功能"""
        # q: [batch_size, seq_len, num_heads, head_dim]
        # k: [batch_size, seq_len, num_heads, head_dim]
        # v: [batch_size, seq_len, num_heads, head_dim]
        
        bsz, q_len, num_heads, head_dim = q.shape
        k_len = k.shape[1]
        
        # 将q,k,v重塑为标准注意力的形状
        q = q.permute(0, 2, 1, 3)  # [bsz, num_heads, q_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [bsz, num_heads, k_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [bsz, num_heads, k_len, head_dim]
        
        # 使用分块计算来节省内存（简化，只对q进行分块）
        chunk_size = 32  # 减小每次处理的序列长度
        attn_output = torch.zeros(bsz, num_heads, q_len, head_dim, device=q.device, dtype=q.dtype)
        
        # 分块计算注意力
        for i in range(0, q_len, chunk_size):
            end_i = min(i + chunk_size, q_len)
            q_chunk = q[:, :, i:end_i]  # [bsz, num_heads, chunk_size, head_dim]
            
            # 计算注意力分数
            attn_scores = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scaling  # [bsz, num_heads, chunk_size, k_len]
            
            # 应用因果掩码（如果需要）
            if causal:
                causal_mask = torch.ones(end_i-i, k_len, device=q.device, dtype=torch.bool)
                for idx in range(end_i-i):
                    causal_mask[idx, :(i+idx+1)] = False  # 保留有效注意力区域
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, k_len]
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            # 应用softmax获取注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # 计算输出
            attn_output[:, :, i:end_i] = torch.matmul(attn_weights, v)
        
        # 恢复为与flash_attn_func相同的输出形状 [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3)
        
        return attn_output
    
    def forward(
        self,
        x,
        rel_pos=None,  # 保持参数兼容性，但不使用
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        # 保存原始 dtype
        original_dtype = x.dtype

        # 不在这里转换输入为 FP16，让投影层在 FP32 下计算（权重是 FP32）
        # 只在调用 flash_attn_func 时临时转换 q, k, v 为 FP16
        # if FLASH_ATTN_AVAILABLE:
        #     x = x.half()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, 2*self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, 2*self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        # 确保张量在 GPU 上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        # 不应用旋转嵌入，直接使用原始q和k
        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # 选择使用flash_attn或标准注意力
        if FLASH_ATTN_AVAILABLE:
            q1 = q1.half()  # 转换为 fp16
            k1 = k1.half()
            v1 = v1.half()
            q2 = q2.half()
            k2 = k2.half()
            v2 = v2.half()
            
            attn11 = flash_attn_func(q1, k1, v1, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, causal=True)
            attn21 = flash_attn_func(q2, k2, v1, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, causal=True)
        else:
            # 使用标准注意力机制代替
            attn11 = self.standard_attention(q1, k1, v1, causal=True)
            attn12 = self.standard_attention(q1, k1, v2, causal=True)
            attn21 = self.standard_attention(q2, k2, v1, causal=True)
            attn22 = self.standard_attention(q2, k2, v2, causal=True)
        
        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        # 保持 lambda 计算在 float32，避免 fp16 精度问题
        lambda_sum_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_sum_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        
        # 裁剪 lambda_sum 防止 exp 溢出
        lambda_sum_1 = torch.clamp(lambda_sum_1, min=-10.0, max=10.0)
        lambda_sum_2 = torch.clamp(lambda_sum_2, min=-10.0, max=10.0)
        
        lambda_1 = torch.exp(lambda_sum_1)  # 在 float32 下计算
        lambda_2 = torch.exp(lambda_sum_2)  # 在 float32 下计算
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).to(attn1.dtype)  # 转换为 attn1 的 dtype
        
        # 在 FP32 下进行关键计算，避免 FP16 溢出
        attn1_fp32 = attn1.float()
        attn2_fp32 = attn2.float()
        lambda_full_fp32 = lambda_full.float()
        attn_fp32 = attn1_fp32 - lambda_full_fp32 * attn2_fp32
        
        # 检查 NaN 和 Inf
        if torch.isnan(attn_fp32).any() or torch.isinf(attn_fp32).any():
            # 如果出现 NaN/Inf，使用更稳定的计算方式
            attn_fp32 = torch.clamp(attn_fp32, min=-50.0, max=50.0)
            attn_fp32 = torch.where(torch.isnan(attn_fp32) | torch.isinf(attn_fp32), 
                                   attn1_fp32, attn_fp32)
        
        attn = attn_fp32.to(attn1.dtype)  # 转换回原始 dtype

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        
        # 确保返回类型与输入 x 的原始类型一致
        if original_dtype != attn.dtype:
            attn = attn.to(dtype=original_dtype)
        
        return attn


def _reorder_for_rope_sin(x: torch.Tensor) -> torch.Tensor:
    """RoPE sin part reorder; x shape (..., seqlen, d_head), d_head even."""
    x_odd = x[..., ::2]
    x_even = x[..., 1::2]
    x_stacked = torch.stack([-x_even, x_odd], dim=-1)
    return x_stacked.flatten(start_dim=-2)


def _apply_rope_to_head(x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x. x: (batch, n_heads, seqlen, d_head); rope_cos/rope_sin: (seqlen, d_head)."""
    T = x.shape[2]
    x_sin = _reorder_for_rope_sin(x)
    return x * rope_cos[:T, :] + x_sin * rope_sin[:T, :]


class MultiheadFlashDiff2WithRoPE(nn.Module):
    """
    DiffAttn + RoPE: same as MultiheadFlashDiff2NoRoPE but applies RoPE to q1/k1/q2/k2
    before attention (same formulation as utils/rope_attention_layer.py).
    """
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        num_kv_heads=None,
        head_dim=None,
        max_pos_enc_len=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2 if head_dim is None else head_dim
        self.scaling = self.head_dim ** -0.5
        self.max_pos_enc_len = max_pos_enc_len

        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self._construct_rope_matrices(max_pos_enc_len)

    def _construct_rope_matrices(self, max_pos_enc_len: int):
        """RoPE sin/cos buffers, same as rope_attention_layer (eq. 34, base 1000)."""
        d_head = self.head_dim
        thetas = 1000 ** (-2.0 * torch.arange(1, d_head // 2 + 1).float() / d_head).repeat_interleave(2)
        positions = torch.arange(1, max_pos_enc_len + 1).float()
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", torch.sin(args))
        self.register_buffer("rope_cos", torch.cos(args))

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bsz, seq, n_heads, head_dim) -> apply RoPE -> (bsz, seq, n_heads, head_dim)."""
        T = x.shape[1]
        if T > self.max_pos_enc_len:
            new_len = T * 2
            self._construct_rope_matrices(new_len)
            self.max_pos_enc_len = new_len
            self.rope_cos = self.rope_cos.to(x.device)
            self.rope_sin = self.rope_sin.to(x.device)
        x_t = x.transpose(1, 2)  # (bsz, n_heads, seq, head_dim)
        x_rope = _apply_rope_to_head(x_t, self.rope_cos, self.rope_sin)
        return x_rope.transpose(1, 2)

    def standard_attention(self, q, k, v, causal=True):
        """Same as NoRoPE version."""
        bsz, q_len, num_heads, head_dim = q.shape
        k_len = k.shape[1]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        chunk_size = 32
        attn_output = torch.zeros(bsz, num_heads, q_len, head_dim, device=q.device, dtype=q.dtype)
        for i in range(0, q_len, chunk_size):
            end_i = min(i + chunk_size, q_len)
            q_chunk = q[:, :, i:end_i]
            attn_scores = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scaling
            if causal:
                causal_mask = torch.ones(end_i - i, k_len, device=q.device, dtype=torch.bool)
                for idx in range(end_i - i):
                    causal_mask[idx, :(i + idx + 1)] = False
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output[:, :, i:end_i] = torch.matmul(attn_weights, v)
        return attn_output.permute(0, 2, 1, 3)

    def forward(self, x, rel_pos=None, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        original_dtype = x.dtype

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, 2 * self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2, self.head_dim)

        device = x.device
        q, k, v = q.to(device), k.to(device), v.to(device)

        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # RoPE on q1, k1, q2, k2 (each bsz, seq, n_heads, head_dim)
        q1 = self._apply_rope(q1)
        k1 = self._apply_rope(k1)
        q2 = self._apply_rope(q2)
        k2 = self._apply_rope(k2)

        if FLASH_ATTN_AVAILABLE:
            q1, k1, v1 = q1.half(), k1.half(), v1.half()
            q2, k2, v2 = q2.half(), k2.half(), v2.half()
            attn11 = flash_attn_func(q1, k1, v1, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, causal=True)
            attn21 = flash_attn_func(q2, k2, v1, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, causal=True)
        else:
            attn11 = self.standard_attention(q1, k1, v1, causal=True)
            attn12 = self.standard_attention(q1, k1, v2, causal=True)
            attn21 = self.standard_attention(q2, k2, v1, causal=True)
            attn22 = self.standard_attention(q2, k2, v2, causal=True)

        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_sum_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_sum_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_sum_1 = torch.clamp(lambda_sum_1, min=-10.0, max=10.0)
        lambda_sum_2 = torch.clamp(lambda_sum_2, min=-10.0, max=10.0)
        lambda_1 = torch.exp(lambda_sum_1)
        lambda_2 = torch.exp(lambda_sum_2)
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).to(attn1.dtype)

        attn1_fp32 = attn1.float()
        attn2_fp32 = attn2.float()
        lambda_full_fp32 = lambda_full.float()
        attn_fp32 = attn1_fp32 - lambda_full_fp32 * attn2_fp32
        if torch.isnan(attn_fp32).any() or torch.isinf(attn_fp32).any():
            attn_fp32 = torch.clamp(attn_fp32, min=-50.0, max=50.0)
            attn_fp32 = torch.where(torch.isnan(attn_fp32) | torch.isinf(attn_fp32), attn1_fp32, attn_fp32)
        attn = attn_fp32.to(attn1.dtype)

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)
        if original_dtype != attn.dtype:
            attn = attn.to(dtype=original_dtype)
        return attn


class MultiheadStandardAttentionWithRoPE(nn.Module):
    """
    标准多头注意力机制，但结合了RoPE（Rotary Position Embedding）
    用于替换DIFF Attention但保留RoPE功能
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        head_dim=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 优先使用传入的head_dim，否则计算
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # 只有在flash_attn可用时才转换为fp16
        if FLASH_ATTN_AVAILABLE:
            self.half()
    
    def forward(
        self,
        x,
        rel_pos,  # rel_pos 包含 cos 和 sin
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        # 只有在flash_attn可用时才转换为fp16
        if FLASH_ATTN_AVAILABLE:
            x = x.half()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)

        # 确保张量在 GPU 上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        # 将 cos 和 sin 转换为 fp16（如果使用flash_attn）
        cos, sin = rel_pos
        if FLASH_ATTN_AVAILABLE:
            cos = cos.half()
            sin = sin.half()

        # 应用旋转嵌入
        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)

        # 重塑为注意力计算的标准格式
        q = q.transpose(1, 2)  # (bsz, num_heads, tgt_len, head_dim)
        k = k.transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)
        v = v.transpose(1, 2)  # (bsz, num_heads, src_len, head_dim)

        # 选择使用flash_attn或标准注意力
        if FLASH_ATTN_AVAILABLE:
            # 使用flash attention
            attn_output = flash_attn_func(q, k, v, causal=True)
        else:
            # 使用标准注意力机制
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            
            # 应用因果掩码
            if attn_mask is None:
                causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=q.device, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, tgt_len, num_heads, head_dim)
        attn_output = attn_output.view(bsz, tgt_len, embed_dim)
        
        attn_output = self.out_proj(attn_output)
        return attn_output


class StandardMultiheadFlashAttention(nn.Module):
    """
    标准多头自注意力：有 FlashAttention 时用 flash_attn_func，否则用 PyTorch 标准实现。
    接口与 nn.MultiheadAttention 一致，便于在 DyGFormerTW 等模型中替换并支持无 Flash 时回退。
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _standard_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """标准注意力，q/k/v: (batch, seq_len, num_heads, head_dim)，返回 (batch, seq_len, num_heads, head_dim)。"""
        bsz, seq_len, num_heads, head_dim = q.shape
        q = q.transpose(1, 2)   # (bsz, num_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # (bsz, num_heads, seq, head_dim)
        return out.transpose(1, 2)  # (bsz, seq, num_heads, head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None, need_weights: bool = False, attn_mask: torch.Tensor = None):
        key = key if key is not None else query
        value = value if value is not None else query
        # query/key/value: (seq_len, batch_size, embed_dim)
        seq_len, bsz, _ = query.shape
        x = query.transpose(0, 1)  # (batch, seq_len, embed_dim)
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key.transpose(0, 1)).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value.transpose(0, 1)).view(bsz, seq_len, self.num_heads, self.head_dim)
        if FLASH_ATTN_AVAILABLE:
            orig_dtype = q.dtype
            q, k, v = q.half(), k.half(), v.half()
            out = flash_attn_func(q, k, v, causal=False)
            out = out.to(orig_dtype)
        else:
            out = self._standard_attn(q, k, v)
        out = out.reshape(bsz, seq_len, self.embed_dim)
        out = self.out_proj(self.dropout(out))
        return out.transpose(0, 1), None  # (seq_len, batch_size, embed_dim), None
