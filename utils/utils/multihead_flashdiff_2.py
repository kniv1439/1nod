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
