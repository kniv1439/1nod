"""CARR dual-branch attention: Token (L×L) + Channel (EMA→DP→C×C→broadcast)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotary import apply_rotary_emb


def _build_ema_matrix(max_len: int, alpha: float) -> torch.Tensor:
    ema_matrix = torch.zeros(max_len, max_len)
    ema_matrix[0, 0] = 1.0
    for i in range(1, max_len):
        for j in range(i):
            ema_matrix[i, j] = ema_matrix[i - 1, j] * (1 - alpha)
        ema_matrix[i, i] = alpha
    return ema_matrix


class CARRTokenChannelAttentionEncoder(nn.Module):
    """
    Post-LN dual-branch encoder (CARRDyG default):
    - Token branch: L×L self-attention on flattened channels.
    - Channel branch: EMA(L) -> DP(L->R) -> C×C on R -> broadcast R->L.
    """

    def __init__(self, channel_dim: int, num_channels: int = 4, num_heads: int = 4,
                 dp_rank: int = 8, alpha: float = 0.9, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        assert channel_dim % num_heads == 0, 'channel_embedding_dim must be divisible by num_heads'

        self.channel_dim = channel_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.total_dim = num_channels * channel_dim
        self.dp_rank = dp_rank
        self.alpha = alpha

        self.token_qkv = nn.Linear(self.total_dim, self.total_dim * 3, bias=True)
        self.channel_qkv = nn.Linear(channel_dim, channel_dim * 3, bias=True)
        self.dp_k = nn.Linear(self.head_dim, dp_rank)
        self.dp_v = nn.Linear(self.head_dim, dp_rank)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.total_dim)
        self.norm2 = nn.LayerNorm(self.total_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.total_dim, 4 * self.total_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.total_dim, self.total_dim),
        )

        self.register_buffer('ema_matrix', _build_ema_matrix(max_len, alpha))

    def _expand_ema_matrix(self, length: int):
        if length <= self.ema_matrix.shape[0]:
            return
        device = self.ema_matrix.device
        old_size = self.ema_matrix.shape[0]
        new_ema = torch.zeros(length, length, device=device)
        new_ema[:old_size, :old_size] = self.ema_matrix
        alpha = self.alpha
        for i in range(old_size, length):
            for j in range(i):
                new_ema[i, j] = new_ema[i - 1, j] * (1 - alpha)
            new_ema[i, i] = alpha
        self.ema_matrix = new_ema

    def ema_along_token(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        self._expand_ema_matrix(length)
        ema = self.ema_matrix[:length, :length]
        return torch.einsum('bghcd,lg->blhcd', x, ema)

    def compress_along_L(self, x: torch.Tensor, mlp: nn.Linear):
        weight = mlp(x)
        weight = F.softmax(weight, dim=1)
        summary = torch.einsum('blhcd,blhcr->brhcd', x, weight)
        return summary, weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_channels, channel_dim = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        x_flat = x.reshape(batch_size, seq_len, self.total_dim)

        token_qkv = self.token_qkv(x_flat).reshape(batch_size, seq_len, 3, num_heads, self.total_dim // num_heads)
        token_qkv = token_qkv.permute(2, 0, 3, 1, 4)
        q_t, k_t, v_t = token_qkv[0], token_qkv[1], token_qkv[2]

        token_score = torch.einsum('bhld,bhmd->bhlm', q_t, k_t) / (head_dim ** 0.5)
        token_attn = self.attn_dropout(F.softmax(token_score, dim=-1))
        token_out = torch.einsum('bhlm,bhmd->bhld', token_attn, v_t)
        token_out = token_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels, channel_dim)

        channel_qkv = self.channel_qkv(x).reshape(batch_size, seq_len, num_channels, 3, num_heads, head_dim)
        channel_qkv = channel_qkv.permute(3, 0, 1, 4, 2, 5)
        q_c, k_c, v_c = channel_qkv[0], channel_qkv[1], channel_qkv[2]

        q_c = self.ema_along_token(q_c)
        k_c = self.ema_along_token(k_c)

        k_r, _ = self.compress_along_L(k_c, self.dp_k)
        v_r, w_v = self.compress_along_L(v_c, self.dp_v)
        q_r, _ = self.compress_along_L(q_c, self.dp_k)

        scale = head_dim ** 0.5
        channel_score = torch.einsum('brhcd,brhed->brhce', q_r, k_r) / scale
        channel_attn = self.attn_dropout(F.softmax(channel_score, dim=-1))
        out_r = torch.einsum('brhce,brhed->brhcd', channel_attn, v_r)

        channel_out = torch.einsum('blhcr,brhcd->blhcd', w_v, out_r)
        channel_out = channel_out.permute(0, 1, 3, 2, 4).reshape(batch_size, seq_len, num_channels, channel_dim)

        out = token_out + channel_out
        out_flat = out.reshape(batch_size, seq_len, self.total_dim)
        x_flat = x_flat + out_flat
        x_flat = self.norm1(x_flat)
        x_flat = x_flat + self.ff(x_flat)
        x_flat = self.norm2(x_flat)

        return x_flat.reshape(batch_size, seq_len, num_channels, channel_dim)


class CARRTokenChannelAttentionEncoderPreLN(nn.Module):
    """
    Pre-LN CARR dual-branch encoder:
    - Token branch: L×L MHA on flattened channels (optional RoPE on Q/K only).
    - Channel branch: EMA(L) -> DP(L->R) -> C×C on R -> broadcast R->L.
    """

    def __init__(self, channel_dim: int, num_channels: int = 4, num_heads: int = 4,
                 dp_rank: int = 8, alpha: float = 0.9, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        assert channel_dim % num_heads == 0, 'channel_embedding_dim must be divisible by num_heads'

        self.channel_dim = channel_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.token_head_dim = (num_channels * channel_dim) // num_heads
        self.total_dim = num_channels * channel_dim
        self.dp_rank = dp_rank
        self.alpha = alpha

        self.token_qkv = nn.Linear(self.total_dim, self.total_dim * 3, bias=True)
        self.channel_qkv = nn.Linear(channel_dim, channel_dim * 3, bias=True)
        self.dp_k = nn.Linear(self.head_dim, dp_rank)
        self.dp_v = nn.Linear(self.head_dim, dp_rank)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.total_dim)
        self.norm2 = nn.LayerNorm(self.total_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.total_dim, 4 * self.total_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.total_dim, self.total_dim),
        )

        self.register_buffer('ema_matrix', _build_ema_matrix(max_len, alpha))

    def _expand_ema_matrix(self, length: int):
        if length <= self.ema_matrix.shape[0]:
            return
        device = self.ema_matrix.device
        old_size = self.ema_matrix.shape[0]
        new_ema = torch.zeros(length, length, device=device)
        new_ema[:old_size, :old_size] = self.ema_matrix
        alpha = self.alpha
        for i in range(old_size, length):
            for j in range(i):
                new_ema[i, j] = new_ema[i - 1, j] * (1 - alpha)
            new_ema[i, i] = alpha
        self.ema_matrix = new_ema

    def ema_along_token(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        self._expand_ema_matrix(length)
        ema = self.ema_matrix[:length, :length]
        return torch.einsum('bghcd,lg->blhcd', x, ema)

    def compress_along_L(self, x: torch.Tensor, mlp: nn.Linear):
        weight = mlp(x)
        weight = F.softmax(weight, dim=1)
        summary = torch.einsum('blhcd,blhcr->brhcd', x, weight)
        return summary, weight

    def _apply_rope_token_qk(self, q_t, k_t, cos, sin):
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)
        return q.transpose(1, 2), k.transpose(1, 2)

    def forward(self, x: torch.Tensor, cos=None, sin=None, rotary_dim=None) -> torch.Tensor:
        batch_size, seq_len, num_channels, channel_dim = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        x_flat = x.reshape(batch_size, seq_len, self.total_dim)
        x_norm = self.norm1(x_flat)
        x_norm_4d = x_norm.reshape(batch_size, seq_len, num_channels, channel_dim)

        token_qkv = self.token_qkv(x_norm).reshape(batch_size, seq_len, 3, num_heads, self.token_head_dim)
        token_qkv = token_qkv.permute(2, 0, 3, 1, 4)
        q_t, k_t, v_t = token_qkv[0], token_qkv[1], token_qkv[2]

        if cos is not None and sin is not None and rotary_dim is not None:
            q_t, k_t = self._apply_rope_token_qk(q_t, k_t, cos, sin)

        token_score = torch.einsum('bhld,bhmd->bhlm', q_t, k_t) / (head_dim ** 0.5)
        token_attn = self.attn_dropout(F.softmax(token_score, dim=-1))
        token_out = torch.einsum('bhlm,bhmd->bhld', token_attn, v_t)
        token_out = token_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels, channel_dim)

        channel_qkv = self.channel_qkv(x_norm_4d).reshape(batch_size, seq_len, num_channels, 3, num_heads, head_dim)
        channel_qkv = channel_qkv.permute(3, 0, 1, 4, 2, 5)
        q_c, k_c, v_c = channel_qkv[0], channel_qkv[1], channel_qkv[2]

        q_c = self.ema_along_token(q_c)
        k_c = self.ema_along_token(k_c)

        k_r, _ = self.compress_along_L(k_c, self.dp_k)
        v_r, w_v = self.compress_along_L(v_c, self.dp_v)
        q_r, _ = self.compress_along_L(q_c, self.dp_k)

        scale = head_dim ** 0.5
        channel_score = torch.einsum('brhcd,brhed->brhce', q_r, k_r) / scale
        channel_attn = self.attn_dropout(F.softmax(channel_score, dim=-1))
        out_r = torch.einsum('brhce,brhed->brhcd', channel_attn, v_r)

        channel_out = torch.einsum('blhcr,brhcd->blhcd', w_v, out_r)
        channel_out = channel_out.permute(0, 1, 3, 2, 4).reshape(batch_size, seq_len, num_channels, channel_dim)

        out_flat = (token_out + channel_out).reshape(batch_size, seq_len, self.total_dim)
        x_flat = x_flat + self.out_dropout(out_flat)
        x_flat = x_flat + self.out_dropout(self.ff(self.norm2(x_flat)))

        return x_flat.reshape(batch_size, seq_len, num_channels, channel_dim)
