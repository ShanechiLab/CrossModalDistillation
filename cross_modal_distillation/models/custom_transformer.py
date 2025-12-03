import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from einops import rearrange, repeat
from flash_attn import flash_attn_func, flash_attn_varlen_func

from cross_modal_distillation.utility.utils import get_activation_function


def get_attention_module(attention_module_name):
    if attention_module_name == "SelfAttention":
        return SelfAttention
    elif attention_module_name == "RotarySelfAttention":
        return RotarySelfAttention
    else:
        raise NotImplementedError(
            f"Requested attention module {attention_module_name} is not implemented. Available options are 'SelfAttention' and 'RotarySelfAttention'."
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, base=10000, max_pos=1024, device=None):
        super().__init__()

        self.d_head = d_head
        self.max_pos = max_pos
        self.device = device

        inv_freq = 1 / (
            base
            ** (torch.arange(0, self.d_head, 2).float().to(self.device) / self.d_head)
        )  # (d//2, )
        self.register_buffer("inv_freq", inv_freq)
        self.build_cache()

    def build_cache(self):
        t = torch.arange(
            self.max_pos,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (self.max_pos, d//2)

        emb = torch.cat((freqs, freqs), dim=-1)  # (self.max_pos, d)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False
        )  # (self.max_pos, d)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False
        )  # (self.max_pos, d)

    def forward(self, position_ids):
        """Returns the rotation matrices"""
        cos = self.cos_cached[position_ids].unsqueeze(2)  # [B, seq_len, 1, d_head]
        sin = self.sin_cached[position_ids].unsqueeze(2)  # [B, seq_len, 1, d_head]
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies the rotation matrices on query and key tensors
    q, k: B x seq_len x num_head x d_head
    """
    q_embed = (q * cos.to(q)) + (
        rotate_half(q) * sin.to(q)
    )  # [B, seq_len, num_heads, d_head]
    k_embed = (k * cos.to(k)) + (
        rotate_half(k) * sin.to(k)
    )  # [B, seq_len, num_heads, d_head]
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, d_hidden, eps=1e-6):
        """
        From https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/llama/modeling_llama.py#L74
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_hidden))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_hidden,
        num_heads=8,
        dropout=0.1,
        use_flash_attention=False,
        use_sdpa_attention=False,
        **kwargs,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_head = self.d_hidden // self.num_heads
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.use_flash_attention = use_flash_attention
        self.use_sdpa_attention = use_sdpa_attention

        assert (
            self.d_hidden % self.num_heads == 0
        ), f"Number of attention heads: {self.num_heads} must divide embedding dimension: {self.d_hidden}."

        self.qkv_proj = nn.Linear(self.d_hidden, 3 * self.d_hidden, bias=True)
        self.o_proj = nn.Linear(self.d_hidden, self.d_hidden, bias=True)

    def get_qkv(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = rearrange(q, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        k = rearrange(k, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        v = rearrange(v, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        return q, k, v

    def get_attention_out(self, q, k, v, seq_lens=None):
        if self.use_flash_attention:
            attention_out = self.get_flash_attention_out(q, k, v, seq_lens)
        elif self.use_sdpa_attention:
            attention_out = self.get_sdpa_attention_out(q, k, v, seq_lens)
        else:
            attention_out = self.get_memory_efficient_attention_out(q, k, v, seq_lens)

        attention_out = self.dropout(attention_out)
        attention_out = rearrange(attention_out, "b n h d_h -> b n (h d_h)")
        out = self.o_proj(attention_out)
        return out

    def get_sdpa_attention_out(self, q, k, v, seq_lens=None):
        if seq_lens is not None:
            assert seq_lens is None, "seq_lens must be None for PyTorch sdpa attention."
        q = q.permute(0, 2, 1, 3)  # b h n d_h
        k = k.permute(0, 2, 1, 3)  # b h n d_h
        v = v.permute(0, 2, 1, 3)  # b h n d_h

        with torch.backends.cuda.sdp_kernel():
            attention_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_rate
            )
        return attention_out.permute(0, 2, 1, 3)  # # b n h d_h

    def get_memory_efficient_attention_out(self, q, k, v, seq_lens=None):
        if seq_lens is not None and q.shape[0] == 1:
            attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        else:
            attn_bias = None

        assert q.shape[-2:] == (
            self.num_heads,
            self.d_head,
        ), f"Memory efficient attention expects inputs of shape (1 (or B), n, num_heads, d_head): (1 (or B) x ... x {self.num_heads} x {self.d_head}) but got {q.shape}!"

        # expects 1 n h d_h input assuming variable length sequences
        attention_out = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=0,
            attn_bias=attn_bias,
        )
        return attention_out

    def get_flash_attention_out(self, q, k, v, seq_lens=None):
        if seq_lens is not None and q.shape[0] == 1:
            assert q.shape[-2:] == (
                self.num_heads,
                self.d_head,
            ), f"Flash attention expects inputs of shape (1 (or B), n, num_heads, d_head): (1 (or B) x ... x {self.num_heads} x {self.d_head}) but got {q.shape}!"

            # Variable length sequences
            q = q.squeeze(dim=0)
            k = k.squeeze(dim=0)
            v = v.squeeze(dim=0)

            if isinstance(seq_lens, torch.Tensor):
                seq_lens = seq_lens.tolist()

            max_len = max(seq_lens)
            seq_lens_tensor = torch.tensor([0] + seq_lens, dtype=torch.int32).to(
                q.device
            )
            cu_seqlens = torch.cumsum(seq_lens_tensor, dim=0).int()

            attention_out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_k=max_len,
                max_seqlen_q=max_len,
            )
            attention_out = attention_out.unsqueeze(dim=0)
        else:
            attention_out = flash_attn_func(
                q=q,
                k=k,
                v=v,
            )
        return attention_out

    def forward(self, x, seq_lens=None, **kwargs):
        if seq_lens is None and x.shape[0] == 1:
            raise ValueError(
                f"'seq_lens' for memory efficient attention with variable length sequences (x.shape[0] == 1) must be non-None."
            )
        q, k, v = self.get_qkv(x)
        out = self.get_attention_out(q, k, v, seq_lens)
        return out


class RotarySelfAttention(SelfAttention):
    """
    Batch first
    """

    def __init__(
        self,
        d_hidden,
        num_heads=8,
        max_pos=1024,
        dropout=0.1,
        use_flash_attention=False,
        use_sdpa_attention=False,
        **kwargs,
    ):
        super().__init__(
            d_hidden=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_sdpa_attention=use_sdpa_attention,
        )
        self.max_pos = max_pos
        self.rotary_emb = RotaryEmbedding(self.d_head, max_pos=self.max_pos)

    def forward(self, x, position_ids=None, seq_lens=None):
        if seq_lens is None and x.shape[0] == 1:
            raise ValueError(
                f"'seq_lens' for memory efficient attention with variable length sequences (x.shape[0] == 1) must be non-None."
            )

        if position_ids is None:
            if x.shape[0] == 1:
                # means we are operating with variable length sequences
                position_ids = [torch.arange(seq_len_) for seq_len_ in seq_lens]
                position_ids = (
                    torch.cat(position_ids).unsqueeze(dim=0).int().to(x.device)
                )
            else:
                position_ids = (
                    repeat(torch.arange(x.shape[1]), "n -> b n", b=x.shape[0])
                    .int()
                    .to(x.device)
                )

        q, k, v = self.get_qkv(x)

        cos, sin = self.rotary_emb(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        v = v.to(q)

        out = self.get_attention_out(q, k, v, seq_lens)
        return out


class GatedTransformerMLP(nn.Module):
    """From LlamaMLP"""

    def __init__(self, d_hidden, mlp_ratio=4, activation="silu", dropout=0.1):
        super().__init__()
        d_feedforward = mlp_ratio * d_hidden
        self.gate_proj = nn.Linear(d_hidden, d_feedforward, bias=True)
        self.down_proj = nn.Linear(d_feedforward, d_hidden, bias=True)
        self.up_proj = nn.Linear(d_hidden, d_feedforward, bias=True)
        self.activation_fn = get_activation_function(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout2(self.down_proj(x))


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden,
        mlp_ratio=4,
        norm="rmsnorm",
        norm_eps=1e-6,
        activation="silu",
        num_heads=8,
        dropout=0.1,
        norm_first=True,
        attention_module_name="RotarySelfAttention",
        use_flash_attention=False,
        use_sdpa_attention=False,
        **attention_module_kwargs,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.norm_first = norm_first

        attention_cls = get_attention_module(attention_module_name)

        self.attention = attention_cls(
            d_hidden=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_sdpa_attention=use_sdpa_attention,
            **attention_module_kwargs,
        )
        self.mlp = GatedTransformerMLP(
            d_hidden=d_hidden,
            mlp_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        if norm.lower() == "rmsnorm":
            self.norm1 = RMSNorm(d_hidden, eps=norm_eps)
            self.norm2 = RMSNorm(d_hidden, eps=norm_eps)
        elif norm.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(d_hidden, eps=norm_eps)
            self.norm2 = nn.LayerNorm(d_hidden, eps=norm_eps)
        else:
            raise NotImplementedError(
                f"Requested normalization layer: {norm} is not implemented for RotaryTransformerEncoderLayer."
            )

    def forward(self, x, position_ids=None, seq_lens=None):
        residual = x
        if self.norm_first:
            # Attention
            x = self.norm1(x)
            x = self.attention(x=x, position_ids=position_ids, seq_lens=seq_lens)
            x = self.dropout(x)
            x = residual + x

            # Gated MLP
            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + x
        else:
            # Attention
            x = self.attention(
                x=x,
                position_ids=position_ids,
                seq_lens=seq_lens,
            )
            x = self.dropout(x)
            x = residual + x
            x = self.norm1(x)

            # Gated MLP
            residual = x
            x = self.mlp(x)
            x = residual + x
            x = self.norm2(x)
        return x


class CustomTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_hidden,
        mlp_ratio=4,
        norm="rmsnorm",
        norm_eps=1e-6,
        activation="gelu",
        num_heads=8,
        dropout=0.1,
        norm_first=True,
        attention_module_name="RotarySelfAttention",
        use_flash_attention=False,
        use_sdpa_attention=False,
        use_final_norm=True,
        return_states=False,
        **attention_module_kwargs,
    ):
        self.d_hidden = d_hidden
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    d_hidden=d_hidden,
                    mlp_ratio=mlp_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    activation=activation,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm_first=norm_first,
                    attention_module_name=attention_module_name,
                    use_flash_attention=use_flash_attention,
                    use_sdpa_attention=use_sdpa_attention,
                    **attention_module_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        if use_final_norm:
            if norm.lower() == "rmsnorm":
                self.norm = RMSNorm(d_hidden, eps=norm_eps)
            elif norm.lower() == "layernorm":
                self.norm = nn.LayerNorm(d_hidden, eps=norm_eps)
        else:
            self.norm = None

        self.return_states = return_states

    def forward(self, x, position_ids=None, seq_lens=None, **kwargs):
        states = []
        for layer in self.layers:
            x = layer(
                x=x,
                position_ids=position_ids,
                seq_lens=seq_lens,
            )
            if self.return_states:
                states.append(x)
        if self.norm:
            x = self.norm(x)

        return states, x
