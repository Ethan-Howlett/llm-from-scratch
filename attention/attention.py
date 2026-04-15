import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self. W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self. register_buffer('mask', torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, disable_causal_mask=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        if not disable_causal_mask:
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.disable_causal_mask = disable_causal_mask

        ####################################################
        # CACHING
        # self.max_seq_len = max_seq_len or context_length
        # self.window_size = window_size or self.max_seq_len
        # self.register_buffer("cache_k", None, persistent=False)
        # self.register_buffer("cache_v", None, persistent=False)
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        if use_cache:
            # to prevent self.ptr_cur became negative
            assert num_tokens <= self.window_size, (
                f"Input chunk size ({num_tokens}) exceeds KV cache window size ({self.window_size}). "
            )

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(
            b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)

        ####################################################
        # NEW
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros(b, self.num_heads,
                                           self.window_size, self.head_dim,
                                           device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # pointer to next free slot

            # if incoming chunk would overflow discard oldest tokens
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                # shift everything left by `overflow` (cheap view-copy)
                self.cache_k[:, :, :-overflow,
                             :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow,
                             :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_cur -= overflow  # pointer after shift

            self.cache_k[:, :, self.ptr_cur:self.ptr_cur +
                         num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur +
                         num_tokens, :] = values_new
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0  # keep pointer sane if you interleave modes
        ####################################################
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)

        if not self.disable_causal_mask:
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        ####################################################
        # NEW
        # K = attn_scores.size(-1)

        # if num_tokens == K:
        #     # No cache → use the pre‑baked triangular mask slice
        #     causal_mask = torch.triu(torch.ones(
        #         num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        # else:
        #     # Cached: need to offset the diagonal by (K − num_tokens)
        #     offset = K - num_tokens  # number of tokens already in cache before this chunk
        #     row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(
        #         1)  # (num_tokens, 1)
        #     col_idx = torch.arange(K, device=x.device).unsqueeze(
        #         0)           # (1, K)
        #     # True where j > i+offset
        #     causal_mask = row_idx + offset < col_idx
        # ####################################################

        # # Use the mask to fill attention scores
        # attn_scores.masked_fill_(
        #     causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
    ####################################################


def main():
    # Setup
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    d_in = inputs.shape[1]  # The input embedding size
    d_out = 2  # The output embedding size

    torch.manual_seed(789)
    self_attn = SelfAttention(d_in, d_out)
    print(self_attn(inputs))
    print("self_attn.shape:", self_attn(inputs).shape, "\n")

    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, dropout=0.1)
    context_vecs = ca(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape, "\n")

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(
        d_in, d_out, context_length, dropout=0.0, num_heads=2)

    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape, "\n")


if __name__ == "__main__":
    main()
