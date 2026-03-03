import torch

from attention import SelfAttention, CausalAttention, MultiHeadAttention


def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> None:
    m = float((a - b).abs().max().item()) if a.numel() else 0.0
    assert m <= tol, f"Expected close tensors, max abs diff={m}, tol={tol}"


def test_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    # y = SimpleSelfAttention().forward(x)
    # assert y.shape == x.shape

    y = CausalAttention(d_in=16, d_out=16, context_length=5, dropout=0.0)(x)
    assert y.shape == x.shape

    y = MultiHeadAttention(d_in=16, d_out=16, context_length=5, dropout=0.0, num_heads=4)(x)
    assert y.shape == x.shape


def test_causal_means_future_does_not_change_past():
    torch.manual_seed(0)
    b, t, d, h = 2, 6, 32, 4

    x1 = torch.randn(b, t, d)
    x2 = x1.clone()
    x2[:, -1, :] += 10.0  # change the FUTURE token a lot

    # If causal masking is correct, earlier outputs should not change.
    attn = MultiHeadAttention(d_in=d, d_out=d, context_length=t, dropout=0.0, num_heads=h)
    y1 = attn(x1)
    y2 = attn(x2)

    # Compare all positions except the last one
    assert_close(y1[:, :-1, :], y2[:, :-1, :])


if __name__ == "__main__":
    test_shapes()
    test_causal_means_future_does_not_change_past()
    print("✅ Quick tests passed")