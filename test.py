import torch
from gpt_model import LayerNorm, FeedForward, TransformerBlock, GPTModel, generate_text_simple, generate_text_simple_cached

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,    # 0 for testing (deterministic)
    "qkv_bias": False,
    "kv_window_size": 1024   # NEW: KV cache window size
}


def test_layer_norm():
    ln = LayerNorm(768)
    x = torch.randn(2, 5, 768)
    y = ln(x)
    assert y.shape == (2, 5, 768), f"Expected (2,5,768), got {y.shape}"
    # After normalization the mean should be ~0
    assert y.mean(dim=-1).abs().max() < 0.01


def test_feedforward():
    cfg = GPT_CONFIG_124M
    ff = FeedForward(cfg)
    x = torch.randn(2, 5, 768)
    y = ff(x)
    assert y.shape == (2, 5, 768), f"Expected (2,5,768), got {y.shape}"


def test_transformer_block():
    cfg = GPT_CONFIG_124M
    block = TransformerBlock(cfg)
    x = torch.randn(2, 5, 768)
    y = block(x)
    assert y.shape == (2, 5, 768), f"Expected (2,5,768), got {y.shape}"


def test_gpt_model():
    cfg = GPT_CONFIG_124M
    model = GPTModel(cfg)
    idx = torch.randint(0, 50257, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, 50257), f"Expected (2,10,50257), got {logits.shape}"

    # Parameter count (before weight tying)
    total = sum(p.numel() for p in model.parameters())
    assert 160_000_000 < total < 170_000_000, f"Expected ~163M params, got {total:,}"


def test_generation():
    torch.manual_seed(42)
    cfg = GPT_CONFIG_124M.copy()
    cfg["n_layers"] = 2        # small model for speed
    cfg["context_length"] = 64
    model = GPTModel(cfg)
    model.eval()

    idx = torch.tensor([[50256]])  # BOS token
    out = generate_text_simple_cached(model, idx, max_new_tokens=5, context_size=64)
    assert out.shape == (1, 6), f"Expected (1,6), got {out.shape}"
    assert out[0, 0].item() == 50256  # first token preserved


if __name__ == "__main__":
    test_layer_norm()
    test_feedforward()
    test_transformer_block()
    test_gpt_model()
    test_generation()
    print("✅ All quick tests passed")