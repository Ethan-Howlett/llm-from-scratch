import tokenization
import attention
import tiktoken
import torch.nn as nn
import torch


def main():
    with open("sample_texts/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = 50257
    output_dim = 256
    max_len = 1024
    context_length = max_len

    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    position_embedding_layer = nn.Embedding(context_length, output_dim)

    max_length = 4
    dataloader = tokenization.create_dataloader(
        raw_text, batch_size=8, max_length=max_length, stride=max_length)

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        position_embeddings = position_embedding_layer(
            torch.arange(max_length))
        input_embeddings = token_embeddings + position_embeddings
        break
    print(input_embeddings.shape)

    torch.manual_seed(123)
    context_length = max_length
    d_in = output_dim
    d_out = d_in

    mha = attention.MultiHeadAttention(
        d_in, d_out, context_length, dropout=0.0, num_heads=2)

    batch = input_embeddings
    context_vecs = mha(batch)
    print("context_vecs.shape:", context_vecs.shape)


if __name__ == "__main__":
    main()
