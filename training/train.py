import time
import tiktoken
import torch
import requests
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention.tokenization import create_dataloader
from gpt.gpt_model import GPTModel, generate_text_simple_cached


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple_cached(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # Invisible plot for aligning ticks
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()
        # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"Uses tensor cores: {torch.cuda.is_available()}")
    print()

    ##############################
    # Load text data
    ##############################
    SAMPLE_TEXT_FOLDER = "sample_texts"
    tokenizer = tiktoken.get_encoding("gpt2")

    if not os.path.exists(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt")):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        with open(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt"), "wb") as f:
            f.write(response.content)
        print("Downloaded the-verdict.txt")
    with open(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt"), "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=4,
        tokenizer=tokenizer,
    )
    val_loader = create_dataloader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=4,
        tokenizer=tokenizer,
    )

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=False
    )
    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        # Shortened context length (orig: 1024, 256 for testing)
        "context_length": 256,
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "kv_window_size": 1024   # NEW: KV cache window size
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 4,
        "weight_decay": 0.1,
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    eval_indices = torch.arange(len(train_losses), dtype=torch.float32)
    plot_losses(eval_indices, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
