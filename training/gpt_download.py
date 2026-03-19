import os
import requests
import torch
import tiktoken

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpt.gpt_model import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

if CHOOSE_MODEL == "gpt2-small (124M)":
    file_name = "gpt2-small-124M.pth"
elif CHOOSE_MODEL == "gpt2-medium (355M)":
    file_name = "gpt2-medium-355M.pth"
elif CHOOSE_MODEL == "gpt2-large (774M)":
    file_name = "gpt2-large-774M.pth"
elif CHOOSE_MODEL == "gpt2-xl (1558M)":
    file_name = "gpt2-xl-1558M.pth"
else:
    raise ValueError(f"Invalid model: {CHOOSE_MODEL}")

url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

if not os.path.exists(file_name):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {file_name}")
else:
    print(f"File {file_name} already exists")


gpt = GPTModel(BASE_CONFIG)
gpt.load_state_dict(torch.load(file_name, weights_only=True))
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
