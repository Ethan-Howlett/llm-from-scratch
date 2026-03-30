"""
Download the GPT model.
"""

import os
import copy
import json
import numpy as np
import torch
import requests
from tqdm import tqdm

import sys
from pathlib import Path
PRETRAINED_MODELS_DIR = Path(__file__).resolve().parent / "pre-trained-models"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpt.gpt_model import GPTModel


def download_and_load_gpt2(model_size, models_dir=None, base_config=None) -> GPTModel:
    if models_dir is None:
        models_dir = os.fspath(PRETRAINED_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    # Validate model size
    allowed_sizes = ("gpt2-small (124M)", "gpt2-medium (355M)",
                     "gpt2-large (774M)", "gpt2-xl (1558M)")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in allowed sizes: {allowed_sizes}")

    # Define paths
    model_size = model_size.replace(" ", "-").replace("(", "").replace(")", "")
    model_dir = os.path.join(models_dir, f"{model_size}.pth")
    base_url = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/"

    download_file(base_url + f"{model_size}.pth", model_dir)
    return load_gpt2(model_size, models_dir)


def download_file(url, destination):
    try:
        # Send a GET request to download the file in streaming mode
        response = requests.get(url, stream=True)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        block_size = 1024 * 1024 * 2  # 2 Megabytes

        # Initialize the progress bar with total file size
        progress_bar_description = url.split(
            "/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file
        return True
    except (requests.exceptions.RequestException, IOError) as e:
        error_message = (
            f"Failed to download ({url}) to {destination}."
            "\nCheck your internet connection or the file availability."
            f"\nError: {e}"
        )
        raise requests.exceptions.RequestException(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_gpt2(model_size, models_dir=None, base_config=None):
    if models_dir is None:
        models_dir = os.fspath(PRETRAINED_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    if base_config is None:
        BASE_CONFIG = {
            "vocab_size": 50257,    # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,       # Dropout rate
            "qkv_bias": True        # Query-key-value bias
        }
    else:
        BASE_CONFIG = copy.deepcopy(base_config)

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Get weights from model file
    file_name = f"{model_size.replace(' ', '-').replace('(', '').replace(')', '')}.pth"
    model_path = os.path.join(models_dir, file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}...")

    # Update the base config with the model config
    BASE_CONFIG.update(model_configs[model_size])
    print(f"Model config: {BASE_CONFIG}")

    # Load the model weights
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Model loaded successfully")
    return model

# CHOOSE_MODEL = "gpt2-medium (355M)"
# BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# if CHOOSE_MODEL == "gpt2-small (124M)":
#     file_name = "gpt2-small-124M.pth"
# elif CHOOSE_MODEL == "gpt2-medium (355M)":
#     file_name = "gpt2-medium-355M.pth"
# elif CHOOSE_MODEL == "gpt2-large (774M)":
#     file_name = "gpt2-large-774M.pth"
# elif CHOOSE_MODEL == "gpt2-xl (1558M)":
#     file_name = "gpt2-xl-1558M.pth"
# else:
#     raise ValueError(f"Invalid model: {CHOOSE_MODEL}")

# url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

# if not os.path.exists(file_name):
#     response = requests.get(url, timeout=60)
#     response.raise_for_status()
#     with open(file_name, "wb") as f:
#         f.write(response.content)
#     print(f"Downloaded to {file_name}")
# else:
#     print(f"File {file_name} already exists")

if __name__ == "__main__":
    print("Downloading and loading GPT-2 model...")
    download_and_load_gpt2("gpt2-medium (355M)")
    print("GPT-2 model downloaded and loaded successfully")
