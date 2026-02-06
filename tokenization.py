import os
import re
import requests
import tiktoken
import importlib
import torch
from torch.utils.data import Dataset, DataLoader

SAMPLE_TEXT_FOLDER = "sample_texts"

if not os.path.exists(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt")):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    response = requests.get(url, timeout=30)
    with open(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt"), "wb") as f:
        f.write(response.content)
    print("Downloaded the-verdict.txt")

with open(os.path.join(SAMPLE_TEXT_FOLDER, "the-verdict.txt"), "r", encoding="utf-8") as f:
    raw_text = f.read()

# print(f"Loaded {len(raw_text)} characters")
# print(f"Preview: {raw_text[:100]}...")

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# vocab = {token:integer for integer, token in enumerate[str | Any](all_tokens)}
# print(f"Vocabulary size: {len(vocab)}")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) >= max_length, 'Number of tokenized inputs must at least be equal to max_length'

        # Use a sliding window to chunk the txt into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0, tokenizer = None):
    # Initialize the tokenizer
    tokenizer = tokenizer or tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

class WordTokenizer:
    def __init__(self):
        self.str_to_int = None
        self.int_to_str = None
        self.split_pattern = r'([,.:;?_!"()\']|--|\s)'
        self.sub_pattern = r'\s+([,.?!"()\'])'

    def encode(self, text: str) -> list[int]:
        if not self.str_to_int:
            self.build_vocab(text)
        preprocessed = re.split(self.split_pattern, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace unknown tokens with "<|UNK|>"
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(self.sub_pattern, r'\1', text)
        return text
    
    def build_vocab(self, text: str):
        preprocessed = re.split(self.split_pattern, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        all_tokens = sorted(list[str](set[str](preprocessed)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        vocab = {token:integer for integer, token in enumerate[str](all_tokens)}
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

class CharacterTokenizer:
    def __init__(self):
        self.str_to_int = None
        self.int_to_str = None
        self.unk_token = "<|unk|>"
        self.split_pattern = r'([,.:;?_!"()\']|--|\s)'
        self.sub_pattern = r'\s+([,.?!"()\'])'

    def encode(self, text: str) -> list[int]:
        if not self.str_to_int:
            self.build_vocab(text)
        unk_id = self.str_to_int[self.unk_token]
        ids = [self.str_to_int[c] if c in self.str_to_int else unk_id for c in text]
        return ids

    def decode(self, ids: list[int]) -> str:
        # If an unknown ID is encountered, use the unk_token string.
        return "".join([self.int_to_str[i] if i in self.int_to_str else self.unk_token for i in ids])
    
    def build_vocab(self, text: str):
        chars = sorted(list(set(text)))
        chars.append(self.unk_token)
        self.str_to_int = {c: i for i, c in enumerate(chars)}
        self.int_to_str = {i: c for c, i in self.str_to_int.items()}

# tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)

# enc_sample = enc_text[50:]
# context_size = 4

# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y:      {y}")

# wordTokenizer = WordTokenizer()
# wordTokenizer.build_vocab(raw_text)

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."

# text = " <|endoftext|> ".join((text1, text2))

# enc_text = wordTokenizer.encode(raw_text)
# print(len(enc_text))
# print(f"Encoded: {enc_text}")
# print(f"Decoded: {wordTokenizer.decode(enc_text)}")
# print(wordTokenizer.decode(wordTokenizer.encode(text)))