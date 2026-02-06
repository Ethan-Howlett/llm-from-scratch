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
    pass

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
        self.split_pattern = r'([,.:;?_!"()\']|--|\s)'
        self.sub_pattern = r'\s+([,.?!"()\'])'

    def encode(self, text: str) -> list[int]:
        if not self.str_to_int:
            self.build_vocab(text)
        ids = [self.str_to_int[c] for c in text]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join([self.int_to_str[i] for i in ids])
    
    def build_vocab(self, text: str):
        self.str_to_int = {c:i for i, c in enumerate[str](sorted(list[str](set[str](text))))}
        self.int_to_str = {i:c for c, i in self.str_to_int.items()}

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

enc_sample = enc_text[50:]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

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