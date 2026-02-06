# Tokenization Analysis Project

A comparative analysis of different tokenization methods for Natural Language Processing (NLP), including character-level, word-level, and Byte Pair Encoding (BPE) tokenization.

## Table of Contents

- [Overview](#overview)
- [Tokenization Methods](#tokenization-methods)
- [Key Findings](#key-findings)
- [Comparison Metrics](#comparison-metrics)
- [Advantages and Disadvantages](#advantages-and-disadvantages)
- [Challenges and Solutions](#challenges-and-solutions)
- [Code Examples](#code-examples)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project implements and compares three tokenization approaches commonly used in NLP:

1. **Character Tokenizer** - Splits text into individual characters
2. **Word Tokenizer** - Splits text by words using regex patterns
3. **BPE Tokenizer (tiktoken)** - OpenAI's Byte Pair Encoding tokenizer

The analysis includes performance benchmarks, multilingual support, error analysis, and recommendations for different use cases.

## Tokenization Methods

### Character Tokenizer

The simplest tokenization approach that splits text into individual characters.

**How it works:**
- Each unique character in the training corpus becomes a token
- Text is encoded by mapping each character to its integer ID
- Decoding simply joins characters back together

**Characteristics:**
- Vocabulary size: Number of unique characters (~100-200 for English)
- Sequence length: Equal to text length (very long)
- OOV handling: Excellent - any character can be encoded if in vocabulary

### Word Tokenizer

Splits text into words and punctuation using regular expression patterns.

**How it works:**
- Uses regex pattern `([,.:;?_!"()\']|--|\s)` to split text
- Builds vocabulary from unique tokens in training corpus
- Unknown words are mapped to `<|unk|>` special token
- Includes `<|endoftext|>` for document boundaries

**Characteristics:**
- Vocabulary size: Number of unique words (thousands to millions)
- Sequence length: Number of words (much shorter than character-level)
- OOV handling: Poor - unknown words become `<|unk|>`

### BPE Tokenizer (tiktoken/GPT-2)

Byte Pair Encoding learns subword units by iteratively merging frequent character pairs.

**How it works:**
- Starts with byte-level vocabulary
- Iteratively merges most frequent adjacent pairs
- Creates subword vocabulary balancing frequency and coverage
- Can encode any byte sequence (no OOV issues)

**Characteristics:**
- Vocabulary size: Fixed at 50,257 tokens (GPT-2)
- Sequence length: Shortest among all methods
- OOV handling: Perfect - uses byte fallback for unknown sequences

## Key Findings

### 1. Compression Efficiency

| Tokenizer | Avg Compression Ratio | Sequence Length |
|-----------|----------------------|-----------------|
| Character | 1.00 chars/token | Longest |
| Word | ~5.5 chars/token | Medium |
| BPE (GPT-2) | ~4.0 chars/token | Shortest |

### 2. Vocabulary Coverage

- **Character**: Uses 100% of vocabulary (small vocab, fully utilized)
- **Word**: Uses ~50-70% of vocabulary (many rare words)
- **BPE**: Uses <5% of vocabulary (large pre-trained vocab, sparse usage)

### 3. Multilingual Performance

BPE tokenizers trained on English show reduced efficiency for non-Latin scripts:

| Language | BPE Compression |
|----------|-----------------|
| English | ~4.0 chars/token |
| Spanish | ~3.8 chars/token |
| Chinese | ~1.2 chars/token |
| Japanese | ~1.0 chars/token |

### 4. Speed Performance

- **tiktoken (BPE)**: Fastest due to optimized Rust implementation
- **Character**: Fast due to simple 1:1 mapping
- **Word**: Moderate speed with regex overhead

## Comparison Metrics

| Metric | Character | Word | BPE |
|--------|-----------|------|-----|
| Vocabulary Size | ★★★★★ (smallest) | ★★☆☆☆ | ★★★☆☆ |
| Sequence Length | ★☆☆☆☆ (longest) | ★★★★☆ | ★★★★★ |
| OOV Handling | ★★★★★ | ★☆☆☆☆ | ★★★★★ |
| Training Speed | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| Inference Speed | ★★☆☆☆ | ★★★★☆ | ★★★★★ |
| Memory Usage | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| Interpretability | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| Morphology | ★☆☆☆☆ | ★★☆☆☆ | ★★★★★ |

## Advantages and Disadvantages

### Character Tokenizer

**Advantages:**
- Simple implementation
- Small, fixed vocabulary
- Handles any text (no OOV)
- Good for noisy text (typos, informal language)
- Language-agnostic

**Disadvantages:**
- Very long sequences increase compute cost
- Difficult for models to learn word-level semantics
- Poor compression ratio
- Struggles with long-range dependencies

### Word Tokenizer

**Advantages:**
- Interpretable tokens (actual words)
- Short sequences
- Good for well-defined vocabularies
- Preserves word boundaries
- Fast tokenization

**Disadvantages:**
- Large vocabulary for diverse text
- Cannot handle unknown words (OOV problem)
- Struggles with morphological variations
- Language/domain specific

### BPE Tokenizer

**Advantages:**
- Best balance of vocab size and sequence length
- No OOV issues (byte-level fallback)
- Captures morphological patterns
- Pre-trained models available
- Efficient inference with optimized implementations

**Disadvantages:**
- Complex training process
- May split words unexpectedly
- Less interpretable than word tokens
- Biased toward training data distribution
- Reduced efficiency for non-English text

## Challenges and Solutions

### Challenge 1: Out-of-Vocabulary Words (Word Tokenizer)

**Problem:** Unknown words become meaningless `<|unk|>` tokens.

**Solution:** 
- Implemented `<|unk|>` token to handle gracefully
- Consider hybrid approach with subword fallback
- Use domain-specific vocabulary expansion

### Challenge 2: Long Sequences (Character Tokenizer)

**Problem:** Character sequences exceed model context limits.

**Solution:**
- Use sliding window approach
- Consider character n-grams
- Apply character-level CNNs for compression

### Challenge 3: Multilingual Support (BPE)

**Problem:** BPE trained on English has poor compression for other languages.

**Solution:**
- Use multilingual tokenizers (mBERT, XLM-R)
- Train custom BPE on target language corpus
- Consider SentencePiece for language-agnostic tokenization

### Challenge 4: Special Characters and Formatting

**Problem:** URLs, emails, code snippets tokenize poorly.

**Solution:**
- Added regex patterns for special formats
- Preserve whitespace in character tokenizer
- Use tiktoken's byte-level encoding for full coverage

## Code Examples

### Basic Usage

```python
import tokenization
import tiktoken

# Initialize tokenizers
char_tokenizer = tokenization.CharacterTokenizer()
word_tokenizer = tokenization.WordTokenizer()
bpe_tokenizer = tiktoken.get_encoding("gpt2")

# Build vocabulary (character and word tokenizers)
with open("sample_texts/the-verdict.txt", "r") as f:
    corpus = f.read()

char_tokenizer.build_vocab(corpus)
word_tokenizer.build_vocab(corpus)

# Tokenize text
text = "Hello, how are you today?"

char_tokens = char_tokenizer.encode(text)
word_tokens = word_tokenizer.encode(text)
bpe_tokens = bpe_tokenizer.encode(text)

print(f"Character: {char_tokens}")
print(f"Word: {word_tokens}")
print(f"BPE: {bpe_tokens}")
```

### Decoding Tokens

```python
# Decode back to text
char_text = char_tokenizer.decode(char_tokens)
word_text = word_tokenizer.decode(word_tokens)
bpe_text = bpe_tokenizer.decode(bpe_tokens)

print(f"Character decoded: {char_text}")
print(f"Word decoded: {word_text}")
print(f"BPE decoded: {bpe_text}")
```

### Creating DataLoaders for Training

```python
# Create PyTorch DataLoader with BPE tokenization
dataloader = tokenization.create_dataloader(
    txt=corpus,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True
)

for batch in dataloader:
    inputs, targets = batch
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    break
```

### Handling Special Tokens (tiktoken)

```python
# Encode with special tokens
text_with_special = "Hello<|endoftext|>World"
tokens = bpe_tokenizer.encode(
    text_with_special, 
    allowed_special={"<|endoftext|>"}
)
print(f"Tokens: {tokens}")
print(f"Decoded: {bpe_tokenizer.decode(tokens)}")
```

### Analyzing Tokenization

```python
def analyze_text(text, tokenizer, name):
    """Analyze tokenization metrics for a text."""
    if name == "BPE":
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer.encode(text)
    
    return {
        "token_count": len(tokens),
        "compression_ratio": len(text) / len(tokens),
        "unique_tokens": len(set(tokens))
    }

# Compare methods
for name, tok in [("Character", char_tokenizer), 
                   ("Word", word_tokenizer), 
                   ("BPE", bpe_tokenizer)]:
    metrics = analyze_text(text, tok, name)
    print(f"{name}: {metrics}")
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- tiktoken
- torch
- matplotlib
- pandas
- numpy

## Usage

1. **Run the analysis notebook:**
   ```bash
   jupyter notebook analysis.ipynb
   ```

2. **Use tokenizers in your code:**
   ```python
   import tokenization
   
   tokenizer = tokenization.WordTokenizer()
   tokenizer.build_vocab(your_text)
   tokens = tokenizer.encode("Your text here")
   ```

3. **Create training data:**
   ```python
   dataloader = tokenization.create_dataloader(
       txt=your_corpus,
       batch_size=8,
       max_length=512
   )
   ```

## Project Structure

```
llm/
├── analysis.ipynb      # Main analysis notebook
├── tokenization.py     # Tokenizer implementations
├── sample_texts/       # Sample text files for analysis
│   ├── the-verdict.txt
│   ├── ai-news.txt
│   ├── family.txt
│   └── pytorch-doc.txt
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## References

- [tiktoken - OpenAI's BPE tokenizer](https://github.com/openai/tiktoken)
- [BPE Paper - Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://github.com/google/sentencepiece)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)

## License

MIT License
