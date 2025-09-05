# LiteGPT

LiteGPT is a lightweight conversational AI model implemented in PyTorch.

## Features
- Custom GPT-like architecture
- Synthetic conversational corpus generation
- Tokenization with GPT-2 tokenizer
- Dialogue manager for interactive chat
- GPU/CPU support
- Model checkpoint saving/loading

## Requirements
- Python 3.8+
- PyTorch
- Transformers

## Installation
```bash
git clone <repo_url>
cd <repo_dir>
pip install torch transformers
```
Usage
bash```
Copy code
python lite_gpt.py```
Type messages interactively. Use exit or quit to leave.

Configuration
MAX_LENGTH: Maximum token sequence length

BATCH_SIZE: Training batch size

EPOCHS: Number of training epochs

LR: Learning rate

MAX_GENERATE_LEN: Maximum tokens to generate per response

DEFAULT_TEMPERATURE: Sampling temperature

MIN_CORPUS_LINES: Minimum lines for synthetic corpus

Directories
MODEL_DIR: Where the model checkpoint is saved

DATA_DIR: Where corpus data is stored

Synthetic Corpus
Generated if no existing corpus is found. Includes user and assistant conversational samples.

Model
LiteGPTBlock: Transformer block

LiteGPT: Full GPT-like model

DialogueManager: Handles generating responses from user input

License
MIT License
