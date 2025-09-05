# LiteGPT

LiteGPT is a small generative transformer-based language model implemented in PyTorch. It is designed for conversational AI and text generation. The model uses a lightweight GPT architecture with multiple transformer blocks, token and positional embeddings, and a causal self-attention mechanism to generate coherent responses.

---

## Features

- Generative conversational AI
- Lightweight transformer architecture suitable for training on small datasets
- Custom GPT-2 tokenizer with special tokens for conversation (`<BOS>`, `<EOS>`, `<user>:`, `<assistant>:`)
- Token-level language modeling with cross-entropy loss
- Temperature-controlled sampling for diverse outputs

---

## Requirements

- Python 3.8+
- PyTorch 2.x
- Transformers library

Install dependencies:

```bash
pip install torch transformers
```
Usage
Run the script to train and interact with LiteGPT:

bash
Copy code
python model.py
Training:

If no corpus exists, a synthetic conversational dataset is generated.

The model trains on the tokenized corpus using a causal language modeling objective.

Training parameters such as MAX_LENGTH, BATCH_SIZE, EPOCHS, and LR are configurable in the script.

Interactive Mode:

After training, the script launches a console-based chat.

Type messages after the You: prompt and LiteGPT will generate responses.

Type exit or quit to end the session.

Model Architecture
Embedding layers: Token embeddings + positional embeddings

Transformer blocks: Multi-head causal self-attention + feed-forward layers with GELU activation, residual connections, and layer normalization

Output layer: Linear projection to vocabulary logits

Generation: Token-by-token sampling using softmax with temperature control

Loss: Cross-entropy for predicting the next token

Generative Behavior
LiteGPT generates text in a left-to-right autoregressive manner. It can:

Respond to user prompts

Complete sentences or paragraphs

Generate diverse outputs depending on the sampling temperature

Configuration Highlights
MAX_LENGTH: Maximum sequence length for inputs

BATCH_SIZE: Number of sequences per training batch

EPOCHS: Number of training epochs

LR: Learning rate for optimizer

MAX_GENERATE_LEN: Maximum tokens generated per response

DEFAULT_TEMPERATURE: Sampling temperature for generation

Author
Created by raziel-star.
