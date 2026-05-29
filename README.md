# GervinDev

A tiny transformer LLM built from scratch with Python + PyTorch.

## Architecture

- **Model**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~2M (configurable)
- **Tokenizer**: Custom BPE (Byte-Pair Encoding)
- **Attention**: Multi-head causal self-attention with 4 heads
- **Layers**: 4 transformer blocks
- **Embedding**: 128-dimensional with learned positional encoding

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample training data (Tiny Shakespeare)
python sample_data.py

# Train the model
python train.py --data data/tiny_shakespeare.txt --epochs 10

# Generate text
python generate.py --prompt "To be or not to be" --max-tokens 200
```

## Project Structure

```
GervinDev/
├── config.py        # Hyperparameters
├── tokenizer.py     # BPE tokenizer
├── model.py         # Transformer architecture
├── dataset.py       # Data loading & batching
├── train.py         # Training loop
├── generate.py      # Text generation
├── sample_data.py   # Download sample dataset
└── requirements.txt # Dependencies
```

## Configuration

Edit `config.py` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 4 | Transformer blocks |
| `d_ff` | 512 | Feed-forward hidden size |
| `vocab_size` | 10,000 | Tokenizer vocabulary |
| `max_seq_len` | 256 | Context window |

## Training on Custom Data

Provide any `.txt` file:

```bash
python train.py --data your_text.txt --epochs 20 --lr 3e-4
```

## License

MIT
