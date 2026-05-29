from dataclasses import dataclass


@dataclass
class GervinConfig:
    vocab_size: int = 10000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 50
    warmup_steps: int = 100
    eval_interval: int = 500
    device: str = "auto"
