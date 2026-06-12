from dataclasses import dataclass


@dataclass
class GervinConfig:
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 50
    warmup_steps: int = 100
    eval_interval: int = 500
    device: str = "auto"
