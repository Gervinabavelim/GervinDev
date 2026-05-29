import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_datasets(token_ids, seq_len, train_split=0.9):
    split = int(len(token_ids) * train_split)
    train_data = TextDataset(token_ids[:split], seq_len)
    val_data = TextDataset(token_ids[split:], seq_len)
    return train_data, val_data
