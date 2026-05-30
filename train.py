import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from config import GervinConfig
from tokenizer import GervinTokenizer
from model import GervinDev
from dataset import load_text_file, create_datasets


def get_device(preference="auto"):
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(args):
    config = GervinConfig()
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr

    device = get_device(config.device)
    print(f"[GervinDev] Device: {device}")

    print(f"[GervinDev] Loading data from {args.data}")
    text = load_text_file(args.data)
    print(f"[GervinDev] Text length: {len(text):,} characters")

    tokenizer = GervinTokenizer(vocab_size=config.vocab_size)
    print("[GervinDev] Training tokenizer...")
    tokenizer.train(text)
    config.vocab_size = tokenizer.vocab_size
    print(f"[GervinDev] Vocab size: {config.vocab_size}")

    token_ids = tokenizer.encode(text)
    print(f"[GervinDev] Token count: {len(token_ids):,}")

    train_data, val_data = create_datasets(token_ids, config.max_seq_len)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=True)

    model = GervinDev(config).to(device)
    print(f"[GervinDev] Parameters: {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs(args.out_dir, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    resume_path = os.path.join(args.out_dir, "best_model.pt")
    if args.resume and os.path.exists(resume_path):
        print(f"[GervinDev] Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        steps_per_epoch = len(train_loader)
        start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        print(f"[GervinDev] Resumed at step {global_step} (epoch ~{start_epoch}), best_val_loss={best_val_loss:.4f}")

    start_time = time.time()

    for epoch in range(start_epoch, config.max_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  step {global_step:>6d} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | {elapsed:.1f}s"
                )

            if global_step > 0 and global_step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  [eval] step {global_step} | val_loss {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "step": global_step,
                        "val_loss": val_loss,
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pt"))
                    tokenizer.save(args.out_dir)
                    print(f"  [save] best model saved (val_loss={val_loss:.4f})")

            global_step += 1

        print(f"[GervinDev] Epoch {epoch + 1}/{config.max_epochs} complete")

    final_checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": global_step,
        "val_loss": best_val_loss,
    }
    torch.save(final_checkpoint, os.path.join(args.out_dir, "final_model.pt"))
    tokenizer.save(args.out_dir)

    total_time = time.time() - start_time
    print(f"\n[GervinDev] Training complete in {total_time:.1f}s")
    print(f"[GervinDev] Best val loss: {best_val_loss:.4f}")
    print(f"[GervinDev] Model saved to {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GervinDev LLM")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(args)
