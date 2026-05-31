import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from tokenizer import GervinTokenizer
from model import GervinDev
from dataset import TextDataset
from chat_data import get_chat_text


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def finetune(args):
    device = get_device()
    print(f"[GervinDev] Device: {device}")

    print(f"[GervinDev] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GervinDev(config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"[GervinDev] Parameters: {model.num_parameters:,}")

    tokenizer = GervinTokenizer()
    tokenizer_dir = args.checkpoint.rsplit("/", 1)[0] if "/" in args.checkpoint else "."
    tokenizer.load(tokenizer_dir)

    chat_text = get_chat_text()
    print(f"[GervinDev] Chat data: {len(chat_text):,} characters")

    token_ids = tokenizer.encode(chat_text)
    print(f"[GervinDev] Chat tokens: {len(token_ids):,}")

    # Repeat the data to give the model more training steps
    repeated_ids = token_ids * args.repeat
    print(f"[GervinDev] Training tokens (repeated {args.repeat}x): {len(repeated_ids):,}")

    dataset = TextDataset(repeated_ids, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[GervinDev] Batches per epoch: {len(loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        count = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
            global_step += 1

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  step {global_step:>5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        avg_loss = epoch_loss / max(count, 1)
        print(f"[GervinDev] Epoch {epoch + 1}/{args.epochs} | avg_loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, config, global_step, avg_loss, tokenizer, args.out_dir)
            print(f"  [save] best model saved (loss={avg_loss:.4f})")

    save_checkpoint(model, optimizer, config, global_step, best_loss, tokenizer, args.out_dir)

    total_time = time.time() - start_time
    print(f"\n[GervinDev] Fine-tuning complete in {total_time:.1f}s")
    print(f"[GervinDev] Best loss: {best_loss:.4f}")
    print(f"[GervinDev] Model saved to {args.out_dir}/")


def save_checkpoint(model, optimizer, config, step, loss, tokenizer, out_dir):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "val_loss": loss,
    }
    torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))
    tokenizer.save(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GervinDev on chat data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Pre-trained checkpoint to start from")
    parser.add_argument("--out-dir", type=str, default="checkpoints",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than pre-training)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--repeat", type=int, default=10,
                        help="Times to repeat the chat data for more training steps")
    args = parser.parse_args()
    finetune(args)
