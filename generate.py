import argparse

import torch
import torch.nn.functional as F

from config import GervinConfig
from tokenizer import GervinTokenizer
from model import GervinDev


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GervinDev(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated = list(token_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]

            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_values, _ = torch.topk(logits, top_k)
                logits[logits < top_values[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text with GervinDev")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GervinDev] Device: {device}")

    print(f"[GervinDev] Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"[GervinDev] Parameters: {model.num_parameters:,}")

    tokenizer = GervinTokenizer()
    tokenizer_dir = args.checkpoint.rsplit("/", 1)[0] if "/" in args.checkpoint else "."
    tokenizer.load(tokenizer_dir)

    print(f"\n--- Prompt ---\n{args.prompt}")
    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
    print(f"\n--- Generated ---\n{output}")


if __name__ == "__main__":
    main()
