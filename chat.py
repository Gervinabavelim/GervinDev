import argparse
import readline

import torch
import torch.nn.functional as F

from tokenizer import GervinTokenizer
from model import GervinDev


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GervinDev(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


STOP_SEQUENCES = ["User:", "\nUser:", "\nYou:"]


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

            token_id = next_token.item()
            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            eos_id = tokenizer.char_to_id.get(tokenizer.eos_token)
            if eos_id is not None and token_id == eos_id:
                break

            decoded_so_far = tokenizer.decode(generated)
            tail = decoded_so_far[-20:]
            if any(stop in tail for stop in STOP_SEQUENCES):
                break

    full_text = tokenizer.decode(generated)
    return full_text


def build_prompt(history):
    parts = []
    for user_msg, assistant_msg in history:
        parts.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
    return "\n\n".join(parts)


def extract_response(full_output, prompt):
    response = full_output[len(prompt):]
    for stop in STOP_SEQUENCES:
        idx = response.find(stop)
        if idx != -1:
            response = response[:idx]
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="GervinDev Interactive Chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-history", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GervinDev] Device: {device}")

    print(f"[GervinDev] Loading model from {args.checkpoint}")
    model, _ = load_model(args.checkpoint, device)
    print(f"[GervinDev] Parameters: {model.num_parameters:,}")

    tokenizer = GervinTokenizer()
    tokenizer_dir = args.checkpoint.rsplit("/", 1)[0] if "/" in args.checkpoint else "."
    tokenizer.load(tokenizer_dir)

    print("\n" + "=" * 50)
    print("  GervinDev Interactive Chat")
    print("  Type your prompt and press Enter.")
    print("  Commands:")
    print("    /tokens N    — set max tokens (current: {})".format(args.max_tokens))
    print("    /temp N      — set temperature (current: {})".format(args.temperature))
    print("    /topk N      — set top-k (current: {})".format(args.top_k))
    print("    /reload      — reload the latest checkpoint")
    print("    /clear       — clear conversation history")
    print("    /quit        — exit")
    print("=" * 50 + "\n")

    max_tokens = args.max_tokens
    temperature = args.temperature
    top_k = args.top_k
    history = []

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt == "/quit":
            print("Bye!")
            break
        elif prompt.startswith("/tokens"):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  Max tokens set to {max_tokens}")
            except (IndexError, ValueError):
                print(f"  Current: {max_tokens}. Usage: /tokens 300")
            continue
        elif prompt.startswith("/temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print(f"  Current: {temperature}. Usage: /temp 0.5")
            continue
        elif prompt.startswith("/topk"):
            try:
                top_k = int(prompt.split()[1])
                print(f"  Top-k set to {top_k}")
            except (IndexError, ValueError):
                print(f"  Current: {top_k}. Usage: /topk 50")
            continue
        elif prompt == "/reload":
            print(f"  Reloading from {args.checkpoint}...")
            model, _ = load_model(args.checkpoint, device)
            print("  Model reloaded.")
            continue
        elif prompt == "/clear":
            history.clear()
            print("  Conversation history cleared.")
            continue

        recent_history = history[-args.max_history:]
        if recent_history:
            chat_prompt = build_prompt(recent_history) + f"\n\nUser: {prompt}\nAssistant:"
        else:
            chat_prompt = f"User: {prompt}\nAssistant:"

        output = generate(model, tokenizer, chat_prompt, max_tokens, temperature, top_k, device)
        response = extract_response(output, chat_prompt)

        if response:
            history.append((prompt, response))
            print(f"\nGervinDev> {response}\n")
        else:
            print("\nGervinDev> [no response generated]\n")


if __name__ == "__main__":
    main()
