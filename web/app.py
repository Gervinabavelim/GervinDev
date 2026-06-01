import sys
import os
import json
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, Response, jsonify, stream_with_context

from tokenizer import GervinTokenizer
from model import GervinDev

app = Flask(__name__)

model = None
tokenizer = None
config = None
device = None
checkpoint_path = None
model_lock = threading.Lock()


def load_model_from_checkpoint(path, dev):
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    cfg = checkpoint["config"]
    m = GervinDev(cfg).to(dev)
    m.load_state_dict(checkpoint["model"])
    m.eval()
    return m, cfg


def generate_stream(prompt, max_tokens=200, temperature=0.8, top_k=40):
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = list(token_ids)
    prev_text = tokenizer.decode(generated)

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

            full_text = tokenizer.decode(generated)
            new_text = full_text[len(prev_text):]
            prev_text = full_text

            if "User:" in new_text:
                clean = new_text.split("User:")[0].rstrip()
                if clean:
                    yield clean
                return

            if new_text:
                yield new_text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 40)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    chat_prompt = f"User: {prompt}\nAssistant:"

    def event_stream():
        with model_lock:
            for chunk in generate_stream(chat_prompt, max_tokens, temperature, top_k):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reload", methods=["POST"])
def reload_endpoint():
    global model, config
    with model_lock:
        model, config = load_model_from_checkpoint(checkpoint_path, device)
    return jsonify({"status": "reloaded", "params": model.num_parameters})


@app.route("/info")
def info_endpoint():
    return jsonify({
        "params": model.num_parameters,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
    })


def main():
    global model, tokenizer, config, device, checkpoint_path

    parser = argparse.ArgumentParser(description="GervinDev Web UI")
    default_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "best_model.pt")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[GervinDev] Device: {device}")
    print(f"[GervinDev] Loading model from {checkpoint_path}")

    model, config = load_model_from_checkpoint(checkpoint_path, device)
    print(f"[GervinDev] Parameters: {model.num_parameters:,}")

    tokenizer = GervinTokenizer()
    tokenizer_dir = checkpoint_path.rsplit("/", 1)[0] if "/" in checkpoint_path else "."
    tokenizer.load(tokenizer_dir)

    print(f"[GervinDev] Starting web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
