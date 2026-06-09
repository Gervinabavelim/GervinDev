import os
import json
import argparse

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import urllib.request

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434"
MODEL = "gervindev"

SYSTEM_PROMPT = """You are GervinDev, a personal AI assistant created by Gervin. You are helpful, friendly, and concise.
Keep responses conversational and to the point. Use plain language.
If asked who you are, say you are GervinDev, Gervin's personal AI assistant.
Do not use markdown formatting like ** or ## in your responses — just plain text."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 1024)
    temperature = data.get("temperature", 0.7)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    def event_stream():
        try:
            payload = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }).encode()

            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("message", {}).get("content"):
                        token = chunk["message"]["content"]
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if chunk.get("done"):
                        break

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'[Error: {str(e)}]'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/info")
def info_endpoint():
    return jsonify({
        "engine": "Ollama (Local)",
        "model": MODEL,
        "name": "GervinDev",
    })


def main():
    parser = argparse.ArgumentParser(description="GervinDev Web UI")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model", type=str, default="gervindev")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags")
        print(f"[GervinDev] Ollama connected")
    except Exception:
        print(f"[GervinDev] Warning: Ollama not running at {OLLAMA_URL}")
        print(f"[GervinDev] Start it with: ollama serve")

    print(f"[GervinDev] Model: {MODEL}")
    print(f"[GervinDev] Starting web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
