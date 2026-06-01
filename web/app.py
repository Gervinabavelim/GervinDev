import os
import json
import argparse

from dotenv import load_dotenv
from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import anthropic

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = Flask(__name__)
client = None

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
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'token': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except anthropic.APIError as e:
            yield f"data: {json.dumps({'token': f'[Error: {e.message}]'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/info")
def info_endpoint():
    return jsonify({
        "engine": "Claude API",
        "model": "claude-sonnet-4-20250514",
        "name": "GervinDev",
    })


def main():
    global client

    parser = argparse.ArgumentParser(description="GervinDev Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[GervinDev] Error: ANTHROPIC_API_KEY not set. Add it to .env file.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    print(f"[GervinDev] Claude API connected")
    print(f"[GervinDev] Starting web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
