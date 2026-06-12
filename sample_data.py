"""Download sample datasets for GervinDev training."""

import os
import urllib.request


DATASETS = {
    "tiny_shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tiny_shakespeare.txt",
        "description": "Tiny Shakespeare (~1.1MB)",
    },
    "alice_in_wonderland": {
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "filename": "alice_in_wonderland.txt",
        "description": "Alice's Adventures in Wonderland by Lewis Carroll (~170KB)",
    },
    "sherlock_holmes": {
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "filename": "sherlock_holmes.txt",
        "description": "The Adventures of Sherlock Holmes by Arthur Conan Doyle (~580KB)",
    },
    "frankenstein": {
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "filename": "frankenstein.txt",
        "description": "Frankenstein by Mary Shelley (~440KB)",
    },
    "pride_and_prejudice": {
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "filename": "pride_and_prejudice.txt",
        "description": "Pride and Prejudice by Jane Austen (~710KB)",
    },
    "great_expectations": {
        "url": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "filename": "great_expectations.txt",
        "description": "Great Expectations by Charles Dickens (~1MB)",
    },
}

DATA_DIR = "data"


def download_dataset(name, force=False):
    if name not in DATASETS:
        print(f"[GervinDev] Unknown dataset: {name}")
        return None

    info = DATASETS[name]
    output_path = os.path.join(DATA_DIR, info["filename"])
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(output_path) and not force:
        print(f"[GervinDev] Already exists: {output_path}")
        return output_path

    print(f"[GervinDev] Downloading {info['description']}...")
    try:
        urllib.request.urlretrieve(info["url"], output_path)
        size = os.path.getsize(output_path)
        print(f"[GervinDev] Downloaded {size:,} bytes to {output_path}")
        return output_path
    except Exception as e:
        print(f"[GervinDev] Failed to download {name}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def strip_gutenberg_header(text):
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG",
                   "End of the Project Gutenberg", "End of Project Gutenberg"]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def download_all(force=False):
    paths = []
    for name in DATASETS:
        path = download_dataset(name, force=force)
        if path:
            paths.append(path)
    return paths


def combine_all(output_path=None):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "combined.txt")

    texts = []
    for name, info in DATASETS.items():
        filepath = os.path.join(DATA_DIR, info["filename"])
        if not os.path.exists(filepath):
            print(f"[GervinDev] Skipping {name} (not downloaded)")
            continue
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if name != "tiny_shakespeare":
            text = strip_gutenberg_header(text)
        texts.append(text)

    combined = "\n\n".join(texts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"[GervinDev] Combined {len(texts)} files -> {output_path} ({len(combined):,} chars)")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download training data for GervinDev")
    parser.add_argument("--only", type=str, default=None,
                        help=f"Download a single dataset: {', '.join(DATASETS.keys())}")
    parser.add_argument("--combine", action="store_true",
                        help="Combine all downloaded files into data/combined.txt")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args()

    if args.only:
        download_dataset(args.only, force=args.force)
    else:
        download_all(force=args.force)

    if args.combine or not args.only:
        combine_all()
