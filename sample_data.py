"""Download a sample dataset to test GervinDev training."""

import os
import urllib.request


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_sample_data(output_path="data/tiny_shakespeare.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"[GervinDev] Data already exists at {output_path}")
        return output_path

    print(f"[GervinDev] Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, output_path)
    size = os.path.getsize(output_path)
    print(f"[GervinDev] Downloaded {size:,} bytes to {output_path}")
    return output_path


if __name__ == "__main__":
    download_sample_data()
