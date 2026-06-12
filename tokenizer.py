import json
import os
from collections import Counter


class GervinTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.merges = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def train(self, text, max_chars=500000):
        if len(text) > max_chars:
            text = text[:max_chars]

        chars = sorted(set(text))
        self.char_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        offset = len(self.special_tokens)
        for i, ch in enumerate(chars):
            self.char_to_id[ch] = i + offset

        tokens = [self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text]
        next_id = len(self.char_to_id)

        num_merges = self.vocab_size - next_id
        print(f"  [tokenizer] {len(tokens):,} tokens, planning {num_merges} merges...")

        for step in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            self.merges[best_pair] = next_id
            self.char_to_id[f"<merge_{next_id}>"] = next_id

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(next_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            next_id += 1

            if (step + 1) % 1000 == 0:
                print(f"  [tokenizer] {step + 1}/{num_merges} merges, {len(tokens):,} tokens remaining")

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        print(f"  [tokenizer] done — vocab size: {self.vocab_size}")

    def encode(self, text):
        tokens = [self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text]

        while True:
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair not in pairs:
                    pairs[pair] = i

            mergeable = {p: self.merges[p] for p in pairs if p in self.merges}
            if not mergeable:
                break

            best_pair = min(mergeable, key=mergeable.get)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, token_ids):
        result = []
        for tid in token_ids:
            tok = self.id_to_char.get(tid, self.unk_token)
            if tok.startswith("<merge_"):
                result.append(self._expand_merge(tid))
            elif tok not in self.special_tokens:
                result.append(tok)
        return "".join(result)

    def _expand_merge(self, merge_id):
        for (a, b), mid in self.merges.items():
            if mid == merge_id:
                left = self._expand_merge(a) if a in [v for v in self.merges.values()] else self.id_to_char.get(a, "")
                right = self._expand_merge(b) if b in [v for v in self.merges.values()] else self.id_to_char.get(b, "")
                if left.startswith("<") and left.endswith(">") and left in self.special_tokens:
                    left = ""
                if right.startswith("<") and right.endswith(">") and right in self.special_tokens:
                    right = ""
                return left + right
        return self.id_to_char.get(merge_id, "")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(os.path.join(path, "tokenizer.json"), "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.merges = {}
        for key, val in data["merges"].items():
            a, b = key.split(",")
            self.merges[(int(a), int(b))] = val
