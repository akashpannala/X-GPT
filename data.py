"""Char-level data pipeline, extracted from BigramModel.ipynb cells 3-10.

Format per Kural is kept verbatim from the notebook so the bigram
experiment and nanoGPT train on the same text.
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DATA_JSON = ROOT / "Data" / "thirukkural.json"
OUT_DIR = ROOT / "data_prepared"


def build_corpus(json_path=DATA_JSON):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    parts = []
    for item in data["kural"]:
        parts.append(f"\n<|start|>\nKural {item['Number']}\n")
        parts.append(f"Tamil: {item['Line1']} {item['Line2']}\n")
        parts.append(
            f"Transliteration: {item['transliteration1']} {item['transliteration2']}\n"
        )
        parts.append(f"English: {item['explanation']}\n")
        parts.append(f"Tamil Meaning: {item['sp']}\n<|end|>\n")
    return "".join(parts)


class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def prepare(out_dir=OUT_DIR, split=0.9):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = build_corpus()
    chars = sorted(set(text))
    tok = CharTokenizer(chars)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * split)
    train, val = data[:n], data[n:]

    (out_dir / "corpus.txt").write_text(text, encoding="utf-8")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(chars), "chars": chars}, f, ensure_ascii=False)
    torch.save(train, out_dir / "train.bin")
    torch.save(val, out_dir / "val.bin")
    print(f"chars={len(text)} vocab={len(chars)} train={len(train)} val={len(val)}")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--split", type=float, default=0.9)
    args = p.parse_args()
    prepare(args.out_dir, args.split)
