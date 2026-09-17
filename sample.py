"""Query a trained nanoGPT checkpoint.

  python sample.py --prompt "Kural 1" --max_tokens 300
  python sample.py --prompt "Tamil:" --temperature 0.8 --top_k 50
"""
import argparse
import json
from pathlib import Path

import torch

from data import CharTokenizer, OUT_DIR
from gpt import GPT, GPTConfig

ROOT = Path(__file__).parent


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoint.pt"))
    p.add_argument("--prompt", default="\n<|start|>\nKural 1\n")
    p.add_argument("--max_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=6646)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    meta = json.load(open(OUT_DIR / "meta.json", encoding="utf-8"))
    tok = CharTokenizer(meta["chars"])
    model, ck = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (iter {ck.get('iter')}, val {ck.get('val'):.4f})")

    ids = [tok.stoi[c] for c in args.prompt if c in tok.stoi]
    skipped = len(args.prompt) - len(ids)
    if skipped:
        print(f"(skipped {skipped} unknown chars in prompt)")
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, args.max_tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))
