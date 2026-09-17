"""Full nanoGPT training loop (Karpathy style) for X-GPT.

Usage:
  python data.py
  python train.py                  # tiny CPU run
  python train.py --preset colab   # bigger GPU run on Colab T4
  python train.py --resume --max_iters 500
  python sample.py --prompt "Kural 1"
"""
import argparse
import json
import math
from pathlib import Path

import torch

from data import CharTokenizer, OUT_DIR
from gpt import GPT, GPTConfig

ROOT = Path(__file__).parent
CKPT = ROOT / "checkpoint.pt"
BEST = ROOT / "best.pt"

PRESETS = {
    "cpu": dict(block_size=64, batch_size=32, n_layer=4, n_head=4, n_embd=128),
    "colab": dict(block_size=128, batch_size=64, n_layer=6, n_head=6, n_embd=192),
}


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train, val, block_size, batch_size, device, iters=100):
    model.eval()
    out = {}
    for split, data in (("train", train), ("val", val)):
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def lr_schedule(step, warmup, max_iters, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_iters - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    meta = json.load(open(OUT_DIR / "meta.json", encoding="utf-8"))
    tok = CharTokenizer(meta["chars"])
    train_data = torch.load(OUT_DIR / "train.bin")
    val_data = torch.load(OUT_DIR / "val.bin")

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=meta["vocab_size"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"params={model.param_count() / 1e6:.2f}M device={device}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    start_iter = 0
    best_val = float("inf")
    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_iter = ck["iter"] + 1
        best_val = ck.get("val", best_val)
        print(f"resumed from iter {start_iter}")

    for it in range(start_iter, args.max_iters):
        lr = lr_schedule(it, args.warmup, args.max_iters, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                args.block_size, args.batch_size, device, args.eval_iters,
            )
            print(
                f"iter {it}: train {losses['train']:.4f} "
                f"val {losses['val']:.4f} lr {lr:.2e}"
            )
            ck = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": cfg.__dict__,
                "iter": it,
                "val": losses["val"],
            }
            torch.save(ck, CKPT)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(ck, BEST)
            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
            preview = model.generate(ctx, 200, temperature=0.8, top_k=50)
            print("--- sample ---")
            print(tok.decode(preview[0].tolist())[:500])
    print(f"done. best val {best_val:.4f}. ckpt -> {CKPT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="cpu", choices=list(PRESETS))
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=6646)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    preset = PRESETS[args.preset]
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    train(args)
