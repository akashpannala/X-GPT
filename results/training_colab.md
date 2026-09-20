# Colab training run

- Date: 2026-09-18 (Colab T4 GPU)
- Command: `python data.py` then `python train.py --preset colab --max_iters 5000 --eval_interval 500`
- Preset colab: block_size=128, batch=64, n_layer=6, n_head=6, n_embd=192, dropout=0.1, lr=3e-4 cosine + warmup 100, AdamW, grad clip 1.0
- Data: 590808 chars, vocab 124, train 531727 / val 59081 (char-level, notebook format)
- Result: iter 4999, val loss 1.3799
- Weights: `checkpoint.pt` + `best.pt` (32MB each, committed), config embedded in checkpoint
- Laptop (Intel iGPU only, torch CPU) is used for sampling only
