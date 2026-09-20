# GPT_From_Scratch
Building An Transformer From Scratch Learning From Andrej Karpathy

## nanoGPT path (Thirukkural)

Decoder-only char GPT in `gpt.py`, data pipeline in `data.py`
(verbatim `BigramModel.ipynb` format), full loop in `train.py`,
query in `sample.py`. Old encoder-decoder NMT stays in `model.py`.

```bash
python data.py
python train.py                  # tiny CPU preset
python train.py --preset colab   # bigger GPU run (Colab T4)
python sample.py --prompt "Kural 1" --max_tokens 300
```

Laptop here has no NVIDIA GPU (Intel iGPU only, torch CPU),
so quick CPU smoke tests run local, real training runs on Colab GPU.

## Latest run (Colab T4, 2026-09-18)

`--preset colab` (6 layers, 192 embd, block 128), 5000 iters → val 1.38.
Weights committed as `checkpoint.pt` / `best.pt`. Details + verbatim
samples in `results/`.
