# 00 — Roadmap: WSI → MIL pipeline

## Goal (one sentence)

Turn each **whole-slide image (WSI)** into a **bag of tile embeddings** with a **frozen pretrained encoder**, then train a small **multiple-instance learning (MIL)** head for **slide-level classification**.

You are **not** training a foundation model from scratch in this tutorial track.

## Typical pipeline

```text
Step 1  Labels + QC + fixed train/val/test splits (patient- or slide-level)
   │
Step 2  OpenSlide → tissue mask → tiles → tile encoder → feature bag (.pt per slide)
   │
Step 3  Train attention-MIL (or other pooling head) on bags; encoder frozen
   │
Step 4  Evaluate on held-out test; optional embedding visualization
```

## Mental model

| Object | Analogy |
|--------|---------|
| `.svs` WSI | A huge poster |
| Pyramid level | Zoom level (pick one level for tiling) |
| 256×256 tile | One postage stamp cut from the poster |
| Tissue filter | Throw away blank background |
| Tile embedding | Turn each stamp into a fixed-length vector (e.g. 768-D) |
| Feature bag | Envelope with all tile vectors for one slide |
| MIL / attention head | Learn which tiles matter, then predict slide label |

## Libraries you will touch

- **OpenSlide** — read WSI pyramids  
- **PyTorch** — tensors, `nn.Module`, training loop  
- **Optional:** `timm`, Hugging Face, or a pathology FM repo for the encoder  

## Environment sketch

```bash
conda create -n wsi-mil python=3.10
conda activate wsi-mil
pip install openslide-python pillow torch torchvision scikit-learn matplotlib
# OpenSlide system library: see OpenSlide install docs for your OS
```

On a remote GPU machine, run Jupyter **inside the same conda env** as OpenSlide and PyTorch: [10_jupyter_lab_remote_gpu.md](10_jupyter_lab_remote_gpu.md).

Next: [01_typical_folder_layout.md](01_typical_folder_layout.md)
