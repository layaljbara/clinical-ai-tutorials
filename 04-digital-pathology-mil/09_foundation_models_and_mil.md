# 09 — Pathology foundation models + MIL (concepts)

## Two-layer story

Most weakly supervised slide classifiers combine:

1. **Tile encoder (frozen)** — turns each histology patch into a vector  
2. **Slide head (trained)** — pools tiles (mean, max, attention) → slide label  

You usually **do not** retrain the foundation model on a small cohort unless you have a deliberate fine-tuning plan and enough data.

## Foundation model vs MIL head

| Piece | Trained on | Your project uses it for |
|-------|------------|--------------------------|
| Tile encoder | Large public / multi-site WSI pretraining | Feature extraction |
| MIL / attention head | **Your** labeled slides | Your classification task |

The encoder knows generic histology patterns; the head learns **your** labels (e.g. tumor grade, rejection, fibrosis stage).

## Attention MIL vs mean pooling

| Pooling | Idea | When it helps |
|---------|------|----------------|
| Mean pool | Average all tile vectors | Simple baseline |
| Max pool | Take strongest tile signal | Rare focal findings |
| Attention MIL | Learn tile importance weights | Heterogeneous slides |

## What to document in a paper / report

- Encoder name + version + input size + normalization  
- Tile size and pyramid level  
- Train/val/test split policy (patient- vs slide-level)  
- Whether encoder was frozen  
- Metrics: AUROC, AUPRC, per-class recall, confusion matrix  

## What this tutorial repo does **not** include

- Whole-slide images  
- Proprietary checkpoints (download from official releases yourself)  
- Project-specific labels or results  

## Further reading (external)

- OpenSlide documentation  
- Original papers for whichever encoder you choose (UNI, Virchow, CTransPath, etc.)  
- Classic MIL surveys (Ilse et al., attention-based MIL)

Next: [10_jupyter_lab_remote_gpu.md](10_jupyter_lab_remote_gpu.md)
