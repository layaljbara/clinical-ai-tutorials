# 05 — Loading feature bags & splits

## What a bag looks like

Each slide → one file, e.g. `features/{slide_id}.pt`:

| Key | Meaning |
|-----|---------|
| `slide_id` | Unique slide id |
| `label` | Slide-level class string or int |
| `split` | `train` / `val` / `test` |
| `features` | `FloatTensor [N, D]` |
| `coords` | Optional tile coordinates |
| `n_tiles` | `N` (varies by slide) |

## Load one bag

```python
import torch

bag = torch.load("features/example_001.pt", map_location="cpu", weights_only=False)
print(bag["slide_id"], bag["label"], bag["features"].shape)
# e.g. example_001 class_a torch.Size([412, 768])
```

## Use frozen splits

```python
import csv
from pathlib import Path

def ids_for_split(name: str, splits_dir="data/splits"):
    path = Path(splits_dir) / f"{name}.csv"
    with path.open() as f:
        return [row["slide_id"] for row in csv.DictReader(f)]

train_ids = ids_for_split("train")
val_ids = ids_for_split("val")
test_ids = ids_for_split("test")
```

Do not reshuffle test slides while tuning hyperparameters.

## Label encoding

```python
LABEL2ID = {"class_a": 0, "class_b": 1}
y = LABEL2ID[bag["label"]]
```

## Class imbalance

If one class is rarer, use **class weights**, **balanced sampling**, or report **per-class recall** and **AUPRC**, not accuracy alone.

## Try it

```bash
python snippets/t05_inspect_bag.py
```

Next: [06_training_mil_head.md](06_training_mil_head.md)
