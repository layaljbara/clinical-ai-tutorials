# 08 — Reading MIL metrics

After training, you usually save a JSON or CSV summary.

## Example metrics file

```python
import json
print(json.load(open("models/mil/metrics.json"))["test"])
```

## Common fields

| Metric | Meaning |
|--------|---------|
| **AUROC** | Ranking quality (higher score = better class separation) |
| **AUPRC** | Useful when the positive class is rare |
| **Balanced accuracy** | Average of per-class recall |
| **Per-class recall** | Of true class A (or B), how many did we catch? |
| **Confusion matrix** | Counts of predicted vs true |

## Confusion matrix (binary)

```text
                Predicted
              class_a  class_b
True class_a     TP_a    FN_a
True class_b     FN_b    TP_b
```

Report **both** classes when data are imbalanced — accuracy alone can mislead.

## Plot training curves

```bash
python snippets/t08_plot_training_curves.py
```

Next: [09_foundation_models_and_mil.md](09_foundation_models_and_mil.md)
