# 06 — Training a MIL head (attention pooling)

## Idea

Each slide is a **bag** of tile vectors `[N, D]`. **Attention MIL** learns which tiles matter, pools them, then classifies at slide level.

```text
features [N, D]
    → attention weights [N]
    → weighted sum → slide vector [D]
    → linear layer → logits [num_classes]
```

The tile encoder stays **frozen**. Only attention + classifier train.

## Minimal attention MIL module

```python
import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, dim=768, n_classes=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.clf = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x: [N, D] one slide
        a = self.attn(x)
        w = torch.softmax(a, dim=0)
        slide = (w * x).sum(dim=0)
        logits = self.clf(slide)
        return logits, w.squeeze(1)
```

## One training step (sketch)

```python
model = AttentionMIL(dim=768, n_classes=2).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()

bag = torch.load("features/example_001.pt", weights_only=False)
x = bag["features"].cuda()
y = torch.tensor([LABEL2ID[bag["label"]]]).cuda()

logits, attn = model(x)
loss = crit(logits.unsqueeze(0), y)
opt.zero_grad()
loss.backward()
opt.step()
```

## Training checklist

1. Dataset: load bags for ids in `train.csv` / `val.csv`  
2. Because `N` varies, **batch size 1** per slide is simplest  
3. Track val AUROC / balanced accuracy  
4. Evaluate once on `test.csv` with a fixed seed  
5. Save checkpoint under `models/mil/`  

## Metrics to report

- AUROC, AUPRC  
- Balanced accuracy / per-class recall  
- Confusion matrix on test  

## Toy snippet

```bash
python snippets/t06_train_toy_epoch.py
```

Next: [07_training_walkthrough.md](07_training_walkthrough.md)
