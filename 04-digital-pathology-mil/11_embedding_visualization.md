# 11 — Embedding visualization (PCA / t-SNE)

**Goal:** See whether **mean-pooled tile embeddings** separate classes in 2D — for **exploration only**, not as your classifier.

## Pipeline

```text
Each slide: bag [N, D]
    → mean-pool → slide vector [D]
    → PCA (optional, reduce D → ~50)
    → t-SNE → 2D (x, y)
    → scatter, color by class label
```

t-SNE **does not use labels** to place points — labels only **color** the plot.

## Why mean-pool first?

Each slide has different `N`. Mean-pooling gives one fixed-length vector per slide so you can stack slides into a matrix `[n_slides, D]`.

## Sketch

```python
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

slide_vectors = []  # list of [D] tensors
labels = []

for path in bag_paths:
    bag = torch.load(path, map_location="cpu", weights_only=False)
    slide_vectors.append(bag["features"].mean(dim=0).numpy())
    labels.append(bag["label"])

X = np.stack(slide_vectors)
X50 = PCA(n_components=min(50, X.shape[0]-1)).fit_transform(X)
XY = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X50)

for lab in sorted(set(labels)):
    idx = [i for i, l in enumerate(labels) if l == lab]
    plt.scatter(XY[idx, 0], XY[idx, 1], label=lab, alpha=0.7)
plt.legend()
plt.title("Mean-pooled embeddings (visualization only)")
plt.savefig("embedding_tsne.png", dpi=150)
```

## Compare encoders fairly

Run the **same** PCA → t-SNE recipe for each encoder (ResNet baseline vs pathology FM). This compares **embedding spaces**, not your trained MIL heads.

## Script

```bash
python snippets/t11_pca_tsne_shapes.py --feat-dir /path/to/features --max-slides 80
```

## Caveats

1. t-SNE is non-linear and stochastic — do not over-interpret local distances.  
2. Separation in a plot ≠ clinical performance — always report held-out test metrics from your MIL model.  
3. Labels must not leak into unsupervised steps if you claim “unsupervised structure.”

Back to: [README.md](README.md)
