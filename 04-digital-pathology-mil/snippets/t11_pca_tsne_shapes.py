#!/usr/bin/env python3
"""Mean-pool slide bags → PCA → t-SNE scatter (visualization only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", type=Path, default=ROOT / "data" / "features")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--max-slides", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "snippets" / "_out" / "embedding_tsne.png")
    args = ap.parse_args()

    bags = sorted(args.feat_dir.glob("*.pt"))[: args.max_slides]
    if not bags:
        raise SystemExit(f"No .pt bags in {args.feat_dir}")

    X_list, labels = [], []
    for p in bags:
        bag = torch.load(p, map_location="cpu", weights_only=False)
        feats = bag["features"]
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats)
        X_list.append(feats.float().mean(dim=0).numpy())
        labels.append(str(bag.get("label", "unknown")))

    X = np.stack(X_list)
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    Xp = PCA(n_components=n_comp).fit_transform(X)
    XY = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42).fit_transform(Xp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(XY[idx, 0], XY[idx, 1], label=lab, alpha=0.75)
    plt.legend()
    plt.title("Mean-pooled tile embeddings (visualization only)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print("wrote", args.out, "n_slides=", len(labels))


if __name__ == "__main__":
    main()
