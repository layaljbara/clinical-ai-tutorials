#!/usr/bin/env python3
"""Plot training curves + confusion matrix from metrics.json (generic labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=ROOT / "models" / "mil" / "metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "mil" / "figures")
    ap.add_argument("--class-a", default="class_a")
    ap.add_argument("--class-b", default="class_b")
    args = ap.parse_args()

    if not args.metrics.is_file():
        raise SystemExit(
            f"Missing {args.metrics}. Train a model first or pass --metrics PATH."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads(args.metrics.read_text())
    hist = m["history"]
    epochs = [h["epoch"] for h in hist]
    loss = [h["train_loss"] for h in hist]
    auroc = [h["val_auroc"] for h in hist]
    bal = [h["val_balanced_accuracy"] for h in hist]
    best_ep = m.get("best_epoch", epochs[-1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(epochs, loss, color="#264653", lw=2)
    axes[0].axvline(best_ep, color="#e76f51", ls="--", label=f"best ep={best_ep}")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, auroc, color="#2a9d8f", lw=2)
    axes[1].axvline(best_ep, color="#e76f51", ls="--")
    axes[1].set_title("Val AUROC")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, bal, color="#e9c46a", lw=2)
    axes[2].axvline(best_ep, color="#e76f51", ls="--")
    axes[2].set_title("Val balanced accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)

    fig.suptitle("Attention-MIL training curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "training_curves.png", dpi=150)
    plt.close()

    if "test" in m and "confusion_matrix" in m["test"]:
        cm = np.array(m["test"]["confusion_matrix"], dtype=float)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], [f"Pred {args.class_a}", f"Pred {args.class_b}"])
        ax.set_yticks([0, 1], [f"True {args.class_a}", f"True {args.class_b}"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_title("Test confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(args.out / "test_confusion.png", dpi=150)
        plt.close()

    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
