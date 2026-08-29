#!/usr/bin/env python3
"""Evaluate a saved MIL checkpoint on val/test bags (generic paths)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LABEL2ID = {"class_a": 0, "class_b": 1}


class AttentionMIL(nn.Module):
    def __init__(self, dim=768, attn_dim=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, attn_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(attn_dim, 1)
        )
        self.clf = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, x):
        a = self.attn(x)
        w = torch.softmax(a, dim=0)
        slide = (w * x).sum(dim=0)
        return self.clf(slide), w.squeeze(-1)


def load_split(name: str, splits_dir: Path, feat_dir: Path):
    items = []
    with (splits_dir / f"{name}.csv").open() as f:
        for row in csv.DictReader(f):
            sid = row.get("slide_id") or row.get("ID")
            p = feat_dir / f"{sid}.pt"
            if not p.is_file():
                continue
            bag = torch.load(p, map_location="cpu", weights_only=False)
            items.append((bag["features"].float(), LABEL2ID[str(bag["label"])], sid))
    return items


@torch.no_grad()
def eval_split(model, items, device):
    ys, ps, preds = [], [], []
    for feats, y, _sid in items:
        logits, _ = model(feats.to(device))
        prob = torch.softmax(logits, dim=0)[1].item()
        pred = int(logits.argmax().item())
        ys.append(y)
        ps.append(prob)
        preds.append(pred)
    y = np.asarray(ys)
    p = np.asarray(ps)
    pred = np.asarray(preds)
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    return {
        "n": len(y),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall_class_a": float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
        "recall_class_b": float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to model checkpoint")
    ap.add_argument("--feat-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = ap.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    drop = ckpt.get("args", {}).get("dropout", 0.25)
    model = AttentionMIL(dropout=drop).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("loaded", args.ckpt.name, "device=", device)

    for split in ("val", "test"):
        items = load_split(split, args.splits_dir, args.feat_dir)
        if not items:
            print(split, "no bags found — skip")
            continue
        m = eval_split(model, items, device)
        print(split, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
