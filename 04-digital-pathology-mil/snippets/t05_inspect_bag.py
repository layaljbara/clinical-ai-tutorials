#!/usr/bin/env python3
"""Inspect one feature bag and summarize a features folder."""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"

pts = sorted(FEAT.glob("*.pt"))
if not pts:
    raise SystemExit(
        f"No bags in {FEAT}. Add example .pt files or run your extraction pipeline first."
    )

path = pts[0]
bag = torch.load(path, map_location="cpu", weights_only=False)
print("file:", path.name)
for k in ["slide_id", "label", "split", "n_tiles"]:
    if k in bag:
        print(f"  {k}: {bag[k]}")
print("  features.shape:", tuple(bag["features"].shape))

n_tiles = []
labels = {}
for p in pts:
    d = torch.load(p, map_location="cpu", weights_only=False)
    n_tiles.append(int(d.get("n_tiles", d["features"].shape[0])))
    lab = str(d.get("label", "unknown"))
    labels[lab] = labels.get(lab, 0) + 1

n_tiles.sort()
print("--- folder ---")
print("n_bags:", len(pts))
print("labels:", labels)
print("tiles min/median/max:", n_tiles[0], n_tiles[len(n_tiles) // 2], n_tiles[-1])
