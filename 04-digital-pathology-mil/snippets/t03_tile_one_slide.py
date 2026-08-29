#!/usr/bin/env python3
"""Tissue-aware 256×256 tiling on ONE slide (demo)."""

from pathlib import Path

import numpy as np
import openslide

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snippets" / "_out"
OUT.mkdir(parents=True, exist_ok=True)

TILE = 256
WHITE_THR = 220
MIN_TISSUE_FRAC = 0.15
TARGET_DS = 4.0


def tissue_frac(rgb) -> float:
    arr = np.asarray(rgb)
    white = (arr[:, :, 0] > WHITE_THR) & (arr[:, :, 1] > WHITE_THR) & (arr[:, :, 2] > WHITE_THR)
    return float((~white).mean())


slides_dir = ROOT / "data" / "slides"
svs = slides_dir / "example.svs"
if not svs.is_file():
    hits = list(slides_dir.glob("*.svs")) + list(slides_dir.glob("*.ndpi"))
    if not hits:
        raise SystemExit("Add a de-identified slide under data/slides/")
    svs = hits[0]

slide = openslide.OpenSlide(str(svs))
downs = list(slide.level_downsamples)
level = min(range(len(downs)), key=lambda i: abs(downs[i] - TARGET_DS))
ds = float(downs[level])
lw, lh = slide.level_dimensions[level]

coords = []
for ly in range(0, lh - TILE + 1, TILE):
    for lx in range(0, lw - TILE + 1, TILE):
        x0, y0 = int(lx * ds), int(ly * ds)
        patch = slide.read_region((x0, y0), level, (TILE, TILE)).convert("RGB")
        if tissue_frac(patch) >= MIN_TISSUE_FRAC:
            coords.append((x0, y0))

print("slide:", svs.name)
print("level:", level, "downsample:", ds)
print("grid_possible:", (lw // TILE) * (lh // TILE))
print("kept_tissue_tiles:", len(coords))

# save a few example tiles
for i, (x0, y0) in enumerate(coords[:6]):
    patch = slide.read_region((x0, y0), level, (TILE, TILE)).convert("RGB")
    patch.save(OUT / f"{svs.stem}_tile{i}_{x0}_{y0}.png")
print("wrote sample tiles to", OUT)
slide.close()
