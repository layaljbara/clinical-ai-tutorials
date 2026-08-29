# 03 — Tissue mask & tiling

## Goal

Cut the WSI into many **256×256** tiles at a chosen pyramid level, keep tiles that contain tissue, drop blank background.

## Why tile counts differ per slide

Tile count depends on:

1. Scan size  
2. Tissue vs background fraction  
3. Your thresholds (e.g. white threshold, minimum tissue fraction)

MIL training handles **variable-length bags** — different `N` per slide is normal.

## Algorithm (typical)

1. Pick a pyramid level (often ~4× downsample from level 0).  
2. Build a thumbnail; mark non-white pixels as tissue.  
3. Walk a **non-overlapping** grid of 256×256 on that level.  
4. Keep tiles whose tissue fraction ≥ threshold.  
5. Store level-0 top-left `(x, y)` for each kept tile.

## Choose pyramid level

```python
TARGET_DOWNSAMPLE = 4.0

def choose_level(slide, target_ds=TARGET_DOWNSAMPLE):
    downs = list(slide.level_downsamples)
    return min(range(len(downs)), key=lambda i: abs(downs[i] - target_ds))
```

## Tissue fraction (simple white filter)

```python
import numpy as np

WHITE_THR = 220
MIN_TISSUE_FRAC = 0.15

def tissue_fraction(rgb):
    arr = np.asarray(rgb)
    white = (arr[:,:,0] > WHITE_THR) & (arr[:,:,1] > WHITE_THR) & (arr[:,:,2] > WHITE_THR)
    return float((~white).mean())
```

## Non-overlapping grid (sketch)

```python
TILE = 256
level = choose_level(slide)
ds = slide.level_downsamples[level]
lw, lh = slide.level_dimensions[level]

for ly in range(0, lh - TILE + 1, TILE):
    for lx in range(0, lw - TILE + 1, TILE):
        x0, y0 = int(lx * ds), int(ly * ds)
        patch = slide.read_region((x0, y0), level, (TILE, TILE)).convert("RGB")
        if tissue_fraction(patch) >= MIN_TISSUE_FRAC:
            # save patch and/or coords for embedding step
            ...
```

## Practice

```bash
python snippets/t03_tile_one_slide.py
```

Next: [04_tile_encoder_features.md](04_tile_encoder_features.md)
