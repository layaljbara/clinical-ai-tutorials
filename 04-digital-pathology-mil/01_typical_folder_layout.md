# 01 — Typical folder layout (example)

Projects vary, but most WSI + MIL pipelines look **something** like this:

```text
your_project/
├── data/
│   ├── labels.csv          # slide_id, label, optional metadata
│   ├── slides/             # .svs / .ndpi paths or symlinks
│   └── splits/             # train.csv, val.csv, test.csv (frozen)
├── preprocessing/
│   ├── scripts/            # tiling, QC, feature extraction
│   └── outputs/
│       ├── tiles/          # optional saved PNG tiles
│       └── features/       # one .pt bag per slide: {slide_id}.pt
├── models/
│   ├── encoders/           # downloaded FM weights (not in git)
│   └── mil/                # training scripts + checkpoints
├── notebooks/              # exploration
└── docs/                   # methods notes
```

## What each split file usually contains

| Column | Meaning |
|--------|---------|
| `slide_id` | Unique slide identifier |
| `label` | Slide-level class (e.g. `benign`, `malignant`) |
| `split` | `train` / `val` / `test` |
| `wsi_path` | Path to the whole-slide file |

## Feature bag (`.pt`) — common keys

| Key | Meaning |
|-----|---------|
| `slide_id` | Identifier |
| `label` | String or int class |
| `features` | `FloatTensor [N_tiles, D]` |
| `coords` | Optional tile coordinates on the slide |
| `n_tiles` | `N` (varies per slide) |

`D` depends on the encoder (768, 1024, 1536, …).

## Good habits

1. **Freeze splits** before training — do not peek at test labels while tuning.  
2. **Keep encoders frozen** when learning a small MIL head (unless you have a deliberate fine-tune plan).  
3. **Never commit** WSIs, patient IDs, or raw bags with real data to a public repo.

Next: [02_openslide_basics.md](02_openslide_basics.md)
