# 02 — OpenSlide basics (open a WSI)

**Goal:** open one `.svs` (or similar) file, understand the **image pyramid**, save a thumbnail and one 256×256 patch.

**Libraries:** `openslide-python`, `Pillow`  
**Snippet:** `python snippets/t02_open_one_slide.py` (after you point it at a slide path)

---

## Why OpenSlide?

A whole-slide image can be **tens of thousands of pixels** on each side. You cannot load it like a normal PNG into one array.

OpenSlide lets you:

1. Read **metadata** (size, microns-per-pixel, objective power)
2. Use a **pyramid** of zoom levels (full res + smaller copies)
3. Read only a **region** (e.g. one 256×256 tile) when you need it

```text
Level 0  ████████████████  full resolution (huge)
Level 1  ████████          ~4× downsample
Level 2  ████              smaller
...
```

---

## Practice script (see `snippets/t02_open_one_slide.py`)

Key steps:

```python
from pathlib import Path
import openslide

slide_path = Path("/path/to/your/slide.svs")
slide = openslide.OpenSlide(str(slide_path))

print(slide.dimensions)              # (width, height) at level 0
print(slide.level_count)
print(list(slide.level_downsamples))

thumb = slide.get_thumbnail((1024, 1024))
thumb.save("thumb.png")

level = 1 if slide.level_count > 1 else 0
w0, h0 = slide.dimensions
x0, y0 = int(w0 * 0.4), int(h0 * 0.4)
patch = slide.read_region((x0, y0), level, (256, 256)).convert("RGB")
patch.save("patch.png")
slide.close()
```

---

## OpenSlide coordinate rule

```text
read_region(location, level, size)
           ▲          ▲      ▲
           │          │      └─ size in pixels AT THAT LEVEL
           │          └─ which pyramid level to decode from
           └─ (x,y) in LEVEL-0 coordinates (even if level≠0)
```

---

## Common mistakes

1. Loading the full WSI with `PIL.Image.open` — too big.  
2. Using level-1 coordinates in `read_region` — must use level-0 coords.  
3. Forgetting `.convert("RGB")` — many encoders expect 3 channels.  
4. Missing system OpenSlide library — install per [OpenSlide docs](https://openslide.org/download/).

---

## Try it

1. Put one de-identified `.svs` under `data/slides/` (not committed to git).  
2. Run `python snippets/t02_open_one_slide.py`.  
3. Open outputs in `snippets/_out/`.

Next: [03_tissue_mask_and_tiling.md](03_tissue_mask_and_tiling.md)
