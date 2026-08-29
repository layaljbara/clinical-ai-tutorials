#!/usr/bin/env python3
"""Open one SVS, print pyramid info, save thumbnail + sample patch."""

from pathlib import Path

import openslide

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snippets" / "_out"
OUT.mkdir(parents=True, exist_ok=True)

slides_dir = ROOT / "data" / "slides"
svs = slides_dir / "example.svs"
if not svs.is_file():
    hits = list(slides_dir.glob("*.svs")) + list(slides_dir.glob("*.ndpi"))
    if not hits:
        raise SystemExit(
            "No WSI found. Add a de-identified slide under data/slides/ (not committed to git)."
        )
    svs = hits[0]

slide = openslide.OpenSlide(str(svs))
print("file:", svs.name)
print("level0_WH:", slide.dimensions)
print("n_levels:", slide.level_count)
print("downsamples:", [round(float(d), 4) for d in slide.level_downsamples])
print("mpp_x:", slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
print("objective:", slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))

thumb = slide.get_thumbnail((1024, 1024))
thumb_path = OUT / f"{svs.stem}_thumb.png"
thumb.save(thumb_path)
print("wrote", thumb_path)

level = 1 if slide.level_count > 1 else 0
ds = float(slide.level_downsamples[level])
w0, h0 = slide.dimensions
x0, y0 = int(w0 * 0.4), int(h0 * 0.4)
patch = slide.read_region((x0, y0), level, (256, 256)).convert("RGB")
patch_path = OUT / f"{svs.stem}_patch_L{level}.png"
patch.save(patch_path)
print("wrote", patch_path, "level", level, "ds", ds)
slide.close()
