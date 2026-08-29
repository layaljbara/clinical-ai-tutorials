# 04 — Tile encoder features (foundation models)

## What this step does

Each RGB tile → one fixed-length vector (e.g. **768-D** or **1024-D**) from a **pretrained** encoder. The encoder is usually **frozen** when you train a MIL head.

Common encoder families (pick one for your project):

| Type | Examples | Notes |
|------|----------|-------|
| ImageNet CNN / ViT | ResNet, ViT via `timm` | Baseline; not histology-specific |
| Pathology foundation models | UNI, Virchow, CTransPath-style models | Stronger for H&E |
| Custom checkpoint | Your lab’s released weights | Follow that repo’s transforms |

This tutorial is **encoder-agnostic** — swap weights and input size per the model card.

## Typical forward pass

```python
import torch
from torchvision import transforms

# Example: 224×224 input, 768-D output (dimensions vary by model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = ...  # load pretrained encoder; model.eval()
tile_tensor = transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]

with torch.no_grad():
    embedding = model(tile_tensor)   # shape depends on architecture
```

Always match **input size**, **normalization**, and **weight loading** to the model you chose.

## Save a feature bag per slide

After embedding all tiles on one slide:

```python
bag = {
    "slide_id": "example_001",
    "label": "class_a",
    "features": torch.stack(tile_embeddings),  # [N, D]
    "coords": coords_list,
    "n_tiles": N,
}
torch.save(bag, "features/example_001.pt")
```

## Practice

```bash
python snippets/t04_embed_one_tile.py
```

Uses a **placeholder** path to encoder weights — replace with your model.

Next: [05_loading_bags_and_splits.md](05_loading_bags_and_splits.md)
