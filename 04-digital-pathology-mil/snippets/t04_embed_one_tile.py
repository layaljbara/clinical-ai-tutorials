#!/usr/bin/env python3
"""Embed one RGB tile with a generic timm encoder (demo)."""

from __future__ import annotations

from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "snippets" / "_out"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer a patch saved by t02/t03; else use a solid-color demo tile
candidates = list(OUT.glob("*_patch*.png")) + list(OUT.glob("*_tile*.png"))
if candidates:
    img_path = candidates[0]
else:
    img_path = OUT / "demo_tile.png"
    Image.new("RGB", (256, 256), color=(180, 120, 140)).save(img_path)

# Example: small ImageNet-pretrained ViT (swap for your pathology FM)
model_name = "vit_tiny_patch16_224"
model = timm.create_model(model_name, pretrained=True, num_classes=0)
model.eval()

tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0).to(device)
with torch.no_grad():
    emb = model(x)

print("model:", model_name)
print("image:", img_path)
print("device:", device)
print("embedding_shape:", tuple(emb.shape))
print("embedding_norm:", float(emb.float().norm()))
torch.save({"path": str(img_path), "emb": emb.cpu()}, OUT / "one_tile_emb.pt")
print("wrote", OUT / "one_tile_emb.pt")
