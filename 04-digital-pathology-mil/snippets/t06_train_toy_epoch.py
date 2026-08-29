#!/usr/bin/env python3
"""Toy Attention-MIL training loop (synthetic demo)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
LABEL2ID = {"class_a": 0, "class_b": 1}


class AttentionMIL(nn.Module):
    def __init__(self, dim: int = 768, n_classes: int = 2):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.clf = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor):
        a = self.attn(x)
        w = torch.softmax(a, dim=0)
        slide = (w * x).sum(dim=0)
        return self.clf(slide), w.squeeze(-1)


def load_few_bags(max_n: int = 8):
    bags = []
    for p in sorted(FEAT.glob("*.pt"))[:max_n]:
        d = torch.load(p, map_location="cpu", weights_only=False)
        label = str(d["label"])
        bags.append((d["features"].float(), LABEL2ID[label], d.get("slide_id", p.stem), label))
    return bags


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bags = load_few_bags(8)
    if not bags:
        print("No feature bags found — using random toy bags")
        bags = [
            (torch.randn(50, 768), 0, "toy0", "class_a"),
            (torch.randn(80, 768), 1, "toy1", "class_b"),
            (torch.randn(60, 768), 0, "toy2", "class_a"),
            (torch.randn(90, 768), 1, "toy3", "class_b"),
        ]

    model = AttentionMIL().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    print("device:", device, "n_bags:", len(bags))
    model.train()
    for epoch in range(3):
        total = 0.0
        correct = 0
        for feats, y, sid, lab in bags:
            x = feats.to(device)
            target = torch.tensor([y], device=device)
            logits, _attn = model(x)
            loss = crit(logits.unsqueeze(0), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            pred = int(logits.argmax().item())
            correct += int(pred == y)
        print(
            f"epoch {epoch+1}: mean_loss={total/len(bags):.4f} acc={correct}/{len(bags)}"
        )

    out = ROOT / "snippets" / "_out"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "toy_attention_mil.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print("wrote", ckpt)


if __name__ == "__main__":
    main()
