# 07 — Training walkthrough (generic)

This section describes a **typical** MIL training loop — not one specific dataset or checkpoint.

## Inputs

- Feature bags: `features/{slide_id}.pt`  
- Splits: `data/splits/{train,val,test}.csv`  
- Frozen tile encoder (already used to build bags)

## Loop outline

1. Build `Dataset` that returns one bag per slide id.  
2. Train `AttentionMIL` for several epochs; monitor validation metric.  
3. Save best checkpoint: `models/mil/best.pt`.  
4. Run inference on test split once.  
5. Write `metrics.json` and optional `predictions.csv`.

## Sketch

```python
for epoch in range(num_epochs):
    model.train()
    for bag in train_loader:
        logits, _ = model(bag["features"])
        loss = criterion(logits.unsqueeze(0), bag["label_id"])
        ...
    val_score = evaluate(model, val_loader)
    if val_score > best:
        torch.save(model.state_dict(), "models/mil/best.pt")
```

## Class imbalance

If classes are imbalanced, try:

```python
weights = torch.tensor([w0, w1], device="cuda")
criterion = nn.CrossEntropyLoss(weight=weights)
```

## Evaluate saved checkpoint

```bash
python snippets/t07_eval_checkpoint.py
```

Next: [08_reading_metrics.md](08_reading_metrics.md)
