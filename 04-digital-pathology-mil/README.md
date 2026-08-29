# 04 — Digital pathology & multiple-instance learning (MIL)

General tutorials on **whole-slide imaging (WSI)**, **OpenSlide**, **tiling**, **pathology foundation-model features**, and **attention-based MIL** — not tied to any one dataset, encoder brand, or research project.

## Who this is for

- Beginners learning the WSI → tile → embedding → slide-classifier pipeline  
- Anyone comparing OpenSlide, PyTorch, and MIL concepts before starting their own project  

## Reading order

1. [`00_roadmap.md`](00_roadmap.md) — pipeline overview  
2. [`01_typical_folder_layout.md`](01_typical_folder_layout.md) — how projects usually organize data  
3. [`02_openslide_basics.md`](02_openslide_basics.md) + [`snippets/t02_open_one_slide.py`](snippets/t02_open_one_slide.py)  
4. [`03_tissue_mask_and_tiling.md`](03_tissue_mask_and_tiling.md) + [`snippets/t03_tile_one_slide.py`](snippets/t03_tile_one_slide.py)  
5. [`04_tile_encoder_features.md`](04_tile_encoder_features.md) + [`snippets/t04_embed_one_tile.py`](snippets/t04_embed_one_tile.py)  
6. [`05_loading_bags_and_splits.md`](05_loading_bags_and_splits.md) + [`snippets/t05_inspect_bag.py`](snippets/t05_inspect_bag.py)  
7. [`06_training_mil_head.md`](06_training_mil_head.md) + [`snippets/t06_train_toy_epoch.py`](snippets/t06_train_toy_epoch.py)  
8. [`07_training_walkthrough.md`](07_training_walkthrough.md) + [`snippets/t07_eval_checkpoint.py`](snippets/t07_eval_checkpoint.py)  
9. [`08_reading_metrics.md`](08_reading_metrics.md) + [`snippets/t08_plot_training_curves.py`](snippets/t08_plot_training_curves.py)  
10. [`09_foundation_models_and_mil.md`](09_foundation_models_and_mil.md) — concepts (any FM + MIL head)  
11. [`10_jupyter_lab_remote_gpu.md`](10_jupyter_lab_remote_gpu.md) — Jupyter on a GPU server  
12. [`11_embedding_visualization.md`](11_embedding_visualization.md) + [`snippets/t11_pca_tsne_shapes.py`](snippets/t11_pca_tsne_shapes.py)  

## Libraries covered

| Library / idea | Role |
|----------------|------|
| **OpenSlide** | Read multi-resolution `.svs` / `.ndpi` whole-slide images |
| **Pillow / NumPy** | Tile extraction and masks |
| **PyTorch** | Feature tensors, MIL modules, training |
| **timm / custom encoders** | Load a pretrained tile encoder (ImageNet, pathology FMs, etc.) |
| **scikit-learn** | PCA / t-SNE for embedding plots (visualization only) |

Bring your **own slides and labels** under an approved data agreement. This repo has **no whole-slide images**.
