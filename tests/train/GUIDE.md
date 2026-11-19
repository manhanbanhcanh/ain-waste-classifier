# WasteNet Training Guide

## Quick Start

```bash
# Quick test (sample data)
python tests/train/train.py --variant standard --epochs 5

# Full training (recommended)
python tests/train/train.py --variant standard --full --epochs 30 --progress
```

## Prerequisites

For full training (`--full`), prepare dataset first:
```bash
python scripts/prepare_dataset.py --confirm
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `standard` | `lite`, `standard`, or `smart` |
| `--full` | off | Use full dataset from `data/train/`, `data/validation/` |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--progress` | off | Show progress bars (needs `tqdm`) |
| `--output` | `models/best_model_{variant}.pt` | Custom output path |
| `--seed` | `42` | Random seed |

## Model Variants

| Variant | Params | CPU Inference | Target Acc | Use Case |
|---------|--------|---------------|------------|----------|
| `lite` | ~5K | ~5ms | >70% | Edge/mobile |
| `standard` | ~4.3M | ~50ms | >85% | **Recommended** |
| `smart` | ~15M | ~150ms | >95% | Max accuracy |

## Common Commands

```bash
# Production standard model
python tests/train/train.py --variant standard --full --epochs 30 --progress

# Maximum accuracy (Smart)
python tests/train/train.py --variant smart --full --epochs 40 --progress

# Edge deployment (Lite)
python tests/train/train.py --variant lite --full --epochs 20 --progress

# CPU only
python tests/train/train.py --variant standard --full --epochs 30 --device cpu --progress

# Custom hyperparameters
python tests/train/train.py --variant smart --full --epochs 50 --batch-size 64 --lr 0.0001 --progress
```

## Output

Models saved to `models/best_model_{variant}.pt` - auto-loaded by Streamlit app.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Full dataset not found" | Run `python scripts/prepare_dataset.py --confirm` |
| No progress bars | `pip install tqdm` |
| CUDA OOM | Reduce `--batch-size` or use `--device cpu` |
| Low accuracy | Increase `--epochs`, try lower `--lr` |

## Using Models

```bash
# Streamlit app (auto-loads models)
streamlit run starter/app.py

# Python API
from starter.student_core import build_model, predict
import torch
model = build_model(variant="standard")
model.load_state_dict(torch.load("models/best_model_standard.pt", weights_only=True))
class_name, confidence = predict(model, "image.jpg")
```