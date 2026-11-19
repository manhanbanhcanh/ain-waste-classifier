# WasteNet Model Documentation

**Visual Waste Sorting Assistant — Project 11**  
**Group 9: Duc Manh, Trang Quynh, Bao Chi | Class: 66FIT3AIN.TT02**

---

## Model Overview

The project now supports **three DRAMATICALLY DIFFERENT model variants** that can be selected in the Streamlit UI:

| Variant | Name | Architecture | Parameters | Inference (CPU) | Target Accuracy |
|---------|------|--------------|------------|-----------------|-----------------|
| `lite` | ⚡ WasteNet-Lite | 1 Conv block + GlobalAvgPool | **~5K** | **~5ms** | >70% |
| `standard` | ⚖️ WasteNet-Standard | 3 Conv blocks + Dual FC | **~4.3M** | **~50ms** | >85% |
| `smart` | 🧠 WasteNet-Smart | 5-Stage Deep Residual + SE Attention | **~15M** | **~150ms** | >95% |

**Common Properties:**
- **Framework:** PyTorch
- **Input Shape:** 128×128×3 (RGB)
- **Output Classes:** 6 (TrashNet)
- **Training Data:** TrashNet dataset (2,527 images)
- **Task:** Waste classification for recycling stations
- **Model Files:** `models/best_model_lite.pt`, `models/best_model_standard.pt`, `models/best_model_smart.pt`

---

## Architecture Details

### Lite Variant (Ultra-Lightweight) — 5K params
```
Input: (3, 128, 128)
│
├─ Conv2d(3 → 16, 3×3, padding=1) + BN + ReLU + MaxPool2d(2)
│   └─ Output: (16, 64, 64)
│
├─ Conv2d(16 → 32, 3×3, padding=1) + BN + ReLU + MaxPool2d(2)
│   └─ Output: (32, 32, 32)
│
├─ GlobalAvgPool2d(1)
│   └─ Output: (32, 1, 1) → 32 features
│
├─ Dropout(0.3)
│
└─ Linear(32 → 6) → Logits
```

**Design:** Single feature extraction block (2 conv layers), Global Average Pooling eliminates large FC layers, minimal dropout. Designed for ultra-fast edge deployment.

### Standard Variant (Classic CNN) — 4.3M params
```
Input: (3, 128, 128)
│
├─ ConvBlock(3 → 16) + MaxPool2d(2)    # 64×64
├─ ConvBlock(16 → 32) + MaxPool2d(2)   # 32×32
├─ ConvBlock(32 → 64) + MaxPool2d(2)   # 16×16
│
├─ Flatten → 64 × 16 × 16 = 16,384 features
│
├─ Linear(16384 → 256) + ReLU + Dropout(0.5)
├─ Linear(256 → 128) + ReLU + Dropout(0.5)
│
└─ Linear(128 → 6) → Logits
```

**Design:** Three progressive convolutional blocks (16→32→64) with BatchNorm, dual large FC classifier with heavy dropout for regularization. Balanced speed/accuracy.

### Smart Variant (Deep Residual + Attention) — 15M params
```
Input: (3, 128, 128)
│
├─ Stem: Conv2d(3 → 64) + BN + ReLU + MaxPool2d(2)   # 64×64
│
├─ Stage 1: 64×64, 64 channels
│   ├─ ConvBlock(64 → 64)  (no pool)
│   ├─ ResidualBlock(64, SE) ×3
│   └─ MaxPool2d(2)                                    # 32×32
│
├─ Stage 2: 32×32, 128 channels
│   ├─ ConvBlock(64 → 128) (no pool)
│   ├─ ResidualBlock(128, SE) ×3
│   └─ MaxPool2d(2)                                    # 16×16
│
├─ Stage 3: 16×16, 256 channels
│   ├─ ConvBlock(128 → 256) (no pool)
│   ├─ ResidualBlock(256, SE) ×2
│   └─ MaxPool2d(2)                                    # 8×8
│
├─ Stage 4: 8×8, 512 channels
│   ├─ ConvBlock(256 → 512) (no pool)
│   ├─ ResidualBlock(512, SE) ×2
│   └─ MaxPool2d(2)                                    # 4×4
│
├─ GlobalAvgPool2d(1)
│   └─ Output: (512, 1, 1) → 512 features
│
├─ Dropout(0.4)
├─ Linear(512 → 1024) + ReLU + Dropout(0.4)
│
└─ Linear(1024 → 6) → Logits
```

**Key Components:**
- **ResidualBlock**: Two 3×3 convs + BN + ReLU + Identity skip + optional SE
- **SEBlock (Squeeze-and-Excitation)**: Channel attention with reduction ratio 16
- **ConvBlock**: Conv3×3 + BN + ReLU + optional MaxPool
- **GlobalAvgPool**: Replaces large FC layers, reduces parameters, improves generalization

### Key Design Choices by Variant

| Feature | Lite | Standard | Smart |
|---------|------|----------|-------|
| **Stages/Blocks** | 1 feature block | 3 ConvBlocks | 4 Stages (Stem + 4 stages) |
| **Residual Connections** | ❌ | ❌ | ✅ 10 blocks (3+3+2+2) |
| **SE Attention** | ❌ | ❌ | ✅ All residual blocks |
| **Batch Normalization** | ✅ (2 layers) | ✅ (all conv) | ✅ (everywhere) |
| **Filter Progression** | 16→32 | 16→32→64 | **64→64→128→256→512** |
| **Classifier** | GAP + Linear(32) | Flatten + FC(256→128→6) | GAP + FC(1024→6) |
| **Dropout** | 0.3 (single) | 0.5 (dual) | 0.4 (dual) |
| **Parameters** | **~5K** | **~4.3M** | **~15M** |

---

## TrashNet Classes (6)

| Index | Class | Description | Typical Items |
|-------|-------|-------------|---------------|
| 0 | `cardboard` | Flattened boxes & packaging paperboard | Shipping boxes, cereal boxes, paper tubes |
| 1 | `glass` | Bottles & jars (clear/colored) | Wine bottles, jam jars, beverage containers |
| 2 | `metal` | Cans & small metal containers | Soda cans, food tins, aluminum foil balls |
| 3 | `paper` | Loose paper, flyers, non-cardboard | Newspapers, magazines, office paper, receipts |
| 4 | `plastic` | Bottles, containers, rigid packaging | Water bottles, detergent bottles, yogurt cups |
| 5 | `trash` | Non-recyclable / landfill | Chip bags, candy wrappers, dirty packaging, broken ceramics |

**Class Order:** Fixed — do not reorder. Matches TrashNet dataset directory structure.

---

## Training Configuration

### Default Hyperparameters
```python
epochs = 20
batch_size = 32
learning_rate = 0.001
optimizer = Adam
loss = CrossEntropyLoss
device = CUDA (if available) else CPU
```

### Data Transforms
```python
transforms.Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1, 1]
])
```

### Dataset Structure Expected
```
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── validation/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

Run `python scripts/prepare_dataset.py --confirm` to create this structure from raw TrashNet data.

---

## Model Loading in App

The Streamlit app **automatically loads** the selected variant's model on startup:

```python
# In app.py — load_model(variant)
model_path = PROJECT_ROOT / "models" / f"best_model_{variant}.pt"

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    # ✅ Trained model loaded — shows green badge in sidebar
else:
    # ⚠️ Falls back to random weights — shows orange warning badge
    model.eval()
```

### Model Selection in UI
The app provides a **three-button selector** at the top of the page:
- ⚡ **Lite** — Ultra-fast (5ms), minimal accuracy (edge/mobile deployment)
- ⚖️ **Standard** — Balanced (50ms), good accuracy (general deployment)
- 🧠 **Smart** — Maximum accuracy (150ms), server deployment

Switching variants instantly reloads the corresponding cached model.

### To Deploy a Trained Model
1. Train using the **Train Model** tab in the app, OR run:
   ```bash
   # Train specific variant (recommended epochs)
   python tests/train.py --variant lite --full --epochs 20
   python tests/train.py --variant standard --full --epochs 30
   python tests/train.py --variant smart --full --epochs 50
   ```
2. Each variant saves to its own file: `models/best_model_{variant}.pt`
3. Restart the app — it will detect and load the trained weights for the selected variant

---

## Performance Expectations

| Metric | Lite | Standard | Smart |
|--------|------|----------|-------|
| **Parameters** | **~5K** | **~4.3M** | **~15M** |
| Model File Size | **~27 KB** | **~17 MB** | **~60 MB** |
| Validation Accuracy | >70% | >85% | >95% |
| Inference Time (CPU) | **~5ms** | **~50ms** | **~150ms** |
| Inference Time (GPU) | ~1ms | ~5ms | ~15ms |
| Best For | Edge devices, mobile, real-time | General purpose, balanced | Server, maximum accuracy |

### Known Limitations (from TrashNet)
- **Specular reflections** on glass/metal → misclassification
- **Crumpled/deformed** items → shape distortion
- **Low light/blur** → feature degradation
- **Novel items** not in TrashNet → low confidence
- **Class imbalance** → bias toward paper/plastic
- **Complex backgrounds** → interference

See `FAILURE_MODES` in `app.py` for full list with mitigations.

---

## Inference API (Python)

```python
import torch
from pathlib import Path
from starter.student_core import build_model, predict, CLASSES, MODEL_VARIANTS

# Load specific variant
variant = "smart"  # or "lite", "standard"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(input_shape=(128, 128, 3), num_classes=6, variant=variant)
model.load_state_dict(torch.load(f"models/best_model_{variant}.pt", map_location=device, weights_only=True))
model.to(device).eval()

# Predict
class_name, confidence = predict(model, Path("test_image.jpg"))
print(f"Predicted: {class_name} ({confidence*100:.1f}%)")

# All class confidences
from starter.app import get_all_confidences  # or copy the function
confidences = get_all_confidences(model, Path("test_image.jpg"), device)
for cls, conf in sorted(confidences.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {conf*100:.1f}%")
```

---

## Experiment Tracking

For ablation studies (required by project), modify in `student_core.py`:
- **Depth:** Add/remove Conv blocks or residual blocks in `WasteNet.__init__`
- **Filters:** Change filter progression (e.g., Smart: 64→128→256→512)
- **Attention:** Toggle `use_se=True/False` in ResidualBlock
- **Dropout:** Modify `nn.Dropout` rates
- **Learning Rate:** Pass `lr` to `train()`
- **Variant:** Compare `lite`, `standard`, `smart` architectures

Results saved to `midterm/result/` with learning rate comparisons.

---

## File Checklist for Deployment

```
models/
├── best_model_lite.pt       ← Lite variant weights (~27 KB)
├── best_model_standard.pt   ← Standard variant weights (~17 MB)
├── best_model_smart.pt      ← Smart variant weights (~60 MB)
└── MODEL.md                 ← This file

starter/
├── app.py                   ← Streamlit app (loads selected variant)
├── student_core.py          ← Model definitions & training logic
└── .streamlit/
    └── config.toml          ← Theme configuration

tests/
└── train.py                 ← CLI training script (supports --variant)

scripts/
└── prepare_dataset.py       ← Dataset preparation (downloads & splits TrashNet)
```

---

## Quick Start

```bash
# 1. Prepare dataset (one-time, downloads ~250MB)
python scripts/prepare_dataset.py --confirm

# 2. Train model (via app or CLI)
# Option A: Streamlit UI → Select variant → "Train Model" tab → Start Training
# Option B: CLI (train specific variant with recommended epochs)
python tests/train.py --variant lite --full --epochs 20
python tests/train.py --variant standard --full --epochs 30
python tests/train.py --variant smart --full --epochs 50

# 3. Run app
streamlit run starter/app.py

# 4. App shows model selector at top — click ⚡ Lite / ⚖️ Standard / 🧠 Smart to switch
```

---

## License & Credits

- **Dataset:** TrashNet (Gary Thung & Mindy Yang, 2016)
- **Architecture:** Custom WasteNet (Project 11 requirement)
- **Framework:** PyTorch, Streamlit
- **Course:** 66FIT3AIN.TT02 — AI Engineering

---

*Last updated: 2026-11-17 | Model version: Project 11 Final Milestone with 3 Dramatically Different Variants*