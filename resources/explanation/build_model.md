# Explanation of WasteNet Class

## Overview
`WasteNet` is a simple convolutional neural network (CNN) built with PyTorch's `nn.Module`. It takes an image tensor as input and outputs raw class scores (logits) for waste classification.

---

## 1. Class Definition & Inheritance
```python
class WasteNet(nn.Module):
```
- Inherits from `nn.Module`, gaining built‑in functionality for parameter management, GPU movement, saving/loading, etc.

---

## 2. Constructor (`__init__`)
```python
def __init__(self, input_shape: tuple[int, int, int], num_classes: int) -> None:
    super().__init__()
    self.input_shape = input_shape
    self.num_classes = num_classes
    h, w, c = input_shape  # height, width, channels
```
### Parameters
- `input_shape`: Expected shape of a single sample **without** batch dimension, e.g., `(H, W, C)`.
  - `H`: image height (pixels)
  - `W`: image width
  - `C`: number of channels (1 for grayscale, 3 for RGB)
- `num_classes`: Number of output categories.

### What happens
- Calls `super().__init__()` to initialize the parent module.
- Stores `input_shape` and `num_classes` as instance attributes.
- Unpacks the shape into `h`, `w`, `c` for readability when defining layers.

---

## 3. Feature Extractor (Convolutional Blocks)
```python
self.conv1 = nn.Conv2d(c, 16, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.pool = nn.MaxPool2d(2, 2)
```
- **Three convolutional layers** gradually increase feature depth:
  - `conv1`: `c → 16` filters
  - `conv2`: `16 → 32` filters
  - `conv3`: `32 → 64` filters
- Each uses a 3×3 kernel with padding=1 to preserve spatial size before pooling.
- **`self.pool`**: 2×2 max‑pool with stride 2, applied after each convolution.
  - Reduces height and width by half each time → total spatial reduction factor of \(2^3 = 8\).
  - Provides translation invariance and cuts computation.

---

## 4. Classifier (Fully‑Connected Layers)
```python
self.fc1 = nn.Linear(64 * (h // 8) * (w // 8), 128)
self.fc2 = nn.Linear(128, num_classes)
self.dropout = nn.Dropout(0.5)
```
- After three poolings, the feature map size is `[batch, 64, h//8, w//8]`.
- Flattened length = `64 * (h // 8) * (w // 8)` → input size of `fc1`.
- **`fc1`**: Projects the flattened features to a 128‑dimensional bottleneck.
- **`fc2`**: Maps the 128‑dim representation to `num_classes` logits (raw scores).
- **`self.dropout`**: Randomly zeroes 50% of activations during training only, reducing over‑fitting.

---

## 5. Forward Pass (`forward`)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, 64 * (self.input_shape[0] // 8) * (self.input_shape[1] // 8))
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
```
### Step‑by‑step
1. **Block 1**: `conv1` → ReLU → max‑pool.
2. **Block 2**: `conv2` → ReLU → max‑pool.
3. **Block 3**: `conv3` → ReLU → max‑pool.
   - After these three blocks: shape `[B, 64, h//8, w//8]`.
4. **Flatten**: `x.view(-1, ...)` → `[B, 64 * (h//8) * (w//8)]`.
5. **Dense 1**: `fc1` → ReLU.
6. **Dropout**: applied only when `model.train()` is active.
7. **Output**: `fc2` → logits of shape `[B, num_classes]`.

The caller typically feeds these logits to a loss like `nn.CrossEntropyLoss` (which applies softmax internally) or applies `torch.softmax` manually for probabilities.

---

## 6. Typical Usage Pattern
```python
model = WasteNet(input_shape=(128, 128, 3), num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for images, labels in dataloader:   # images: [B, C, H, W]
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```
- Input tensor must be **4‑D**: `[batch_size, channels, height, width]`.
- If data is in `[B, H, W, C]` format, permute with `images.permute(0, 3, 1, 2)` before feeding.

---

## 7. Design Summary
| Component | Purpose | Common tweaks |
|-----------|---------|---------------|
| Conv layers (3×3, pad=1) | Extract local spatial features | Number of filters, kernel size, depth |
| MaxPool (2×2) | Down‑sample, add invariance | Pool size, stride, or use adaptive pooling |
| FC1 | Combine high‑level info into compact vector | Output size (128) |
| Dropout | Regularization | Dropout probability (0.5) |
| FC2 | Produce class logits | Must equal `num_classes` |
| ReLU activations | Non‑linearity | LeakyReLU, ELU, etc. |

This network is a compact CNN suitable for modest image sizes (e.g., 64×64, 128×128) and a limited number of classes. For larger images or more complex tasks, consider adding more blocks, batch normalization, or using a pretrained backbone.
