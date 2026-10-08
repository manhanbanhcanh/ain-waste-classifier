# Explanation of `train()` Function in `student_core.py`

## Overview
The `train()` function implements a complete training loop for a PyTorch model on the TrashNet waste classification dataset. It handles data loading, model training, validation, and returns a history dictionary for plotting training/validation curves.

---

## Function Signature
```python
def train(
    model: Any,
    train_dir: Path,
    validation_dir: Path,
    *,
    epochs: int = 10,
    **kwargs: Any,
) -> Any:
```

### Parameters
- `model`: An untrained PyTorch model (e.g., from `build_model()`).
- `train_dir`: Path to directory containing training images, organized in subfolders per class.
- `validation_dir`: Path to directory containing validation images, same structure as `train_dir`.
- `epochs`: Number of training epochs (default: 10).
- `**kwargs`: Additional optional arguments:
  - `batch_size`: Batch size for DataLoaders (default: 32).
  - `lr`: Learning rate for the Adam optimizer (default: 0.001).

### Returns
- A dictionary `history` containing lists of per-epoch metrics:
  - `'train_loss'`: Training loss per epoch.
  - `'train_acc'`: Training accuracy per epoch.
  - `'val_loss'`: Validation loss per epoch.
  - `'val_acc'`: Validation accuracy per epoch.

---

## Step‑by‑Step Implementation

### 1. Import Heavy Libraries Inside the Function
```python
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader
```
* **Why inside?** To keep the module import‑light so that sample tests and sanity checks run without requiring PyTorch/torchvision to be installed.

### 2. Set Up Device (GPU/CPU)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
- Moves the model to GPU if available, otherwise CPU.

### 3. Define Image Transformations
```python
transform = transforms.Compose([
    transforms.Resize((model.input_shape[0], model.input_shape[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1,1]
])
```
- Resizes images to the shape expected by the model.
- Converts PIL images to PyTorch tensors.
- Normalizes pixel values from [0,1] to [-1,1] using mean=0.5, std=0.5.

### 4. Load Datasets and Create DataLoaders
```python
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=transform)

batch_size = kwargs.get('batch_size', 32)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
```
- `datasets.ImageFolder` expects directory structure: `root/class_x/xxx.jpg`.
- `DataLoader` handles batching, shuffling (for training), and iteration.

### 5. Define Loss Function and Optimizer
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001))
```
- `CrossEntropyLoss` is standard for multi‑class classification (combines softmax + NLL loss).
- Adam optimizer with learning rate from `kwargs` (default 0.001).

### 6. Initialize Training History
```python
history: dict[str, list[float]] = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}
```
- Stores metrics for each epoch to enable plotting later.

### 7. Training Loop Over Epochs
```python
for epoch in range(epochs):
    model.train()  # set model to training mode
    ...            # training phase (see below)
    model.eval()   # set model to evaluation mode
    ...            # validation phase (see below)
```
- **`model.train()`** enables layers like dropout and batch norm to behave appropriately for training.
- **`model.eval()`** disables them for validation/inference.

#### A. Training Phase
```python
running_loss = 0.0
correct_train = 0
total_train = 0

for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total_train += labels.size(0)
    correct_train += (predicted == labels).sum().item()

train_loss = running_loss / len(train_loader)
train_acc = 100 * correct_train / total_train
```
- Zero gradients, forward pass, compute loss, backward pass, optimize.
- Track loss and accuracy across batches.

#### B. Validation Phase
```python
running_loss = 0.0
correct_val = 0
total_val = 0
with torch.no_grad():  # no gradient tracking to save memory/computation
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

validation_loss = running_loss / len(validation_loader)
validation_acc = 100 * correct_val / total_val
```
- Same as training but without backpropagation; we only compute loss and accuracy.

### 8. Record History and Print Progress
```python
history['train_loss'].append(train_loss)
history['train_acc'].append(train_acc)
history['val_loss'].append(validation_loss)
history['val_acc'].append(validation_acc)

print(f'Epoch {epoch+1}/{epochs}: '
      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
      f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%')
```

### 9. Return History
After all epochs, return the `history` dictionary for external plotting.

---

## Usage Example
```python
model = build_model(input_shape=(128, 128, 3), num_classes=6)
history = train(
    model,
    train_dir=Path("data/train"),
    validation_dir=Path("data/validation"),
    epochs=20,
    batch_size=64,
    lr=0.0005
)
# Plot history['train_acc'] vs history['val_acc'] etc.
```

---

## Compliance Notes
- All heavy imports (torch, torchvision) are inside the function, satisfying the requirement that the module can be imported without deep‑learning frameworks installed.
- The function returns a simple dictionary (not a framework‑specific History object) so it works with any PyTorch‑based model and can be easily plotted with matplotlib or similar.
