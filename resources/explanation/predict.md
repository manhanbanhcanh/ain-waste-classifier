# Explanation of `predict()` Function in `student_core.py`

## Overview
The `predict()` function performs a single‑image inference pass using a trained PyTorch model. It loads an image, applies the same preprocessing used during training, runs the model, converts the raw outputs to class probabilities via softmax, and returns the predicted class name along with its confidence score.

---

## Function Signature
```python
def predict(model: Any, image_path: Path) -> tuple[str, float]:
```

### Parameters
- `model`: A trained PyTorch model (e.g., the output of `train()`).
- `image_path`: Path to the image file to classify.

### Returns
- A tuple `(predicted_class, confidence)` where:
  - `predicted_class` is one of the class names in `CLASSES` (e.g., `"plastic"`).
  - `confidence` is a float in the range `[0, 1]` representing the model’s confidence in that prediction.

---

## Step‑by‑Step Implementation

### 1. Import Heavy Libraries Inside the Function
```python
import torch  
import torch.nn.functional as F  
from torchvision import transforms  
from PIL import Image  
```
* **Why inside?** Keeps the module import‑light so that sample tests and sanity checks work without requiring PyTorch/torchvision/Pillow to be installed.

### 2. Set Up Device (GPU/CPU) and Put Model in Evaluation Mode
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```
- Moves the model to the appropriate device.
- `model.eval()` disables training‑specific behaviors (e.g., dropout, batch norm uses running statistics).

### 3. Define Image Transformations (Must Match Training)
```python
transform = transforms.Compose([
    transforms.Resize((model.input_shape[0], model.input_shape[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```
- The transform **must be identical** to the one used during training to ensure consistent input distribution.
  - Resize to the model’s expected height and width.
  - Convert the PIL image to a PyTorch tensor (scales pixel values to `[0, 1]`).
  - Normalize to `[-1, 1]` using mean=0.5, std=0.5.

### 4. Load and Preprocess the Image
```python
image = Image.open(image_path).convert('RGB')
transformed_image = transform(image)  
image_tensor = transformed_image.unsqueeze(0)    # Add batch dimension
image_tensor = image_tensor.to(device)  
```
- `Image.open()` loads the image from disk.
- `.convert('RGB')` ensures the image has three channels (even if it was grayscale or had an alpha channel).
- `transform(image)` applies the resizing, tensor conversion, and normalization.
- `.unsqueeze(0)` adds a batch dimension, turning shape `[C, H, W]` into `[1, C, H, W]`.
- `.to(device)` moves the tensor to the same device as the model.

### 5. Make Prediction (Inference)
```python
with torch.no_grad():  # Disable gradient computation to save memory and speed up
    outputs = model(image_tensor)          # Raw logits from the model
    probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
    confidence, predicted_idx = torch.max(probabilities, 1)  # Get highest probability and its index
```
- `torch.no_grad()` context prevents PyTorch from tracking operations for autograd.
- `model(image_tensor)` performs a forward pass.
- `F.softmax(outputs, dim=1)` converts the logits to a probability distribution over classes.
- `torch.max(probabilities, 1)` returns the maximum probability value and the class index along dimension 1 (the class dimension).

### 6. Convert Index to Class Name and Return
```python
predicted_class: str = CLASSES[predicted_idx.item()]  
confidence_score: float = confidence.item()  

return (predicted_class, confidence_score)
```
- `predicted_idx.item()` extracts the integer index from the tensor.
- `CLASSES` is the global tuple of class names (defined at the top of the module) in the order expected by the model.
- `confidence.item()` extracts the confidence value as a Python float.

---

## Usage Example
```python
model = build_model()
# Assume model has been trained via train() or loaded from a checkpoint
pred_class, conf = predict(model, Path("data/test/plastic/image_001.jpg"))
print(f"Predicted: {pred_class} (confidence: {conf:.2%})")
```

---

## Compliance Notes
- All heavy imports are inside the function, satisfying the requirement that the module can be imported without deep‑learning frameworks installed.
- The function returns a simple tuple of `(str, float)`, which matches the specification and can be easily used by the grading scripts or a UI.
- The preprocessing exactly mirrors that used in `train()`, ensuring that the model sees data in the same distribution it was trained on.
