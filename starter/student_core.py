"""
# Project 11 AI core — visual waste sorting.

1. Requirement:

Fill in ``build_model``, ``train``, and ``predict`` yourself. Do **not**:

- paste in a finished/pretrained CNN architecture from a tutorial and call
  it done;
- load third-party solution weights (``.h5``, ``.pkl``, ``.pt``, ``.ckpt``);
- call a hosted vision/LLM API in place of your own CNN.

The required experiment (see ``README.md``) is to change at least one
of: architecture depth, filter count, dropout, augmentation, or learning
rate, and to compare the effect on your training/validation curves.

Heavy imports (TensorFlow/Keras, PyTorch, etc.) belong *inside* these
functions, not at module import time, so that ``data/sample/`` smoke tests
and ``tests/test_project_11_sanity.py`` keep working even before you (or a
grader) install a deep-learning framework.

2. Implementation:

Core Implementation (build_model)

  - Custom CNN Architecture: Built from scratch with 3 convolutional blocks (16→32→64 filters) followed by dense layers
  - Key Features:
    - Conv2d → ReLU → MaxPool2d blocks for feature extraction
    - Dropout (0.5) in dense layers to prevent overfitting
    - Fully connected classifier ending in 6 outputs (TrashNet classes)
  - Compliance: No pretrained weights, all heavy imports inside functions

  Supporting Functions

  - train(): Complete training loop with:
    - CrossEntropyLoss and Adam optimizer
    - Training/validation loss/accuracy tracking per epoch
    - Device-agnostic execution (GPU/CPU fallback)
    - PyTorch DataLoaders for batch processing
  - predict(): Single-image inference returning (class_name, confidence):
    - Matches training transforms (resize, tensor conversion, normalization)
    - Softmax conversion to probabilities
    - Returns required tuple format

  Usage & Experimentation

  The architecture is ready for the required experiments:
  - Depth: Modify number of conv layers in WasteNet
  - Filter counts: Adjust 16→32→64 progression
  - Dropout: Change nn.Dropout(0.5) rate
  - Learning rate: Pass lr kwarg to train()

  All imports remain inside functions as required, ensuring sample tests work without deep learning frameworks installed. The original student_core.py remains
  untouched per your request.

**Important: Every `#type: ignore` is removable, only needed in development**
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# TrashNet's six classes, in the order the dataset defines them. Do not
# reorder or rename these — labels, the sample data, and the sanity tests
# all depend on this exact tuple.
CLASSES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

CLASS_DESCRIPTIONS: dict[str, str] = {
    "cardboard": "Flattened boxes and packaging paperboard.",
    "glass": "Bottles and jars (clear or colored glass).",
    "metal": "Cans and other small metal containers.",
    "paper": "Loose paper, flyers, and non-cardboard paper waste.",
    "plastic": "Bottles, containers, and rigid plastic packaging.",
    "trash": "Items that do not belong to the other five recyclable classes.",
}


def class_description(class_name: str) -> str:
    """Return a short, student-editable description of a TrashNet class.

    This is infrastructure text for the Final UI, not the graded core.
    """
    return CLASS_DESCRIPTIONS.get(class_name, "Unknown class.")


def build_model(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = len(CLASSES),
) -> Any:
    """Build and return an *untrained* CNN for ``num_classes`` waste categories.

    Implement your own architecture (a stack of Conv2D/pooling layers
    followed by dense layers is enough). Do not load a pretrained model or
    third-party solution weights.

    Args:
        input_shape: (height, width, channels) of the images you resize to.
        num_classes: number of output classes (6 for the TrashNet labels).

    Returns:
        An untrained model object (e.g. a compiled ``tf.keras.Model``).
    """
    import torch  
    import torch.nn as nn  
    import torch.nn.functional as F  

    class WasteNet(nn.Module):
        def __init__(
            self,
            input_shape: tuple[int, int, int],
            num_classes: int
        ) -> None:
            super().__init__()
            self.input_shape: tuple[int, int, int] = input_shape
            self.num_classes: int = num_classes

            h: int = input_shape[0]
            w: int = input_shape[1]
            c: int = input_shape[2]  # define height, weight, color channel (3 for RGB)

            self.conv1 = nn.Conv2d(c, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

            self.pool = nn.MaxPool2d(2, 2)

            # connect classifier
            self.fc1 = nn.Linear(64 * (h // 8) * (w // 8), 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * (self.input_shape[0] // 8) * (self.input_shape[1] // 8))
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return WasteNet(input_shape, num_classes)


def train(
    model: Any,
    train_dir: Path,
    validation_dir: Path,
    *,
    epochs: int = 10,
    **kwargs: Any,
) -> Any:
    """Train ``model`` on images under ``train_dir``, validating on ``validation_dir``.

    Both directories are expected to contain one sub-folder per class (see
    ``data/README.md``): either ``data/sample/`` for a tiny smoke test, or
    ``data/train/`` and ``data/validation/`` after running
    ``scripts/prepare_dataset.py --confirm``.

    Return whatever training-history object your framework produces (for
    example a Keras ``History``) so the required experiment can plot
    training/validation curves.
    """
    import torch  
    import torch.nn as nn  
    import torch.optim as optim  
    from torchvision import datasets, transforms  
    from torch.utils.data import DataLoader  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
            transforms.Resize((model.input_shape[0], model.input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1,1]
        ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=transform)

    batch_size = kwargs.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.001))

    # training history
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # training loop
    for epoch in range(epochs):
        model.train()
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

        # validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
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

        # store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(validation_loss)
        history['val_acc'].append(validation_acc)

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%')

    return history


def predict(model: Any, image_path: Path) -> tuple[str, float]:
    """Return ``(predicted_class, confidence)`` for the image at ``image_path``.

    ``predicted_class`` must be one of ``CLASSES``. ``confidence`` is a
    float in ``[0, 1]``.
    """

    import torch  
    import torch.nn.functional as F  
    from torchvision import transforms  
    from PIL import Image  

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # define transform 
    # !important: must match the one used in training
    transform = transforms.Compose([
        transforms.Resize((model.input_shape[0], model.input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)  
    image_tensor = transformed_image.unsqueeze(0)    # Add batch dimension
    image_tensor = image_tensor.to(device)  

    # make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class: str = CLASSES[predicted_idx.item()]  
    confidence_score: float = confidence.item()  

    return (predicted_class, confidence_score)  