"""Project 11 AI core — visual waste sorting.

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
    *,
    depth: int = 4,
    base_filters: int = 32,
    dropout: float = 0.5,
    lightweight: bool = False,
) -> Any:
    """Build and return an *untrained* CNN for ``num_classes`` waste categories.

    Implement your own architecture (a stack of Conv2D/pooling layers
    followed by dense layers is enough). Do not load a pretrained model or
    third-party solution weights.

    Args:
        input_shape: (height, width, channels) of the images you resize to.
        num_classes: number of output classes (6 for the TrashNet labels).
        depth: Number of convolutional blocks (3, 4, or 5).
        base_filters: Number of filters in the first conv layer (doubles each block).
        dropout: Dropout probability for the classifier head.
        lightweight: If True, uses Global Average Pooling and smaller FC layer
                     for a much lighter model (~10x fewer params).

    Returns:
        An untrained model object (e.g. a compiled ``tf.keras.Model``).
    """
    import torch
    import torch.nn as nn

    class WasteCNN(nn.Module):
        """Custom CNN for TrashNet waste classification."""

        def __init__(
            self,
            input_shape: tuple[int, int, int],
            num_classes: int,
            depth: int = 4,
            base_filters: int = 32,
            dropout: float = 0.5,
            lightweight: bool = False,
        ):
            super().__init__()
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.depth = depth
            self.base_filters = base_filters
            self.dropout = dropout
            self.lightweight = lightweight

            # Build convolutional feature extractor
            layers = []
            in_channels = input_shape[2]
            current_filters = base_filters

            for i in range(depth):
                layers.extend([
                    nn.Conv2d(in_channels, current_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ])
                in_channels = current_filters
                current_filters *= 2

            self.features = nn.Sequential(*layers)

            if lightweight:
                # Use Global Average Pooling + small FC layer
                # Final feature map channels = base_filters * 2^(depth-1)
                final_channels = base_filters * (2 ** (depth - 1))
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(final_channels, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(64, num_classes),
                )
            else:
                # Calculate flattened size dynamically
                with torch.no_grad():
                    dummy = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
                    dummy = self.features(dummy)
                    flattened_size = dummy.view(1, -1).size(1)

                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=dropout),
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(512, num_classes),
                )

        def forward(self, x):
            x = self.features(x)
            if self.lightweight:
                x = self.pool(x)
                x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = WasteCNN(input_shape, num_classes, depth, base_filters, dropout, lightweight)
    return model


def train(
    model: Any,
    train_dir: Path,
    validation_dir: Path,
    *,
    epochs: int = 10,
    lr: float = 0.001,
    augmentation: str = "basic",
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

    Args:
        model: The model to train.
        train_dir: Path to training data directory.
        validation_dir: Path to validation data directory.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        augmentation: Data augmentation strategy ("none", "basic", "strong").
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    # Check if we have CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Data transforms based on augmentation strategy
    if augmentation == "none":
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation == "basic":
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unknown augmentation strategy: {augmentation}")

    # For validation: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class WasteDataset(Dataset):
        def __init__(self, root_dir: Path, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.samples = []
            self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

            for class_name in CLASSES:
                class_dir = root_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.png"):
                        self.samples.append((img_path, self.class_to_idx[class_name]))
                    for img_path in class_dir.glob("*.jpg"):
                        self.samples.append((img_path, self.class_to_idx[class_name]))
                    for img_path in class_dir.glob("*.jpeg"):
                        self.samples.append((img_path, self.class_to_idx[class_name]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = WasteDataset(train_dir, transform=train_transform)
    val_dataset = WasteDataset(validation_dir, transform=val_transform)

    # Handle case where sample data might be very small
    batch_size = min(32, len(train_dataset)) if len(train_dataset) > 0 else 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / train_total if train_total > 0 else 0
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_total if val_total > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Create a simple history object that mimics Keras History
    class History:
        def __init__(self, history_dict):
            self.history = history_dict

    return History(history)


def predict(model: Any, image_path: Path) -> tuple[str, float]:
    """Return ``(predicted_class, confidence)`` for the image at ``image_path``.

    ``predicted_class`` must be one of ``CLASSES``. ``confidence`` is a
    float in ``[0, 1]``.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Transform for inference (same as validation)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Handle both file path and uploaded file (BytesIO)
    if hasattr(image_path, 'read'):
        # It's a file-like object (Streamlit uploaded file)
        image = Image.open(image_path).convert("RGB")
    else:
        # It's a Path or string
        image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted_idx.item()]
    confidence_value = confidence.item()

    return predicted_class, confidence_value