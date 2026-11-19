"""
# Project 11 AI core - visual waste sorting.

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

  - Three Model Variants (DRAMATICALLY DIFFERENT):
    - Lite: 1 Conv block (16→32 filters), GlobalAvgPool, tiny FC - ~5K params - Ultra-fast edge
    - Standard: 3 Conv blocks (16→32→64), 2 FC layers - ~4.3M params - Balanced
    - Smart: 5-Stage Deep Residual (64→64→128→256→512), 10 SE Residual blocks, GlobalAvgPool - ~15M params - Best accuracy

  - Key Features by Variant:
    - Lite: Minimal, single conv block, no residuals, no attention
    - Standard: Classic CNN, progressive filters, dropout regularization
    - Smart: Deep residuals, Squeeze-Excitation attention, batch norm throughout, wider filters

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
  - Filter counts: Adjust filter progression
  - Dropout: Change nn.Dropout rate
  - Learning rate: Pass lr kwarg to train()

All imports remain inside functions as required, ensuring sample tests work without deep learning frameworks installed.

**Important: Every `#type: ignore` is removable, as it is only needed in development**
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

# TrashNet's six classes, in the order the dataset defines them. Do not
# reorder or rename these - labels, the sample data, and the sanity tests
# all depend on this exact tuple.
CLASSES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

# Model variants
MODEL_VARIANTS: tuple[str, ...] = ("lite", "standard", "smart")

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
    variant: Literal["lite", "standard", "smart"] = "standard",
) -> Any:
    """Build and return a WasteNet model of the specified variant.

    Args:
        input_shape: (height, width, channels) of the images you resize to.
        num_classes: Number of output classes (6 for TrashNet).
        variant: Model variant - "lite", "standard", or "smart".

    Returns:
        An untrained model object (torch.nn.Module).
    """
    import torch
    import torch.nn as nn

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation block for channel attention (Smart variant)."""
        def __init__(self, channels: int, reduction: int = 16) -> None:
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    class ResidualBlock(nn.Module):
        """Residual block with optional SE attention (Smart variant)."""
        def __init__(self, channels: int, use_se: bool = False) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
            self.se = SEBlock(channels) if use_se else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += identity
            return self.relu(out)

    class ConvBlock(nn.Module):
        """Convolutional block: Conv + BN + ReLU + Pool (Standard/Smart)."""
        def __init__(self, in_channels: int, out_channels: int, use_pool: bool = True) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(2, 2) if use_pool else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.bn(self.conv(x)))
            x = self.pool(x)
            return x

    class WasteNet(nn.Module):
        def __init__(
            self,
            input_shape: tuple[int, int, int],
            num_classes: int,
            variant: str = "standard",
        ) -> None:
            super().__init__()
            self.input_shape: tuple[int, int, int] = input_shape
            self.num_classes: int = num_classes
            self.variant: str = variant

            h: int = input_shape[0]
            w: int = input_shape[1]
            c: int = input_shape[2]  # height, width, color channel (3 for RGB)

            if variant == "lite":
                # ============================================================
                # LITE: Ultra-lightweight - Single conv block, GlobalAvgPool
                # ~0.1M params, ~5ms CPU inference
                # ============================================================
                self.features = nn.Sequential(
                    nn.Conv2d(c, 16, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),           # 64x64
                    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),           # 32x32
                )
                # Global Average Pooling instead of large FC
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(32, num_classes)
                )

            elif variant == "standard":
                # ============================================================
                # STANDARD: Classic 3-block CNN with FC classifier
                # ~2.1M params, ~50ms CPU inference
                # ============================================================
                self.features = nn.Sequential(
                    # Block 1: 128 -> 64
                    ConvBlock(c, 16, use_pool=True),
                    # Block 2: 64 -> 32
                    ConvBlock(16, 32, use_pool=True),
                    # Block 3: 32 -> 16
                    ConvBlock(32, 64, use_pool=True),
                )
                # Flatten + FC
                fc_input_size = 64 * (h // 8) * (w // 8)  # 64 * 16 * 16 = 16384
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(fc_input_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )

            elif variant == "smart":
                # ============================================================
                # SMART: Deep residual CNN with SE attention - WIDE & DEEP
                # ~8M params, ~150ms CPU inference
                # ============================================================
                # Stem: 128 -> 64
                self.stem = nn.Sequential(
                    nn.Conv2d(c, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )

                # Stage 1: 64x64, 64 channels
                self.stage1 = nn.Sequential(
                    ConvBlock(64, 64, use_pool=False),
                    ResidualBlock(64, use_se=True),
                    ResidualBlock(64, use_se=True),
                    ResidualBlock(64, use_se=True),
                )
                self.pool1 = nn.MaxPool2d(2, 2)  # 32x32

                # Stage 2: 32x32, 128 channels
                self.stage2 = nn.Sequential(
                    ConvBlock(64, 128, use_pool=False),
                    ResidualBlock(128, use_se=True),
                    ResidualBlock(128, use_se=True),
                    ResidualBlock(128, use_se=True),
                )
                self.pool2 = nn.MaxPool2d(2, 2)  # 16x16

                # Stage 3: 16x16, 256 channels
                self.stage3 = nn.Sequential(
                    ConvBlock(128, 256, use_pool=False),
                    ResidualBlock(256, use_se=True),
                    ResidualBlock(256, use_se=True),
                )
                self.pool3 = nn.MaxPool2d(2, 2)  # 8x8

                # Stage 4: 8x8, 512 channels
                self.stage4 = nn.Sequential(
                    ConvBlock(256, 512, use_pool=False),
                    ResidualBlock(512, use_se=True),
                    ResidualBlock(512, use_se=True),
                )
                self.pool4 = nn.MaxPool2d(2, 2)  # 4x4

                # Global Average Pooling + Classifier
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(512, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, num_classes)
                )

            else:
                raise ValueError(f"Unknown variant: {variant}. Choose from: {MODEL_VARIANTS}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.variant == "lite":
                x = self.features(x)      # 32x32x32
                x = self.gap(x)           # 32x1x1
                x = x.view(x.size(0), -1) # flatten to (B, 32)
                x = self.classifier(x)

            elif self.variant == "standard":
                x = self.features(x)      # 16x16x64
                x = self.classifier(x)

            elif self.variant == "smart":
                x = self.stem(x)          # 64x64x64
                x = self.stage1(x)        # 64x64x64
                x = self.pool1(x)         # 32x32x64
                x = self.stage2(x)        # 32x32x128
                x = self.pool2(x)         # 16x16x128
                x = self.stage3(x)        # 16x16x256
                x = self.pool3(x)         # 8x8x256
                x = self.stage4(x)        # 8x8x512
                x = self.pool4(x)         # 4x4x512
                x = self.gap(x)           # 512x1x1
                x = x.view(x.size(0), -1) # flatten to (B, 512)
                x = self.classifier(x)

            return x

    return WasteNet(input_shape, num_classes, variant)


def train(
    model: Any,
    train_dir: Path,
    validation_dir: Path,
    *,
    epochs: int = 10,
    progress: bool = False,
    **kwargs: Any,
) -> Any:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    # Optional tqdm import for progress bar
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
        HAS_TQDM: bool = True
    except ImportError:
        tqdm = None  # type: ignore[assignment]
        HAS_TQDM = False
        if progress:
            print("Note: tqdm not installed, progress bar disabled. Install with: pip install tqdm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((model.input_shape[0], model.input_shape[1])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1,1]
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((model.input_shape[0], model.input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=val_transform)

    # Handle class imbalance with WeightedRandomSampler
    targets = [label for _, label in train_dataset.samples]
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    batch_size = kwargs.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights for loss function (inverse frequency)
    weight_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=kwargs.get('lr', 0.001), weight_decay=1e-4)

    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # training history
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience = kwargs.get('patience', 10)
    patience_counter = 0
    best_model_state = None

    # training loop
    epoch_iterator = range(epochs)
    if progress and HAS_TQDM:
        epoch_iterator = tqdm(epoch_iterator, desc="Epochs", unit="epoch")  # type: ignore[operator]

    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training batch loop with optional progress bar
        train_loader_iter = train_loader
        if progress and HAS_TQDM:
            train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")  # type: ignore[operator]

        for inputs, labels in train_loader_iter:
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

        # Validation batch loop with optional progress bar
        val_loader_iter = validation_loader
        if progress and HAS_TQDM:
            val_loader_iter = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, unit="batch")  # type: ignore[operator]

        with torch.no_grad():
            for inputs, labels in val_loader_iter:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        validation_loss = running_loss / len(validation_loader)
        validation_acc = 100 * correct_val / total_val

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(validation_loss)
        history['val_acc'].append(validation_acc)
        history['lr'].append(current_lr)

        # Print epoch summary (tqdm handles the progress bar, so we print summary after)
        if not (progress and HAS_TQDM):
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%, '
                  f'LR: {current_lr:.6f}')
        else:
            # With tqdm, print a clean summary line
            tqdm.write(f'Epoch {epoch+1}/{epochs}: '  # type: ignore[union-attr]
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%, '
                  f'LR: {current_lr:.6f}')

        # Early stopping check
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs (best val loss: {best_val_loss:.4f})')
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def predict(model: Any, image_path: Path) -> tuple[str, float]:

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
    # ToTensor() returns a torch.Tensor - cast for type checker
    image_tensor = torch.as_tensor(transformed_image).unsqueeze(0)    # Add batch dimension
    image_tensor = image_tensor.to(device)

    # make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class: str = CLASSES[int(predicted_idx.item())]
    confidence_score: float = float(confidence.item())

    return (predicted_class, confidence_score)


def get_model_info(variant: str) -> dict[str, Any]:
    """Get information about a model variant without building it."""
    info = {
        "lite": {
            "name": "WasteNet-Lite",
            "description": "Ultra-lightweight single-block CNN with GlobalAvgPool for edge deployment",
            "conv_blocks": 1,
            "filters": "16→32",
            "params": "~5K",
            "inference_time_cpu": "~5ms",
            "target_accuracy": ">70%",
            "features": "Minimal, no residuals, no attention, GAP classifier",
        },
        "standard": {
            "name": "WasteNet-Standard",
            "description": "Classic 3-block CNN with dual FC layers for balanced deployment",
            "conv_blocks": 3,
            "filters": "16→32→64",
            "params": "~4.3M",
            "inference_time_cpu": "~50ms",
            "target_accuracy": ">85%",
            "features": "Progressive filters, dropout regularization, FC classifier",
        },
        "smart": {
            "name": "WasteNet-Smart",
            "description": "Deep 5-stage WIDE residual CNN with SE attention for maximum accuracy",
            "conv_blocks": 5,
            "filters": "64→64→128→256→512 (+ residuals + SE)",
            "params": "~15M",
            "inference_time_cpu": "~150ms",
            "target_accuracy": ">95%",
            "features": "Deep wide residuals, Squeeze-Excitation attention, BatchNorm throughout, GAP classifier, 10 residual blocks (3+3+2+2)",
        },
    }
    return info.get(variant, info["standard"])