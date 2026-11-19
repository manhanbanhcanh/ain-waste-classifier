"""Streamlit application - Project 11 Visual Waste Sorting Assistant (Final).

This application enables users:
- Select model variant (Lite/Standard/Smart)
- Upload arbitrary JPG/PNG images of waste items
- View instant visual previews
- Display the predicted TrashNet class with per-class confidence bars
- Surface educational disposal guidance per category
- Highlight known classification failure modes
- Train models with real-time progress tracking

The AI core is implemented in `starter/student_core.py`.
"""
from __future__ import annotations

import sys
import tempfile
import time
import threading
from pathlib import Path
import html
from typing import Callable, Optional

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import (
    CLASSES,
    MODEL_VARIANTS,
    class_description,
    build_model,
    predict,
    get_model_info,
)  # noqa: E402

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Visual Waste Sorting Assistant",
    page_icon=":material/recycling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Educational disposal guidance per category
DISPOSAL_GUIDANCE = {
    "cardboard": {
        "icon": ":material/package:",
        "title": "Cardboard & paperboard",
        "guidance": [
            "Flatten boxes to save space in recycling bins",
            "Remove any plastic tape, labels, or bubble wrap",
            "Keep dry - wet cardboard contaminates the recycling stream",
            "Pizza boxes: recycle clean tops, compost greasy bottoms",
        ],
        "color": "green",
    },
    "glass": {
        "icon": ":material/local_drink:",
        "title": "Glass bottles & jars",
        "guidance": [
            "Rinse containers - residue contaminates other recyclables",
            "Remove caps/lids (often metal or plastic, recycle separately)",
            "No need to remove paper labels",
            "Do NOT recycle: drinking glasses, ceramics, light bulbs, mirrors, window glass",
        ],
        "color": "blue",
    },
    "metal": {
        "icon": ":material/recycling:",
        "title": "Metal cans & containers",
        "guidance": [
            "Rinse food cans thoroughly",
            "Aluminum cans: crush to save space",
            "Aerosol cans: must be completely empty",
            "Do NOT recycle: scrap metal, electronics, batteries, paint cans",
        ],
        "color": "gray",
    },
    "paper": {
        "icon": ":material/description:",
        "title": "Paper & mixed paper",
        "guidance": [
            "Keep paper dry and clean",
            "Remove plastic windows from envelopes",
            "Staples and paper clips are OK",
            "Do NOT recycle: tissue, paper towels, waxed paper, laminated paper",
        ],
        "color": "orange",
    },
    "plastic": {
        "icon": ":material/local_cafe:",
        "title": "Plastic bottles & containers",
        "guidance": [
            "Check for recycling symbol (1-7) - only recycle accepted types",
            "Rinse and crush bottles, keep caps on",
            "Do NOT recycle: plastic bags, film, Styrofoam, utensils, straws",
            "When in doubt, throw it out - wishcycling contaminates the stream",
        ],
        "color": "blue",
    },
    "trash": {
        "icon": ":material/delete:",
        "title": "Landfill / general waste",
        "guidance": [
            "Items that cannot be recycled in standard municipal programs",
            "Includes: chip bags, candy wrappers, dirty packaging, broken ceramics",
            "Consider: can this be reduced, reused, or composted instead?",
            "Check local guidelines for hazardous waste (batteries, electronics, chemicals)",
        ],
        "color": "gray",
    },
}

# Known failure modes for the model
FAILURE_MODES = {
    "specular_reflections": {
        "title": "Specular reflections on glass/metal",
        "description": "Bright highlights on shiny surfaces can dominate features, causing misclassification as 'trash' or 'plastic'.",
        "mitigation": "Try photographing from a different angle or in diffused lighting.",
        "icon": ":material/wb_incandescent:",
    },
    "deformable_objects": {
        "title": "Crumpled/deformed plastic & paper",
        "description": "Heavy crumpling changes shape and texture significantly, reducing recognition accuracy.",
        "mitigation": "Flatten items before photographing when possible.",
        "icon": ":material/transform:",
    },
    "low_light_blur": {
        "title": "Low-light or blurred captures",
        "description": "Insufficient lighting or camera shake reduces feature clarity, especially for transparent materials (glass).",
        "mitigation": "Use adequate lighting, stabilize camera, or enable flash.",
        "icon": ":material/blur_on:",
    },
    "novel_items": {
        "title": "Novel/unseen item types",
        "description": "Items not represented in TrashNet (e.g., specific brand packaging, composite materials) may be misclassified.",
        "mitigation": "Model confidence will be low - trust the confidence bars, not just the top prediction.",
        "icon": ":material/help_outline:",
    },
    "class_imbalance": {
        "title": "Class imbalance bias",
        "description": "TrashNet has uneven class distributions; the model may bias toward majority classes (paper, plastic).",
        "mitigation": "Check per-class confidence; low confidence on minority classes (glass, metal) is expected.",
        "icon": ":material/balance:",
    },
    "background_clutter": {
        "title": "Complex backgrounds",
        "description": "Cluttered backgrounds can confuse the classifier, especially for small waste items.",
        "mitigation": "Place item on a plain, contrasting background.",
        "icon": ":material/layers:",
    },
}


# Model variant display info
MODEL_DISPLAY = {
    "lite": {
        "label": "Lite",
        "icon": ":material/speed:",
        "description": "Ultra-fast inference (~5K params, ~5ms CPU)",
        "color": "green",
    },
    "standard": {
        "label": "Standard",
        "icon": ":material/balance:",
        "description": "Balanced speed/accuracy (~4.3M params, ~50ms CPU)",
        "color": "blue",
    },
    "smart": {
        "label": "Smart",
        "icon": ":material/psychology:",
        "description": "Maximum accuracy (~15M params, ~150ms CPU)",
        "color": "violet",
    },
}


@st.cache_resource
def load_model(variant: str = "standard"):
    """Load the trained model from disk or build a fresh one for the given variant."""
    model_path = PROJECT_ROOT / "models" / f"best_model_{variant}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Type-safe variant for build_model
    variant_literal: "Literal['lite', 'standard', 'smart']" = variant  # type: ignore[assignment]
    model = build_model(input_shape=(128, 128, 3), num_classes=len(CLASSES), variant=variant_literal)
    model.to(device)

    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            return model, device, True
        except Exception as e:
            st.sidebar.warning(f"Could not load saved {variant} model: {e}. Using untrained model.")

    model.eval()
    return model, device, False


def get_all_confidences(model: torch.nn.Module, image_path: Path, device: torch.device) -> dict[str, float]:
    """Get confidence scores for all classes."""
    image = Image.open(image_path).convert("RGB")
    image = transforms.Resize((128, 128))(image)
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    return {cls: probabilities[i].item() for i, cls in enumerate(CLASSES)}


# =============================================================================
# Training with Progress Tracking
# =============================================================================


class TrainingProgress:
    """Container for training progress updates."""
    def __init__(self):
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.train_loss: float = 0.0
        self.train_acc: float = 0.0
        self.val_loss: float = 0.0
        self.val_acc: float = 0.0
        self.current_lr: float = 0.0
        self.history: dict[str, list[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        self.start_time: float = 0.0
        self.epoch_start_time: float = 0.0
        self.is_training: bool = False
        self.is_stopped: bool = False
        self.error: str | None = None
        self.device: str = "cpu"

    def estimate_remaining(self) -> float:
        """Estimate remaining time in seconds."""
        if self.current_epoch <= 0 or self.start_time <= 0:
            return 0.0
        elapsed = time.time() - self.start_time
        avg_per_epoch = elapsed / self.current_epoch
        remaining_epochs = self.total_epochs - self.current_epoch
        return avg_per_epoch * remaining_epochs

    def get_progress_pct(self) -> float:
        """Get overall progress percentage."""
        if self.total_epochs == 0:
            return 0.0
        return self.current_epoch / self.total_epochs


def train_with_progress(
    model: torch.nn.Module,
    train_dir: Path,
    validation_dir: Path,
    progress_callback: Callable[[TrainingProgress], None],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
) -> dict[str, list[float]]:
    """
    Train model with real-time progress callback.

    Args:
        model: The model to train
        train_dir: Training data directory
        validation_dir: Validation data directory
        progress_callback: Called with TrainingProgress after each epoch
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on (auto-detected if None)

    Returns:
        Training history dictionary
    """
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    progress = TrainingProgress()
    progress.total_epochs = epochs
    progress.device = device.type
    progress.start_time = time.time()
    progress.is_training = True

    # Get input shape from model (WasteNet has input_shape attribute)
    input_shape: tuple[int, int, int] = getattr(model, 'input_shape', (128, 128, 3))
    h, w = input_shape[0], input_shape[1]

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights for loss function (inverse frequency)
    weight_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    try:
        for epoch in range(epochs):
            if progress.is_stopped:
                break

            progress.current_epoch = epoch + 1
            progress.epoch_start_time = time.time()
            progress_callback(progress)

            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                if progress.is_stopped:
                    break

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

            # Validation phase
            model.eval()
            running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    if progress.is_stopped:
                        break
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
            progress.current_lr = current_lr

            # Store history
            progress.train_loss = train_loss
            progress.train_acc = train_acc
            progress.val_loss = validation_loss
            progress.val_acc = validation_acc
            progress.history['train_loss'].append(train_loss)
            progress.history['train_acc'].append(train_acc)
            progress.history['val_loss'].append(validation_loss)
            progress.history['val_acc'].append(validation_acc)
            progress.history['lr'].append(current_lr)

            progress_callback(progress)

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

        progress.is_training = False
        return progress.history

    except Exception as e:
        progress.error = str(e)
        progress.is_training = False
        progress_callback(progress)
        raise


def render_model_selector():
    """Render the model variant selector at the top of the app."""
    # Initialize session state if not set
    if "selected_variant" not in st.session_state:
        st.session_state.selected_variant = "standard"
    # Store previous valid variant to prevent deselection
    if "_prev_variant" not in st.session_state:
        st.session_state._prev_variant = "standard"

    def on_variant_change():
        """Callback to prevent deselection - if None selected, restore previous."""
        new_variant = st.session_state.selected_variant
        if new_variant not in MODEL_VARIANTS:
            # User tried to deselect - restore previous valid variant
            st.session_state.selected_variant = st.session_state._prev_variant
        else:
            # Valid selection - update previous
            st.session_state._prev_variant = new_variant

    # Use segmented control for compact, visible selection
    variant = st.segmented_control(
        "Model variant",
        options=list(MODEL_VARIANTS),
        format_func=lambda v: MODEL_DISPLAY[v]["label"],
        key="selected_variant",
        help="Choose model variant: Lite (fast), Standard (balanced), Smart (accurate)",
        on_change=on_variant_change,
    )

    # Show description of selected variant
    if variant:
        display = MODEL_DISPLAY[variant]
        st.caption(f"{display['icon']} {display['description']}")


def render_training_section(model, device, variant: str):
    """Render the model training section with real-time progress."""
    st.header("Train model", icon=":material/school:")

    # Check if dataset exists
    data_dir = PROJECT_ROOT / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    dataset_ready = train_dir.exists() and val_dir.exists()

    if not dataset_ready:
        st.warning(
            "Dataset not found. Please prepare the dataset first using "
            "`scripts/prepare_dataset.py --confirm` or ensure the following structure exists:\n"
            "- `data/train/` (class subdirectories)\n"
            "- `data/validation/` (class subdirectories)"
        )
        if st.button("View dataset preparation script", icon=":material/description:", width="stretch"):
            st.code("python scripts/prepare_dataset.py --confirm", language="bash")
        return

    # Initialize training state in session
    training_key = f"training_progress_{variant}"
    thread_key = f"training_thread_{variant}"
    history_key = f"training_history_{variant}"

    if training_key not in st.session_state:
        st.session_state[training_key] = TrainingProgress()
    if thread_key not in st.session_state:
        st.session_state[thread_key] = None
    if history_key not in st.session_state:
        st.session_state[history_key] = None

    progress: TrainingProgress = st.session_state[training_key]

    # Training configuration
    with st.container(border=True):
        st.subheader("Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=20, step=1, key=f"epochs_{variant}")
        with col2:
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1, key=f"batch_{variant}")
        with col3:
            lr = st.selectbox("Learning rate", [0.01, 0.001, 0.0001], index=1, format_func=lambda x: f"{x:.4f}", key=f"lr_{variant}")

        col4, col5 = st.columns(2)
        with col4:
            st.caption(f"Device: **{device.type.upper()}**")
        with col5:
            model_size = sum(p.numel() for p in model.parameters())
            st.caption(f"Model params: **{model_size:,}**")

    # Training controls
    st.subheader("Controls")

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        # Start training
        start_disabled = progress.is_training
        if st.button(
            "Start training",
            type="primary",
            icon=":material/play_arrow:",
            disabled=start_disabled,
            width="stretch",
            key=f"start_{variant}",
        ):
            progress.__init__()
            progress.total_epochs = epochs
            progress.start_time = time.time()
            progress.is_training = True

            def training_worker():
                try:
                    history = train_with_progress(
                        model=model,
                        train_dir=train_dir,
                        validation_dir=val_dir,
                        progress_callback=lambda _: None,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                    )
                    st.session_state[history_key] = history
                except Exception as e:
                    progress.error = str(e)
                finally:
                    progress.is_training = False

            thread = threading.Thread(target=training_worker, daemon=True)
            st.session_state[thread_key] = thread
            thread.start()
            st.rerun()

        # Stop training
        stop_disabled = not progress.is_training
        if st.button(
            "Stop",
            icon=":material/stop:",
            disabled=stop_disabled,
            width="stretch",
            key=f"stop_{variant}",
        ):
            progress.is_stopped = True
            st.toast("Training stop requested...", icon=":material/warning:")

        # Save model
        save_disabled = st.session_state[history_key] is None
        model_path = PROJECT_ROOT / "models" / f"best_model_{variant}.pt"
        if st.button(
            "Save model",
            icon=":material/save:",
            disabled=save_disabled,
            width="stretch",
            key=f"save_{variant}",
        ):
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            st.success(f"Model saved to {model_path}")
            st.info(f"Note: The {variant} model is cached. Restart the Streamlit app to load the new weights.")
            st.toast("Model saved! Restart app to use it.", icon=":material/check_circle:")
            time.sleep(1)
            st.rerun()

    # Progress display
    if progress.is_training or progress.current_epoch > 0:
        st.subheader("Progress")

        # Overall progress
        progress_pct = progress.get_progress_pct()
        st.progress(progress_pct, text=f"Epoch {progress.current_epoch}/{progress.total_epochs} ({progress_pct*100:.0f}%)")

        # Metrics row
        if progress.current_epoch > 0:
            metric_cols = st.columns(6)
            with metric_cols[0]:
                st.metric("Train loss", f"{progress.train_loss:.4f}")
            with metric_cols[1]:
                st.metric("Train acc", f"{progress.train_acc:.2f}%")
            with metric_cols[2]:
                st.metric("Val loss", f"{progress.val_loss:.4f}")
            with metric_cols[3]:
                st.metric("Val acc", f"{progress.val_acc:.2f}%")
            with metric_cols[4]:
                st.metric("LR", f"{progress.current_lr:.2e}" if progress.current_lr > 0 else "N/A")
            with metric_cols[5]:
                if progress.is_training:
                    eta = progress.estimate_remaining()
                    st.metric("ETA", f"{eta/60:.1f} min" if eta > 60 else f"{eta:.0f} sec")
                else:
                    st.metric("Status", "Complete" if not progress.error else "Error")

        # Live charts
        if progress.history['train_loss']:
            st.markdown("**Training curves**")
            chart_cols = st.columns(3)

            with chart_cols[0]:
                loss_data = {
                    "Epoch": list(range(1, len(progress.history['train_loss']) + 1)),
                    "Train loss": progress.history['train_loss'],
                    "Val loss": progress.history['val_loss'],
                }
                st.line_chart(loss_data, x="Epoch", y=["Train loss", "Val loss"], height=300)

            with chart_cols[1]:
                acc_data = {
                    "Epoch": list(range(1, len(progress.history['train_acc']) + 1)),
                    "Train acc": progress.history['train_acc'],
                    "Val acc": progress.history['val_acc'],
                }
                st.line_chart(acc_data, x="Epoch", y=["Train acc", "Val acc"], height=300)

            with chart_cols[2]:
                if progress.history['lr']:
                    lr_data = {
                        "Epoch": list(range(1, len(progress.history['lr']) + 1)),
                        "Learning Rate": progress.history['lr'],
                    }
                    st.line_chart(lr_data, x="Epoch", y="Learning Rate", height=300)

        # Epoch detail
        if progress.current_epoch > 0:
            with st.expander(f"Epoch {progress.current_epoch} details", expanded=True):
                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.write(f"**Train loss:** {progress.train_loss:.4f}")
                with detail_cols[1]:
                    st.write(f"**Train acc:** {progress.train_acc:.2f}%")
                with detail_cols[2]:
                    st.write(f"**Val loss:** {progress.val_loss:.4f}")
                with detail_cols[3]:
                    st.write(f"**Val acc:** {progress.val_acc:.2f}%")

                if progress.is_training:
                    epoch_elapsed = time.time() - progress.epoch_start_time
                    st.caption(f"Epoch time: {epoch_elapsed:.1f}s | Total elapsed: {time.time() - progress.start_time:.1f}s")

        # Error display
        if progress.error:
            st.error(f"Training failed: {progress.error}")

        # Auto-rerun while training
        if progress.is_training:
            time.sleep(1)
            st.rerun()

    # Show final results if training completed
    elif st.session_state[history_key] is not None and not progress.is_training:
        st.success("Training completed!")
        history = st.session_state[history_key]

        # Final metrics
        final_cols = st.columns(4)
        with final_cols[0]:
            st.metric("Final train loss", f"{history['train_loss'][-1]:.4f}")
        with final_cols[1]:
            st.metric("Final train acc", f"{history['train_acc'][-1]:.2f}%")
        with final_cols[2]:
            st.metric("Final val loss", f"{history['val_loss'][-1]:.4f}")
        with final_cols[3]:
            st.metric("Final val acc", f"{history['val_acc'][-1]:.2f}%")

        # Final charts
        st.markdown("**Final training curves**")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            loss_data = {
                "Epoch": list(range(1, len(history['train_loss']) + 1)),
                "Train loss": history['train_loss'],
                "Val loss": history['val_loss'],
            }
            st.line_chart(loss_data, x="Epoch", y=["Train loss", "Val loss"], height=300)
        with chart_cols[1]:
            acc_data = {
                "Epoch": list(range(1, len(history['train_acc']) + 1)),
                "Train acc": history['train_acc'],
                "Val acc": history['val_acc'],
            }
            st.line_chart(acc_data, x="Epoch", y=["Train acc", "Val acc"], height=300)


def render_confidence_bars(confidences: dict[str, float], predicted_class: str):
    """Display per-class confidence bars using native st.progress."""
    st.subheader("Per-class confidence scores")

    # Sort by confidence descending
    sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    for cls, conf in sorted_confidences:
        is_predicted = cls == predicted_class
        guidance = DISPOSAL_GUIDANCE.get(cls, {})
        icon = guidance.get("icon", ":material/package:")

        col1, col2, col3 = st.columns([2, 6, 1], vertical_alignment="center")

        with col1:
            if is_predicted:
                st.markdown(f"**{icon} {cls}** :material/star:")
            else:
                st.markdown(f"{icon} {cls}")

        with col2:
            st.progress(conf, text=f"{conf*100:.1f}%")

        with col3:
            if is_predicted:
                st.badge("Predicted", color="green")

    st.caption("Bars show model confidence per class. Higher = more certain.")


def render_disposal_guidance(class_name: str):
    """Display educational disposal guidance for the predicted class."""
    guidance = DISPOSAL_GUIDANCE.get(class_name, {})

    with st.container(border=True):
        st.markdown(f"### {guidance.get('icon', ':material/package:')} {guidance.get('title', class_name)} - Disposal guidance")

        for i, tip in enumerate(guidance.get("guidance", []), 1):
            st.markdown(f"{i}. {tip}")


def render_failure_modes():
    """Display known classification failure modes in expanders."""
    st.subheader("Known classification limitations", icon=":material/warning:")
    st.caption(
        "This model was trained on the TrashNet dataset (2,527 images). "
        "When confidence is low, consider these factors:"
    )

    for _, failure in FAILURE_MODES.items():
        with st.expander(f"{failure['icon']} {failure['title']}", expanded=False):
            st.markdown(f"**Problem:** {failure['description']}")
            st.markdown(f"**Mitigation:** {failure['mitigation']}")


def render_sidebar(model_loaded: bool, device: torch.device, variant: str):
    """Render the sidebar with class info, model status, and test samples."""
    with st.sidebar:
        st.header("TrashNet classes", icon=":material/category:")

        for cls in CLASSES:
            guidance = DISPOSAL_GUIDANCE.get(cls, {})
            icon = guidance.get("icon", ":material/package:")
            desc = class_description(cls)
            st.markdown(f"**{icon} {cls}** - {desc}")

        st.divider()

        # Current variant info
        display = MODEL_DISPLAY[variant]
        st.markdown(f"### {display['icon']} Current model: {display['label']}")
        info = get_model_info(variant)
        st.caption(f"**Params:** {info['params']}")
        st.caption(f"**Inference (CPU):** {info['inference_time_cpu']}")
        st.caption(f"**Target accuracy:** {info['target_accuracy']}")

        st.divider()

        # Model status
        st.subheader("Model status", icon=":material/memory:")
        if model_loaded:
            st.badge("Trained model loaded", icon=":material/check_circle:", color="green")
        else:
            st.badge("Untrained model (random weights)", icon=":material/warning:", color="orange")
            st.caption("Train a model using `tests/train.py` or `scripts/prepare_dataset.py --confirm` first.")

        st.caption(f"Device: {device.type.upper()}")
        st.caption(f"Classes: {len(CLASSES)}")

        st.divider()

        # Quick test images - use pills with on_change for instant loading
        st.subheader("Quick test samples", icon=":material/science:")
        sample_dir = PROJECT_ROOT / "data" / "sample"

        if sample_dir.exists():
            available_classes = []
            for cls in CLASSES:
                cls_dir = sample_dir / cls
                if cls_dir.exists():
                    images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
                    if images:
                        available_classes.append(cls)

            if available_classes:
                # Initialize session state for quick test selection
                if "quick_test_select" not in st.session_state:
                    st.session_state.quick_test_select = available_classes[0]
                if "_prev_quick_test" not in st.session_state:
                    st.session_state._prev_quick_test = available_classes[0]

                def on_quick_test_change():
                    """Callback to prevent deselection and load test sample."""
                    new_selection = st.session_state.quick_test_select
                    if new_selection not in available_classes:
                        # User tried to deselect - restore previous valid selection
                        st.session_state.quick_test_select = st.session_state._prev_quick_test
                    else:
                        # Valid selection - update previous and load test sample
                        st.session_state._prev_quick_test = new_selection
                        cls_dir = sample_dir / new_selection
                        # Validate path safety
                        try:
                            cls_dir_resolved = cls_dir.resolve()
                            sample_dir_resolved = sample_dir.resolve()
                            if not str(cls_dir_resolved).startswith(str(sample_dir_resolved)):
                                return
                        except Exception:
                            return
                        if cls_dir.exists():
                            images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
                            if images:
                                st.session_state["test_image"] = str(images[0])
                                st.session_state["test_class"] = new_selection
                                st.rerun()

                # Use pills with on_change callback for instant response (no double-click needed)
                st.pills(
                    "Test sample",
                    options=available_classes,
                    format_func=lambda c: f"{DISPOSAL_GUIDANCE.get(c, {}).get('icon', ':material/package:')} {c}",
                    selection_mode="single",
                    key="quick_test_select",
                    label_visibility="collapsed",
                    on_change=on_quick_test_change,
                )
        else:
            st.caption("No sample images found. Run dataset preparation to add samples.")

        st.divider()
        st.caption("Project 11 - Visual Waste Sorting Assistant")
        st.caption("Group 9 - 66FIT3AIN.TT02")


def render_upload_section():
    """Render the image upload and classification section."""
    st.subheader("Upload waste image", icon=":material/upload:")

    # Check if there's a test image from sidebar
    test_image_path = st.session_state.get("test_image")
    test_class = st.session_state.get("test_class")

    if test_image_path:
        st.info(f"Using test sample: **{test_class}** - {Path(test_image_path).name}")
        if st.button("Clear test image", icon=":material/clear:", width="stretch"):
            del st.session_state["test_image"]
            del st.session_state["test_class"]
            st.rerun()

    # Drag-and-drop upload zone (file_uploader supports drag-and-drop natively)
    uploaded = st.file_uploader(
        "Drag and drop a waste photo here, or click to browse",
        type=["png", "jpg", "jpeg"],
        help="Supports PNG, JPG, JPEG. Drag and drop or click to select.",
        label_visibility="collapsed",
    )

    # Visual drag-and-drop hint
    if uploaded is None and not test_image_path:
        st.caption(":material/cloud_upload: Drag & drop an image file, or click to browse")

    # Determine which image to use
    image_path = None
    image_source = ""

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.getbuffer())
            image_path = Path(tmp.name)
        image_source = f"Uploaded: {html.escape(uploaded.name)}"
    elif test_image_path:
        image_path = Path(test_image_path)
        safe_test_class = test_class or "Unknown"
        image_source = f"Test sample: {html.escape(safe_test_class)}"

    return image_path, image_source, uploaded


def render_prediction_section(image_path: Path | None, image_source: str, model, device):
    """Render the classification results section."""
    st.subheader("Classification result", icon=":material/psychology:")

    if image_path and image_path.exists():
        # Display image preview
        st.image(str(image_path), caption=image_source)

        # Classify button
        if st.button("Classify waste", type="primary", icon=":material/search:", width="stretch"):
            with st.spinner("Analyzing image..."):
                try:
                    predicted_class, confidence = predict(model, image_path)
                    all_confidences = get_all_confidences(model, image_path, device)

                    st.session_state["prediction"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "all_confidences": all_confidences,
                        "image_source": image_source,
                    }
                    st.rerun()

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                finally:
                    # Clean up temp file if it was an upload
                    if "uploaded" in st.session_state and st.session_state.get("uploaded"):
                        image_path.unlink(missing_ok=True)

        # Display prediction results if available
        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            predicted_class = pred["class"]
            confidence = pred["confidence"]
            all_confidences = pred["all_confidences"]
            image_source = pred["image_source"]

            confidence_pct = confidence * 100

            # Confidence level styling
            if confidence_pct >= 80:
                conf_color = "green"
                conf_label = "High confidence"
                conf_icon = ":material/check_circle:"
            elif confidence_pct >= 50:
                conf_color = "orange"
                conf_label = "Medium confidence"
                conf_icon = ":material/help:"
            else:
                conf_color = "red"
                conf_label = "Low confidence - verify manually"
                conf_icon = ":material/error:"

            # Main prediction card
            with st.container(border=True):
                col1, col2 = st.columns([3, 1], vertical_alignment="center")

                with col1:
                    st.markdown(f"#### {conf_icon} Predicted: **{predicted_class.upper()}**")
                    st.markdown(f"Confidence: **{confidence_pct:.1f}%** - _{conf_label}_")
                    st.caption(f"Source: {image_source}")

                with col2:
                    st.badge(conf_label, color=conf_color)

            # Confidence bars
            render_confidence_bars(all_confidences, predicted_class)

            # Disposal guidance
            render_disposal_guidance(predicted_class)

            # Low confidence warning
            if confidence_pct < 50:
                st.warning(
                    "**Low confidence warning:** The model is uncertain about this prediction. "
                    "This may be due to image quality (blur, low light, glare), unusual angle, "
                    "heavy deformation, item type not well-represented in training data, or "
                    "complex background interference. Please verify manually and check the "
                    "known limitations below."
                )
    else:
        st.info("Upload an image or select a test sample from the sidebar to begin classification.")


def render_footer():
    """Render the app footer."""
    st.divider()
    st.caption("Visual Waste Sorting Assistant - Project 11 Final Milestone")
    st.caption("Group 9: Duc Manh, Trang Quynh, Bao Chi | Class: 66FIT3AIN.TT02")
    st.caption("Model: Custom CNN (WasteNet) trained on TrashNet dataset | Framework: PyTorch")


def main():
    # Render model variant selector at top
    render_model_selector()

    st.divider()

    # Load model for selected variant - render_model_selector ensures valid variant
    variant = st.session_state.get("selected_variant", "standard")

    with st.skeleton():
        model, device, model_loaded = load_model(variant)

    # Render sidebar
    render_sidebar(model_loaded, device, variant)

    # Main content area - tabs for different functions
    tab_classify, tab_train = st.tabs([
        "Classify waste",
        "Train model",
    ])

    with tab_classify:
        # Main content area - two columns
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            image_path, image_source, uploaded = render_upload_section()
            if uploaded:
                st.session_state["uploaded"] = True

        with col2:
            render_prediction_section(image_path, image_source, model, device)

        # Full-width sections
        render_failure_modes()

    with tab_train:
        render_training_section(model, device, variant)

    render_footer()


if __name__ == "__main__":
    main()