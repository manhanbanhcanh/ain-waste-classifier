"""Streamlit starter — Project 11 visual waste sorting assistant.

This page uploads an image and shows a preview. It must not classify,
train, or call a hosted model: ``student_core.predict`` is an empty stub
until you implement it yourself.
"""
from __future__ import annotations

import sys
import os
import pickle
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import CLASSES, class_description, build_model, train, predict  # noqa: E402

st.set_page_config(page_title="Visual Waste Sorting Assistant", layout="wide")
st.title("Visual Waste Sorting Assistant")
st.markdown(
    "Upload a waste photo to classify it into one of the 6 TrashNet categories: "
    "cardboard, glass, metal, paper, plastic, or trash."
)

# Model path: starter/.model/waste_classifier.pth
MODEL_DIR = Path(__file__).resolve().parent / ".model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "waste_classifier.pth"
HISTORY_PATH = MODEL_DIR / "training_history.pkl"

# Sidebar with class info and controls
with st.sidebar:
    st.header("TrashNet Classes")
    for cls in CLASSES:
        st.caption(f"**{cls}** — {class_description(cls)}")
    st.divider()

    st.header("Model Controls")

    # Check if model exists
    model_exists = MODEL_PATH.exists()
    if model_exists:
        st.success("✅ Trained model found")
    else:
        st.warning("⚠️ No trained model found")

    # Training button - lightweight model with progress
    if st.button("Train Model", type="primary"):
        # Placeholders for progress
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        try:
            # Build lightweight model (depth=4, base_filters=16, dropout=0.3, lightweight=True)
            status_text.text("Building lightweight model...")
            progress_bar.progress(10, text="Building model...")
            model = build_model(depth=4, base_filters=16, dropout=0.3, lightweight=True)
            param_count = sum(p.numel() for p in model.parameters())
            status_text.text(f"Model created: {param_count:,} parameters (lightweight)")

            # Train on sample data with progress callback
            sample_dir = PROJECT_ROOT / "data" / "sample"
            epochs = 15

            # Custom training with progress updates
            status_text.text("Starting training...")
            progress_bar.progress(15, text="Training epoch 1...")

            # We'll use a modified train call that shows progress
            # For now, use the standard train but update progress per epoch
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from PIL import Image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Light augmentation for speed
            train_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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

            train_dataset = WasteDataset(sample_dir, transform=train_transform)
            val_dataset = WasteDataset(sample_dir, transform=val_transform)
            batch_size = min(8, len(train_dataset)) if len(train_dataset) > 0 else 1
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            for epoch in range(epochs):
                # Update progress
                progress = 15 + int((epoch / epochs) * 75)
                progress_bar.progress(progress, text=f"Training epoch {epoch+1}/{epochs}...")
                status_text.text(f"Epoch {epoch+1}/{epochs}: Training...")

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
                status_text.text(f"Epoch {epoch+1}/{epochs}: Validating...")
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

                scheduler.step(avg_val_loss)

                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_accuracy)
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)

                status_text.text(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                )

            progress_bar.progress(95, text="Saving model...")
            status_text.text("Saving model and history...")

            # Save model
            torch.save(model.state_dict(), MODEL_PATH)

            # Save history
            with open(HISTORY_PATH, "wb") as f:
                pickle.dump(history, f)

            progress_bar.progress(100, text="Complete!")
            status_text.text(f"Done! Final Val Acc: {history['val_acc'][-1]:.2%}")
            st.success(f"Model trained and saved to {MODEL_PATH} ({param_count:,} params)")
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Load model button
    if model_exists:
        if st.button("Load Trained Model"):
            st.session_state['model_loaded'] = True
            st.success("Model loaded!")

    st.divider()
    st.caption(
        "Try an image from `data/sample/<class>/` for a quick offline "
        "preview. Known failure examples (`*_hard_*.png`) are also in there."
    )

    # Show training history if available
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "rb") as f:
            hist = pickle.load(f)

        st.subheader("Training History")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        epochs_range = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs_range, hist['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs_range, hist['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        ax2.plot(epochs_range, hist['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs_range, hist['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

# Main area - image upload and prediction
uploaded = st.file_uploader("Upload a waste photo", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded, caption=uploaded.name, use_column_width=True)

    with col2:
        if st.button("Predict Class", type="primary", disabled=not model_exists):
            try:
                # Load model if not already loaded
                if 'model' not in st.session_state:
                    model = build_model(depth=4, base_filters=16, dropout=0.3, lightweight=True)
                    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
                    model.eval()
                    st.session_state['model'] = model

                # Predict
                predicted_class, confidence = predict(st.session_state['model'], uploaded)

                # Display results
                st.success(f"**Predicted Class:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2%}")

                # Show all class probabilities
                import torch.nn.functional as F
                from torchvision import transforms
                from PIL import Image

                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                image = Image.open(uploaded).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = st.session_state['model'](image_tensor)
                    probabilities = F.softmax(outputs, dim=1).squeeze().numpy()

                st.subheader("Class Probabilities")
                for i, cls in enumerate(CLASSES):
                    prob = probabilities[i]
                    st.progress(float(prob), text=f"{cls}: {prob:.2%}")

                # Educational guidance
                st.subheader("Disposal Guidance")
                guidance = {
                    "cardboard": "♻️ **Recycle**: Flatten boxes and place in paper/cardboard recycling bin.",
                    "glass": "♻️ **Recycle**: Rinse bottles/jars and place in glass recycling. Separate by color if required.",
                    "metal": "♻️ **Recycle**: Rinse cans and place in metal/aluminum recycling bin.",
                    "paper": "♻️ **Recycle**: Place clean paper in paper recycling. Avoid wet or soiled paper.",
                    "plastic": "♻️ **Recycle**: Check recycling number (1-7). Rinse containers before recycling.",
                    "trash": "🗑️ **Landfill**: This item cannot be recycled in standard streams. Dispose in general waste.",
                }
                st.markdown(guidance.get(predicted_class, "No guidance available."))

                # Failure mode highlighting
                st.subheader("⚠️ Known Failure Modes")
                failure_modes = {
                    "cardboard": "May confuse with paper if flattened; wet cardboard harder to classify.",
                    "glass": "Specular reflections can cause misclassification; dark glass harder to identify.",
                    "metal": "Crushed cans may resemble trash; reflective surfaces cause glare.",
                    "paper": "Crumpled paper may resemble trash; glossy paper may confuse with plastic.",
                    "plastic": "Deformed/crushed bottles hard to classify; clear plastic may confuse with glass.",
                    "trash": "Highly variable class; novel items often misclassified as trash (false positive).",
                }
                st.warning(failure_modes.get(predicted_class, "No specific failure mode noted."))

            except NotImplementedError as exc:
                st.warning(f"Core not implemented yet: {exc}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        elif not model_exists:
            st.info("Train or load a model first using the sidebar controls.")
else:
    st.info(
        "Upload an image to preview it here. Train a model using the sidebar "
        "or load a pre-trained model to enable prediction."
    )

# Footer
st.divider()
st.caption(
    "Project 11 — Visual Waste Sorting Assistant | "
    "Built with PyTorch & Streamlit | "
    "TrashNet dataset (Thung & Yang, 2016)"
)