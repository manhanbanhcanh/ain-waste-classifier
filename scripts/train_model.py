#!/usr/bin/env python3
"""Training script for Project 11 waste classifier.

This script trains the CNN on the full TrashNet dataset (after running
prepare_dataset.py --confirm) and saves the model and training history.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import build_model, train, CLASSES


def main():
    parser = argparse.ArgumentParser(description="Train waste classifier on TrashNet")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Data directory")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models", help="Model output directory")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--depth", type=int, default=4, choices=[3, 4, 5], help="Network depth (conv blocks)")
    parser.add_argument("--base-filters", type=int, default=32, help="Base filter count")
    parser.add_argument("--augmentation", type=str, default="basic", choices=["none", "basic", "strong"], help="Data augmentation strategy")
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "validation"

    if not train_dir.exists() or not val_dir.exists():
        print("Error: Training/validation data not found.")
        print("Run: python scripts/prepare_dataset.py --confirm")
        return 1

    print(f"Training on {train_dir}")
    print(f"Validating on {val_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Depth: {args.depth}")
    print(f"Base filters: {args.base_filters}")
    print(f"Augmentation: {args.augmentation}")

    # Build model with ablation parameters
    model = build_model(
        depth=args.depth,
        base_filters=args.base_filters,
        dropout=args.dropout,
    )

    # Train
    history = train(
        model,
        train_dir,
        val_dir,
        epochs=args.epochs,
        lr=args.lr,
        augmentation=args.augmentation,
    )

    # Save model
    args.model_dir.mkdir(exist_ok=True)
    model_path = args.model_dir / "waste_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save history
    history_path = args.model_dir / "training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"History saved to {history_path}")

    # Print final metrics
    final_train_acc = history.history['train_acc'][-1]
    final_val_acc = history.history['val_acc'][-1]
    final_train_loss = history.history['train_loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"\nFinal Results:")
    print(f"  Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.4f}")
    print(f"  Val Loss:   {final_val_loss:.4f}, Val Acc:   {final_val_acc:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())