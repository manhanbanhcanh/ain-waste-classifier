#!/usr/bin/env python3
"""Train the WasteNet model and save to models/best_model_{variant}.pt.

This script trains the model using the TrashNet dataset and saves the
trained weights for use in the Streamlit app.

Usage:
    python tests/train.py                          # Quick train on sample data (fast)
    python tests/train.py --full                   # Train on full dataset (requires data/train/)
    python tests/train.py --variant smart --full --epochs 40  # Train Smart variant
    python tests/train.py --variant lite --full --epochs 20   # Train Lite variant
    python tests/train.py --epochs 30              # Custom epochs
    python tests/train.py --device cpu             # Force CPU training
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import CLASSES, MODEL_VARIANTS, build_model, train


def main():
    parser = argparse.ArgumentParser(description="Train WasteNet model for waste classification")
    parser.add_argument(
        "--variant",
        choices=MODEL_VARIANTS,
        default="standard",
        help="Model variant to train: lite, standard, or smart (default: standard)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset from data/train/ and data/validation/ (requires prepare_dataset.py --confirm first)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output model path (default: models/best_model_{variant}.pt)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars during training (requires tqdm)",
    )
    args = parser.parse_args()

    # Set default output path based on variant
    if args.output is None:
        args.output = PROJECT_ROOT / "models" / f"best_model_{args.variant}.pt"

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model variant: {args.variant}")

    # Determine data directories
    if args.full:
        train_dir = PROJECT_ROOT / "data" / "train"
        val_dir = PROJECT_ROOT / "data" / "validation"
        if not train_dir.exists() or not val_dir.exists():
            print("ERROR: Full dataset not found at data/train/ and data/validation/")
            print("Run: python scripts/prepare_dataset.py --confirm")
            sys.exit(1)
        print("Using full dataset from data/")
    else:
        # Use sample data for quick testing
        train_dir = PROJECT_ROOT / "data" / "sample"
        val_dir = PROJECT_ROOT / "data" / "sample"
        print("Using sample data from data/sample/ (quick test mode)")

    # Build model with selected variant
    print(f"\nBuilding WasteNet-{args.variant.capitalize()} model...")
    model = build_model(input_shape=(128, 128, 3), num_classes=len(CLASSES), variant=args.variant)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train dir: {train_dir}")
    print(f"  Val dir: {val_dir}")

    start_time = time.time()

    try:
        history = train(
            model=model,
            train_dir=train_dir,
            validation_dir=val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            progress=args.progress,
        )

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Print final metrics
        print("\nFinal Results:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Train Acc:  {history['train_acc'][-1]:.2f}%")
        print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
        print(f"  Val Acc:    {history['val_acc'][-1]:.2f}%")

        # Save model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"\nModel saved to: {args.output}")

        # Verify it can be loaded
        print("Verifying model can be loaded...")
        test_model = build_model(input_shape=(128, 128, 3), num_classes=len(CLASSES), variant=args.variant)
        test_model.load_state_dict(torch.load(args.output, map_location="cpu", weights_only=True))
        test_model.eval()
        print("[OK] Model verified successfully!")

        if not args.full:
            print("\nNOTE: This was trained on sample data only (1-2 images per class).")
            print("For production use, run with --full after preparing the full dataset:")
            print("  python scripts/prepare_dataset.py --confirm")
            print(f"  python tests/train.py --variant {args.variant} --full --epochs 30")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()