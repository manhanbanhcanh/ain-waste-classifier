#!/usr/bin/env python3
"""Ablation experiment script for Project 11.

Runs architectural ablation experiments varying:
- Network depth
- Filter count
- Dropout probability
- Data augmentation
- Learning rate

Generates learning curves and comparison plots.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.student_core import build_model, train, CLASSES


def run_experiment(
    name: str,
    train_dir: Path,
    val_dir: Path,
    epochs: int,
    **model_kwargs
) -> dict:
    """Run a single training experiment and return history."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    # Separate model kwargs from train kwargs
    model_params = {}
    train_params = {}

    model_param_names = {'depth', 'base_filters', 'dropout', 'input_shape', 'num_classes'}
    train_param_names = {'lr', 'augmentation', 'epochs'}

    for k, v in model_kwargs.items():
        if k in model_param_names:
            model_params[k] = v
        elif k in train_param_names:
            train_params[k] = v
        else:
            train_params[k] = v

    for k, v in model_params.items():
        print(f"  Model {k}: {v}")
    for k, v in train_params.items():
        print(f"  Train {k}: {v}")

    model = build_model(**model_params)
    history = train(model, train_dir, val_dir, epochs=epochs, **train_params)

    return {
        'name': name,
        'config': {**model_params, **train_params},
        'history': history.history,
        'final_train_acc': history.history['train_acc'][-1],
        'final_val_acc': history.history['val_acc'][-1],
        'final_train_loss': history.history['train_loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
    }


def plot_ablation_results(results: list[dict], output_dir: Path, param_name: str):
    """Plot learning curves for ablation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for result in results:
        label = f"{result['name']} (val_acc={result['final_val_acc']:.3f})"
        epochs = range(1, len(result['history']['train_loss']) + 1)

        axes[0, 0].plot(epochs, result['history']['train_loss'], label=label)
        axes[0, 1].plot(epochs, result['history']['val_loss'], label=label)
        axes[1, 0].plot(epochs, result['history']['train_acc'], label=label)
        axes[1, 1].plot(epochs, result['history']['val_acc'], label=label)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Ablation Study: {param_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"ablation_{param_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['name'] for r in results]
    val_accs = [r['final_val_acc'] for r in results]
    train_accs = [r['final_train_acc'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, train_accs, width, label='Train Acc', alpha=0.7)
    ax.bar(x + width/2, val_accs, width, label='Val Acc', alpha=0.7)

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Final Accuracy Comparison: {param_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"ablation_{param_name.lower().replace(' ', '_')}_summary.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per experiment")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "ablation_results", help="Output directory")
    parser.add_argument("--param", type=str, choices=['depth', 'filters', 'dropout', 'lr', 'augmentation', 'all'],
                        default='all', help="Which parameter to ablate")
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "validation"

    if not train_dir.exists() or not val_dir.exists():
        print("Error: Training/validation data not found.")
        print("Run: python scripts/prepare_dataset.py --confirm")
        return 1

    all_results = []

    if args.param in ['depth', 'all']:
        print("\n\n#### ABLATION: Network Depth ####")
        depth_results = []
        for depth in [3, 4, 5]:
            result = run_experiment(
                f"depth_{depth}",
                train_dir, val_dir, args.epochs,
                depth=depth,
            )
            depth_results.append(result)
            all_results.append(result)

        plot_ablation_results(depth_results, args.output_dir, "Depth")

    if args.param in ['filters', 'all']:
        print("\n\n#### ABLATION: Filter Count ####")
        filter_results = []
        for base_filters in [16, 32, 64]:
            result = run_experiment(
                f"filters_{base_filters}",
                train_dir, val_dir, args.epochs,
                base_filters=base_filters,
            )
            filter_results.append(result)
            all_results.append(result)

        plot_ablation_results(filter_results, args.output_dir, "Filters")

    if args.param in ['dropout', 'all']:
        print("\n\n#### ABLATION: Dropout Probability ####")
        dropout_results = []
        for dropout in [0.2, 0.3, 0.5, 0.7]:
            result = run_experiment(
                f"dropout_{dropout}",
                train_dir, val_dir, args.epochs,
                dropout=dropout,
            )
            dropout_results.append(result)
            all_results.append(result)

        plot_ablation_results(dropout_results, args.output_dir, "Dropout")

    if args.param in ['lr', 'all']:
        print("\n\n#### ABLATION: Learning Rate ####")
        lr_results = []
        for lr in [0.0001, 0.001, 0.01]:
            result = run_experiment(
                f"lr_{lr}",
                train_dir, val_dir, args.epochs,
                lr=lr,
            )
            lr_results.append(result)
            all_results.append(result)

        plot_ablation_results(lr_results, args.output_dir, "Learning Rate")

    if args.param in ['augmentation', 'all']:
        print("\n\n#### ABLATION: Data Augmentation ####")
        aug_results = []
        for aug_name in ["none", "basic", "strong"]:
            result = run_experiment(
                f"aug_{aug_name}",
                train_dir, val_dir, args.epochs,
                augmentation=aug_name,
            )
            aug_results.append(result)
            all_results.append(result)

        plot_ablation_results(aug_results, args.output_dir, "Augmentation")

    # Save all results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {args.output_dir}")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())