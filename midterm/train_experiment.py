"""
# project 11 ai core — training and experiment script

this script provides functionality to:
1. train the cnn model implemented in student_core.py
2. conduct architectural ablation studies as required by the midterm
3. plot training/validation loss and accuracy curves
4. compare performance against naive baselines
5. save experiment results for reporting

required experiment (from midterm specifications):
- vary at least one design parameter: network depth, filter count, dropout,
  augmentation strategy, or learning rate
- plot training/validation curves across epochs
- analyze overfitting onset
- report macro-averaged metrics on held-out test split
- compare against naive baseline (majority-class or color-histogram nearest-centroid)

usage:
    python train_experiment.py --help
    python train_experiment.py --mode ablation --param dropout --values 0.3 0.5 0.7
    python train_experiment.py --mode baseline --model_path path/to/trained/model.pt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# heavy imports are placed inside functions to allow offline tests to work
# without requiring tensorflow/pytorch to be installed

import sys
from pathlib import Path
# Add the parent directory to sys.path so we can import starter
sys.path.append(str(Path(__file__).parent.parent))


def run_baseline_experiment(data_dir: Path) -> Dict[str, float | int | str]:
    """
    run a naive baseline experiment for comparison.

    implements a majority-class classifier as the naive baseline.
    returns macro-averaged accuracy and additional info.
    """
    # import inside function to maintain compatibility with offline tests
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import numpy as np

    # simple transform for baseline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # load test data
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # get all labels to compute majority class
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())

    # find majority class
    unique, counts = np.unique(all_labels, return_counts=True)
    majority_class = unique[np.argmax(counts)]

    # calculate accuracy of always predicting majority class
    correct = sum(1 for label in all_labels if label == majority_class)
    accuracy = correct / len(all_labels)

    # for macro-averaged metrics with majority class predictor:
    # precision, recall, and f1 will be 0 for all classes except majority class
    # for simplicity, we report accuracy as the main metric
    return {
        'accuracy': accuracy,
        'majority_class': majority_class.item() if hasattr(majority_class, 'item') else int(majority_class),
        'note': 'macro-averaged precision/recall/f1 are 0 for non-majority classes'
    }


def run_training_experiment(
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    model_config: Dict[str, Any],
    epochs: int = 25
) -> Dict[str, Any]:
    """
    train a model with given configuration and return training history.

    args:
        train_dir: path to training data
        validation_dir: path to validation data
        test_dir: path to test data
        model_config: dictionary containing model parameters
            - input_shape: tuple of (height, width, channels)
            - num_classes: int
            - lr: float (learning rate)
            - dropout: float (dropout rate)
            - conv_filters: list of ints (e.g., [16, 32, 64])
        epochs: number of training epochs

    returns:
        dictionary containing training history and final test results
    """
    # import inside function
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from starter.student_core import build_model, predict, CLASSES

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create transforms
    transform = transforms.Compose([
        transforms.Resize((model_config['input_shape'][0], model_config['input_shape'][1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])

    # load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # create data loaders
    batch_size = model_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # build model
    model = build_model(
        input_shape=model_config['input_shape'],
        num_classes=model_config['num_classes']
    )
    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config.get('lr', 0.001))

    # training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # training loop
    for epoch in range(epochs):
        # training phase
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

        print(f'epoch {epoch+1}/{epochs}: '
              f'train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%, '
              f'val loss: {validation_loss:.4f}, val acc: {validation_acc:.2f}%')

    # final test evaluation
    model.eval()
    correct_test = 0
    total_test = 0
    class_correct = [0] * model_config['num_classes']
    class_total = [0] * model_config['num_classes']

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_acc = 100 * correct_test / total_test
    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                 for i in range(model_config['num_classes'])]

    return {
        'history': history,
        'test_accuracy': test_acc,
        'class_accuracies': dict(zip(CLASSES, class_acc)),
        'final_model': model,
        'config': model_config
    }


def plot_training_curves(history: Dict[str, List[float]], save_path: Path | None = None):
    """
    plot training and validation loss/accuracy curves.

    args:
        history: dictionary containing train_loss, train_acc, val_loss, val_acc lists
        save_path: optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(history['train_loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='training loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='validation loss')
        ax1.set_title('training and validation loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='training accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='validation accuracy')
        ax2.set_title('training and validation accuracy')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"training curves saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available. skipping plot generation.")
        print("to install: pip install matplotlib")


def run_ablation_study(
    data_dir: Path,
    param_name: str,
    param_values: List[Any],
    base_config: Dict[str, Any],
    epochs: int = 20
) -> Dict[str, Any]:
    """
    run an ablation study varying a specific parameter.

    args:
        data_dir: path to data directory containing train/validation/test
        param_name: name of parameter to vary (e.g., 'dropout', 'lr', 'conv_filters')
        param_values: list of values to try for the parameter
        base_config: base configuration dictionary
        epochs: number of epochs to train each configuration

    returns:
        dictionary containing results for each parameter value
    """
    results = {}

    for value in param_values:
        print(f"\n{'='*50}")
        print(f"testing {param_name} = {value}")
        print(f"{'='*50}")

        # update config with current parameter value
        config = base_config.copy()
        config[param_name] = value

        # train model
        result = run_training_experiment(
            train_dir=data_dir / "train",
            validation_dir=data_dir / "validation",
            test_dir=data_dir / "test",
            model_config=config,
            epochs=epochs
        )

        results[str(value)] = {
            'test_accuracy': result['test_accuracy'],
            'class_accuracies': result['class_accuracies'],
            'final_epoch_val_acc': result['history']['val_acc'][-1],
            'final_epoch_val_loss': result['history']['val_loss'][-1],
            'config_used': config
        }

        # plot and save curves for this configuration
        plot_path = data_dir.parent / f"dev-test" / "missed" / f"curves_{param_name}_{value}.png"
        plot_training_curves(result['history'], save_path=plot_path)

    return results


def compare_with_baseline(
    test_results: Dict[str, float],
    baseline_results: Dict[str, float]
) -> Dict[str, Any]:
    """
    compare model results with naive baseline.

    args:
        test_results: results from trained model
        baseline_results: results from naive baseline

    returns:
        dictionary containing comparison metrics
    """
    improvement = test_results['accuracy'] - baseline_results['accuracy']
    relative_improvement = (improvement / baseline_results['accuracy']) * 100 if baseline_results['accuracy'] > 0 else 0

    return {
        'model_accuracy': test_results['accuracy'],
        'baseline_accuracy': baseline_results['accuracy'],
        'absolute_improvement': improvement,
        'relative_improvement_percent': relative_improvement,
        'is_better': improvement > 0
    }


def main():
    parser = argparse.ArgumentParser(description='run training experiments for project 11')
    parser.add_argument('--mode', choices=['baseline', 'single', 'ablation'],
                       default='single', help='experiment mode')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='path to data directory (default: data)')
    parser.add_argument('--param', type=str, default='dropout',
                       help='parameter to vary in ablation study (default: dropout)')
    parser.add_argument('--values', type=str, nargs='+',
                       default=['0.3', '0.5', '0.7'],
                       help='values to test in ablation study')
    parser.add_argument('--epochs', type=int, default=20,
                       help='number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='learning rate (default: 0.001)')

    args = parser.parse_args()

    # convert string values to appropriate types
    def convert_value(val):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val  # keep as string if not numeric

    param_values = [convert_value(v) for v in args.values]

    # base configuration
    base_config = {
        'input_shape': (128, 128, 3),
        'num_classes': 6,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'dropout': 0.5,  # default dropout rate
    }

    data_dir = Path(args.data_dir)

    if args.mode == 'baseline':
        print("running baseline experiment...")
        if not (data_dir / "test").exists():
            print("error: test data not found. please run prepare_dataset.py --confirm first.")
            return

        baseline_results = run_baseline_experiment(data_dir)
        print(f"\nbaseline results:")
        print(f"  accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  majority class: {baseline_results['majority_class']} ({list(baseline_results.keys())[0] if len(baseline_results) > 1 else 'n/a'})")

        # save results
        results_dir = Path("../dev-test/missed")
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "baseline_results.json", 'w') as f:
            json.dump(baseline_results, f, indent=2)

    elif args.mode == 'single':
        print("running single training experiment...")
        if not all((data_dir / d).exists() for d in ["train", "validation", "test"]):
            print("error: train/validation/test data not found. please run prepare_dataset.py --confirm first.")
            return

        print(f"training with config: {base_config}")
        result = run_training_experiment(
            train_dir=data_dir / "train",
            validation_dir=data_dir / "validation",
            test_dir=data_dir / "test",
            model_config=base_config,
            epochs=args.epochs
        )

        print(f"\nfinal results:")
        print(f"  test accuracy: {result['test_accuracy']:.2f}%")
        print(f"  class accuracies:")
        for cls, acc in result['class_accuracies'].items():
            print(f"    {cls}: {acc:.2f}%")

        # plot training curves
        plot_path = Path("../dev-test/missed") / "training_curves.png"
        plot_training_curves(result['history'], save_path=plot_path)

        # save results
        results_dir = Path("../dev-test/missed")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_to_save = {
            'test_accuracy': result['test_accuracy'],
            'class_accuracies': result['class_accuracies'],
            'final_epoch_val_acc': result['history']['val_acc'][-1],
            'final_epoch_val_loss': result['history']['val_loss'][-1],
            'config': base_config
        }
        with open(results_dir / "single_experiment_results.json", 'w') as f:
            json.dump(results_to_save, f, indent=2)

    elif args.mode == 'ablation':
        print(f"running ablation study on parameter: {args.param}")
        print(f"values to test: {param_values}")

        if not all((data_dir / d).exists() for d in ["train", "validation", "test"]):
            print("error: train/validation/test data not found. please run prepare_dataset.py --confirm first.")
            return

        # run baseline for comparison
        print("\nrunning baseline experiment for comparison...")
        baseline_results = run_baseline_experiment(data_dir)

        # run ablation study
        ablation_results = run_ablation_study(
            data_dir=data_dir,
            param_name=args.param,
            param_values=param_values,
            base_config=base_config,
            epochs=args.epochs
        )

        # compare each ablation result with baseline
        comparisons = {}
        for value_str, result in ablation_results.items():
            # convert result to format expected by compare_with_baseline
            test_result_format = {
                'accuracy': result['test_accuracy'] / 100.0  # convert percentage to proportion
            }
            comparison = compare_with_baseline(test_result_format, baseline_results)
            comparisons[value_str] = comparison

            print(f"\n{args.param} = {value_str}:")
            print(f"  model accuracy: {result['test_accuracy']:.2f}%")
            print(f"  baseline accuracy: {baseline_results['accuracy']*100:.2f}%")
            print(f"  improvement: {comparison['absolute_improvement']*100:+.2f}% "
                  f"({comparison['relative_improvement_percent']:+.2f}%)")

        # save all results
        results_dir = Path("dev-test/missed")
        results_dir.mkdir(parents=True, exist_ok=True)

        # save ablation results
        with open(results_dir / f"ablation_{args.param}_results.json", 'w') as f:
            json.dump({
                'param_name': args.param,
                'param_values': param_values,
                'results': ablation_results,
                'baseline': baseline_results,
                'comparisons': comparisons
            }, f, indent=2)

        # create summary report
        summary_lines = [
            f"ablation study summary: {args.param}",
            f"{'='*50}",
            f"baseline accuracy: {baseline_results['accuracy']*100:.2f}%",
            ""
        ]

        for value_str, result in ablation_results.items():
            comp = comparisons[value_str]
            summary_lines.extend([
                f"{args.param} = {value_str}:",
                f"  test accuracy: {result['test_accuracy']:.2f}%",
                f"  improvement: {comp['absolute_improvement']*100:+.2f}% "
                f"({comp['relative_improvement_percent']:+.2f}%)",
                ""
            ])

        with open(results_dir / f"ablation_{args.param}_summary.txt", 'w') as f:
            f.write('\n'.join(summary_lines))

        print(f"\nresults saved to {results_dir}/")


if __name__ == "__main__":
    main()