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
    raise NotImplementedError("Implement build_model() with your own CNN architecture.")


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
    raise NotImplementedError("Implement train() with your own training loop.")


def predict(model: Any, image_path: Path) -> tuple[str, float]:
    """Return ``(predicted_class, confidence)`` for the image at ``image_path``.

    ``predicted_class`` must be one of ``CLASSES``. ``confidence`` is a
    float in ``[0, 1]``.
    """
    raise NotImplementedError("Implement predict() using your trained model.")
