"""Offline sanity tests for Project 11 (visual waste sorting assistant).

These tests do not grade the AI core. They check that:

- the pinned TrashNet source/license/checksum method is documented;
- ``data/sample/`` ships small original images for every class, and only
  those small images (no full dataset committed);
- no pretrained weight files (``.h5``/``.pkl``/``.pt``/``.ckpt``) exist;
- the starter modules import fine without TensorFlow installed;
- the graded functions are still empty stubs (``raise NotImplementedError``).

They must pass **offline** using ``data/sample/`` only, and never require
(or trigger) the full TrashNet download.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    """Load a module by explicit file path under a unique ``sys.modules`` key.

    Every project in this kit names its starter package ``starter``. Using a
    plain ``import starter.student_core`` would collide in ``sys.modules``
    when this project's tests run in the same pytest session as another
    project's tests (see the kit-wide verification command, which runs both
    Project 11 and Project 12 sanity tests together). Loading by path avoids
    that collision entirely.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student_core = _load_module(
    "project_11_waste_classifier_student_core", PROJECT_ROOT / "starter" / "student_core.py"
)

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
FORBIDDEN_WEIGHT_SUFFIXES = (".h5", ".pkl", ".pt", ".ckpt")


def test_classes_match_trashnet_six():
    assert student_core.CLASSES == CLASSES


def test_license_file_documents_source_and_checksum_method():
    license_path = PROJECT_ROOT / "data" / "LICENSE.md"
    assert license_path.is_file(), "data/LICENSE.md is missing"
    text = license_path.read_text(encoding="utf-8")
    assert "github.com/garythung/trashnet" in text
    assert "MIT" in text
    assert "checksum" in text.lower()
    assert "sha256" in text.lower()


def test_sample_data_has_every_class_and_stays_small():
    sample_dir = PROJECT_ROOT / "data" / "sample"
    assert sample_dir.is_dir(), "data/sample/ is missing"
    all_images: list[Path] = []
    for cls in CLASSES:
        cls_dir = sample_dir / cls
        assert cls_dir.is_dir(), f"data/sample/{cls}/ is missing"
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        assert images, f"data/sample/{cls}/ has no sample images"
        all_images.extend(images)
    assert len(all_images) <= 20, "data/sample/ must stay <=20 images for offline smoke tests"


def test_sample_data_includes_a_hard_variant():
    sample_dir = PROJECT_ROOT / "data" / "sample"
    hard_images = list(sample_dir.rglob("*hard*"))
    assert hard_images, "data/sample/ should ship at least one hard/blur/dark variant"


def test_sample_images_are_valid_png_bytes():
    sample_dir = PROJECT_ROOT / "data" / "sample"
    png_magic = b"\x89PNG\r\n\x1a\n"
    pngs = list(sample_dir.rglob("*.png"))
    assert pngs, "no PNG sample images found"
    for path in pngs:
        header = path.read_bytes()[:8]
        assert header == png_magic, f"{path} is not a valid PNG"


def test_full_dataset_is_not_committed():
    data_dir = PROJECT_ROOT / "data"
    for split in ("train", "validation", "test", "raw", "trashnet", "dataset-resized"):
        split_dir = data_dir / split
        assert not split_dir.exists(), (
            f"data/{split}/ must not be committed; it is written only by "
            "`scripts/prepare_dataset.py --confirm` and is git-ignored"
        )
    assert not (data_dir / "dataset-resized.zip").exists(), (
        "the TrashNet archive must not be committed into the student tree"
    )


def test_no_pretrained_or_solution_weight_files():
    for path in PROJECT_ROOT.rglob("*"):
        assert path.suffix.lower() not in FORBIDDEN_WEIGHT_SUFFIXES, (
            f"forbidden weight file found: {path}"
        )


def test_build_model_is_not_implemented():
    with pytest.raises(NotImplementedError):
        student_core.build_model()


def test_train_is_not_implemented():
    sample_dir = PROJECT_ROOT / "data" / "sample"
    with pytest.raises(NotImplementedError):
        student_core.train(model=None, train_dir=sample_dir, validation_dir=sample_dir)


def test_predict_is_not_implemented():
    sample_image = next((PROJECT_ROOT / "data" / "sample" / "cardboard").glob("*.png"))
    with pytest.raises(NotImplementedError):
        student_core.predict(model=None, image_path=sample_image)


def test_class_description_covers_every_class():
    for cls in CLASSES:
        description = student_core.class_description(cls)
        assert isinstance(description, str) and description


def test_starter_app_compiles():
    import py_compile

    py_compile.compile(str(PROJECT_ROOT / "starter" / "app.py"), doraise=True)


def test_prepare_dataset_dry_run_touches_nothing():
    """--confirm-less invocation must not hit the network or filesystem."""
    import runpy

    data_dir = PROJECT_ROOT / "data"
    before = {p for p in data_dir.iterdir()} if data_dir.exists() else set() # type: ignore

    argv_backup = sys.argv
    sys.argv = ["prepare_dataset.py"]
    try:
        runpy.run_path(str(PROJECT_ROOT / "scripts" / "prepare_dataset.py"), run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (0, None)
    finally:
        sys.argv = argv_backup

    after = {p for p in data_dir.iterdir()} if data_dir.exists() else set() # type: ignore
    assert before == after, "dry-run prepare_dataset.py must not write to data/"
