#!/usr/bin/env python3
"""Download and split the TrashNet dataset for Project 11.

This script is **optional preflight**, not the student zip gate. Grading and
the offline pytest sanity suite use ``data/sample/`` (see ``data/README.md``),
never this download. Run this only if you want to train on the full ~2500
image TrashNet dataset.

Source (only): https://github.com/garythung/trashnet
Pinned commit: 6fa2b878c6c1b4304b91109070ce0edf9279bb31
License: MIT, Copyright (c) 2017 Gary Thung — see ``data/LICENSE.md``.

Without ``--confirm`` this prints what it *would* do and exits without
touching the network or the filesystem. With ``--confirm`` it downloads the
single pinned archive, verifies its sha256 (optionally against a value you
supply with ``--sha256``), and writes ``data/train/``, ``data/validation/``,
and ``data/test/`` (all git-ignored — never zip these into your submission).

Usage:
    python scripts/prepare_dataset.py            # dry run, no download
    python scripts/prepare_dataset.py --confirm   # actually download + split
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/garythung/trashnet"
PINNED_COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
DATASET_URL = (
    "https://raw.githubusercontent.com/garythung/trashnet/"
    f"{PINNED_COMMIT}/data/dataset-resized.zip"
)
PINNED_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
# 70/13/17 train/validation/test matches the split used in the TrashNet paper.
SPLIT = {"train": 0.70, "validation": 0.13, "test": 0.17}
DEFAULT_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def download(dest: Path) -> None:
    print(f"Source: {REPO_URL} (pinned commit {PINNED_COMMIT})")
    print(f"Downloading {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, dest)  # noqa: S310 - single pinned HTTPS URL


def report_checksum(path: Path, expected_sha256: str | None) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"sha256({path.name}) = {digest}")
    print("Record this digest next to the pinned commit in your midterm report.")
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise SystemExit(
            "Checksum mismatch: the downloaded archive does not match "
            f"--sha256 {expected_sha256!r}. Re-download from the pinned URL "
            "above, or double check the value you passed in."
        )


def extract_and_split(zip_path: Path, out_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path) as zf:
        names_by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            cls = parts[-2]
            if cls in CLASSES and name.lower().endswith((".jpg", ".jpeg", ".png")):
                names_by_class[cls].append(name)

        any_found = False
        for cls, names in names_by_class.items():
            if not names:
                continue
            any_found = True
            rng.shuffle(names)
            n = len(names)
            n_train = int(n * SPLIT["train"])
            n_val = int(n * SPLIT["validation"])
            splits = {
                "train": names[:n_train],
                "validation": names[n_train : n_train + n_val],
                "test": names[n_train + n_val :],
            }
            for split_name, split_names in splits.items():
                split_dir = out_dir / split_name / cls
                split_dir.mkdir(parents=True, exist_ok=True)
                for name in split_names:
                    with zf.open(name) as src, open(split_dir / Path(name).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            print(
                f"{cls}: {n} images -> "
                f"train={len(splits['train'])} "
                f"validation={len(splits['validation'])} "
                f"test={len(splits['test'])}"
            )
        if not any_found:
            raise SystemExit(
                "No images found for any of the six TrashNet classes inside "
                "the archive. The zip layout may have changed upstream; "
                "check data/LICENSE.md and the pinned commit before retrying."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually download the pinned archive and write data/train|validation|test/.",
    )
    parser.add_argument(
        "--sha256",
        default=PINNED_SHA256,
        help="Expected sha256 of the pinned archive (see data/LICENSE.md). Pass empty to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Shuffle seed for the train/validation/test split (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip after extracting (default: delete it).",
    )
    args = parser.parse_args(argv)

    if not args.confirm:
        print("Dry run (default, no network or filesystem access). This would:")
        print(f"  1. Download {DATASET_URL}")
        print(f"  2. Verify sha256 == {PINNED_SHA256}")
        print(
            f"  3. Split into {DATA_DIR}/train|validation|test/ "
            f"(70/13/17 per class, seed={args.seed})"
        )
        print("Re-run with --confirm to actually do this. Read data/LICENSE.md first.")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "dataset-resized.zip"
    download(zip_path)
    expected = args.sha256 or None
    report_checksum(zip_path, expected)
    extract_and_split(zip_path, DATA_DIR, args.seed)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print(
        "Done. data/train/, data/validation/, data/test/ are ready and "
        "git-ignored — do not include them in your submission zip."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
