"""Split Objects365 val set: 5K for validation, 75K moved to training via symlinks.

Replicates the D-FINE / LW-DETR / RF-DETR convention where only the first 5,000
validation images are kept for evaluation and the remaining ~75K are added to training.

Usage:
    python working_dir/split_obj365_val.py --dataset-dir /path/to/Objects365
    python working_dir/split_obj365_val.py --dataset-dir /path/to/Objects365 --val-size 5000 --dry-run
"""

import argparse
import os
from pathlib import Path

from ultralytics.utils import TQDM


def create_symlinks(src_files, dst_dir, dry_run=False):
    """Create symlinks in dst_dir pointing to src_files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for src in TQDM(src_files, desc=f"Symlinking to {dst_dir.name}"):
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        if not dry_run:
            os.symlink(src.resolve(), dst)
        created += 1
    return created


def main():
    parser = argparse.ArgumentParser(description="Split Objects365 val into val_5k + val_as_train")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Root of Objects365 dataset")
    parser.add_argument("--val-size", type=int, default=5000, help="Number of images to keep for validation")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without creating symlinks")
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    val_size = args.val_size

    images_val = root / "images" / "val"
    labels_val = root / "labels" / "val"

    assert images_val.is_dir(), f"Not found: {images_val}"
    assert labels_val.is_dir(), f"Not found: {labels_val}"

    # Collect and sort val images
    val_images = sorted(images_val.glob("*.*"))
    val_images = [f for f in val_images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    print(f"Found {len(val_images)} val images")

    assert len(val_images) > val_size, f"Val set ({len(val_images)}) must be larger than val_size ({val_size})"

    # Split: first val_size for validation, rest for training
    keep_val = val_images[:val_size]
    move_to_train = val_images[val_size:]
    print(f"Keeping {len(keep_val)} for val_5k, moving {len(move_to_train)} to val_as_train")

    # Find corresponding label files
    keep_val_labels = [labels_val / f.with_suffix(".txt").name for f in keep_val]
    move_to_train_labels = [labels_val / f.with_suffix(".txt").name for f in move_to_train]

    # Check label coverage
    missing_labels = sum(1 for lb in keep_val_labels + move_to_train_labels if not lb.exists())
    if missing_labels:
        print(f"Warning: {missing_labels} label files not found (background images without annotations)")

    # Create symlinked directories
    for name, img_files, lbl_files in [
        ("val_5k", keep_val, keep_val_labels),
        ("val_as_train", move_to_train, move_to_train_labels),
    ]:
        img_dst = root / "images" / name
        lbl_dst = root / "labels" / name

        n_img = create_symlinks(img_files, img_dst, dry_run=args.dry_run)
        # Only symlink labels that exist (some images may have no annotations)
        existing_labels = [lb for lb in lbl_files if lb.exists()]
        n_lbl = create_symlinks(existing_labels, lbl_dst, dry_run=args.dry_run)

        action = "Would create" if args.dry_run else "Created"
        print(f"{action} {n_img} image + {n_lbl} label symlinks in {name}/")

    # Verify final split counts
    print(f"\n--- Split verification ---")
    images_train = root / "images" / "train"
    images_val_as_train = root / "images" / "val_as_train"
    images_val_5k = root / "images" / "val_5k"

    n_train_orig = len(list(images_train.glob("*.*"))) if images_train.is_dir() else 0
    n_val_as_train = len(list(images_val_as_train.glob("*.*"))) if images_val_as_train.is_dir() else 0
    n_val_5k = len(list(images_val_5k.glob("*.*"))) if images_val_5k.is_dir() else 0
    n_total_train = n_train_orig + n_val_as_train

    print(f"  images/train:         {n_train_orig:>10,} images")
    print(f"  images/val_as_train:  {n_val_as_train:>10,} images (from val)")
    print(f"  Total training:       {n_total_train:>10,} images")
    print(f"  images/val_5k:        {n_val_5k:>10,} images")
    print(f"  images/val (original):{len(val_images):>10,} images (untouched)")

    # Print dataset YAML snippet
    print("\n--- Add this to your dataset YAML ---")
    print(f"path: {root}")
    print("train: [images/train, images/val_as_train]")
    print("val: images/val_5k")


if __name__ == "__main__":
    main()
