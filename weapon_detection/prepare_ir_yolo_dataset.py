from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from weapon_detection/ir_gun and ir_nogun."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="weapon_detection root directory",
    )
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_clean_dirs(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_items(items: list[Path], train: float, val: float, seed: int) -> dict[str, list[Path]]:
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val : n_train + n_val + n_test],
    }


def copy_pair(image_path: Path, label_path: Path, out_images: Path, out_labels: Path) -> None:
    shutil.copy2(image_path, out_images / image_path.name)
    shutil.copy2(label_path, out_labels / label_path.name)


def write_yaml(dataset_dir: Path) -> None:
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "gun"},
    }
    with (dataset_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    if abs((args.train + args.val + args.test) - 1.0) > 1e-9:
        raise ValueError("train + val + test must sum to 1.0")

    root = args.root.resolve()
    ir_gun = root / "ir_gun"
    ir_nogun = root / "ir_nogun"
    dataset_dir = root / "dataset"

    if not ir_gun.is_dir() or not ir_nogun.is_dir():
        raise FileNotFoundError(
            "Expected directories weapon_detection/ir_gun and weapon_detection/ir_nogun"
        )

    ensure_clean_dirs(dataset_dir)

    gun_images = sorted([p for p in ir_gun.iterdir() if p.suffix.lower() in VALID_EXTS])
    nogun_images = sorted([p for p in ir_nogun.iterdir() if p.suffix.lower() in VALID_EXTS])

    missing_labels: list[Path] = []
    valid_gun_images: list[Path] = []
    for img in gun_images:
        label = img.with_suffix(".txt")
        if label.is_file():
            valid_gun_images.append(img)
        else:
            missing_labels.append(img)

    if not valid_gun_images and gun_images:
        print("No labeled gun images found. Add YOLO .txt labels in ir_gun to continue.")

    gun_split = split_items(valid_gun_images, args.train, args.val, args.seed)
    nogun_split = split_items(nogun_images, args.train, args.val, args.seed)

    for split in ("train", "val", "test"):
        out_i = dataset_dir / "images" / split
        out_l = dataset_dir / "labels" / split

        for img in gun_split[split]:
            copy_pair(img, img.with_suffix(".txt"), out_i, out_l)

        for img in nogun_split[split]:
            target_img = out_i / img.name
            target_lbl = out_l / img.with_suffix(".txt").name
            shutil.copy2(img, target_img)
            target_lbl.write_text("", encoding="utf-8")

    write_yaml(dataset_dir)

    print(f"Dataset ready at: {dataset_dir}")
    print(
        "Counts -> labeled_gun: "
        f"{len(valid_gun_images)}, nogun: {len(nogun_images)}, missing_gun_labels: {len(missing_labels)}"
    )
    if missing_labels:
        print("Missing labels for these gun images:")
        for p in missing_labels:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
