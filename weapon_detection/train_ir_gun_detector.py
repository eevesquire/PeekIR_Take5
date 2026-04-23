from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO IR gun detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset" / "data.yaml",
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model checkpoint (e.g. yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto/cpu/0. 'auto' picks CUDA if available else cpu.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "detect",
        help="Output project directory",
    )
    parser.add_argument("--name", type=str, default="ir_gun_detector", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Resolved device=auto -> {device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
    )

    print("Training complete.")
    print(
        "Best weights should be under: "
        f"{(args.project / args.name / 'weights' / 'best.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
