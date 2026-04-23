from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IR gun detection inference and save boxed images."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained weights (best.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "inference_outputs",
        help="Directory to save annotated images",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted([p for p in source.iterdir() if p.suffix.lower() in VALID_EXTS])
    raise FileNotFoundError(f"Source not found: {source}")


def draw_and_save(img_path: Path, boxes, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        label = f"gun {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            img,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No images found in source: {source}")

    model = YOLO(str(weights))

    gun_count = 0
    for img_path in images:
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        result = results[0]
        boxes = result.boxes
        has_gun = boxes is not None and len(boxes) > 0
        status = "GUN" if has_gun else "NO_GUN"
        if has_gun:
            gun_count += 1

        out_path = output_dir / img_path.name
        draw_and_save(img_path, boxes if boxes is not None else [], out_path)

        confs = []
        if boxes is not None:
            confs = [f"{float(b.conf[0]):.2f}" for b in boxes]
        print(f"{img_path.name}: {status}" + (f" conf={','.join(confs)}" if confs else ""))

    print(
        f"\nDone. Processed {len(images)} image(s). "
        f"Detected gun in {gun_count} image(s). Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
