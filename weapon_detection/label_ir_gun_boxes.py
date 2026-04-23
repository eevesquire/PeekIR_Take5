from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "IR Gun Labeler"


class LabelSession:
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.idx = 0
        self.start_pt: tuple[int, int] | None = None
        self.temp_pt: tuple[int, int] | None = None
        self.is_drawing = False
        self.boxes_per_image: dict[Path, list[tuple[int, int, int, int]]] = {}

    def current_path(self) -> Path:
        return self.image_paths[self.idx]

    def load_existing_boxes(self, img_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
        txt = img_path.with_suffix(".txt")
        if not txt.is_file():
            return []

        boxes: list[tuple[int, int, int, int]] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = parts
            if cls != "0":
                continue
            xcf, ycf, bwf, bhf = float(xc), float(yc), float(bw), float(bh)
            x1 = int((xcf - bwf / 2.0) * w)
            y1 = int((ycf - bhf / 2.0) * h)
            x2 = int((xcf + bwf / 2.0) * w)
            y2 = int((ycf + bhf / 2.0) * h)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def get_boxes(self, img_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
        if img_path not in self.boxes_per_image:
            self.boxes_per_image[img_path] = self.load_existing_boxes(img_path, w, h)
        return self.boxes_per_image[img_path]

    def save_boxes(self, img_path: Path, boxes: list[tuple[int, int, int, int]], w: int, h: int) -> None:
        lines: list[str] = []
        for x1, y1, x2, y2 in boxes:
            x1c = max(0, min(w - 1, min(x1, x2)))
            y1c = max(0, min(h - 1, min(y1, y2)))
            x2c = max(0, min(w - 1, max(x1, x2)))
            y2c = max(0, min(h - 1, max(y1, y2)))
            bw = x2c - x1c
            bh = y2c - y1c
            if bw < 2 or bh < 2:
                continue
            xc = (x1c + x2c) / 2.0 / w
            yc = (y1c + y2c) / 2.0 / h
            wn = bw / w
            hn = bh / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out = img_path.with_suffix(".txt")
        out.write_text("\n".join(lines), encoding="utf-8")

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pt = (x, y)
            self.temp_pt = (x, y)
            self.is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.temp_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing and self.start_pt is not None:
            self.temp_pt = (x, y)
            x1, y1 = self.start_pt
            x2, y2 = self.temp_pt
            img = cv2.imread(str(self.current_path()))
            if img is not None:
                h, w = img.shape[:2]
                boxes = self.get_boxes(self.current_path(), w, h)
                boxes.append((x1, y1, x2, y2))
            self.is_drawing = False
            self.start_pt = None
            self.temp_pt = None

    def draw(self, img, boxes: list[tuple[int, int, int, int]]):
        canvas = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                canvas,
                f"gun #{i + 1}",
                (min(x1, x2), max(18, min(y1, y2) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.is_drawing and self.start_pt and self.temp_pt:
            cv2.rectangle(canvas, self.start_pt, self.temp_pt, (80, 200, 255), 1)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            img_path = self.current_path()
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                self.idx = (self.idx + 1) % len(self.image_paths)
                continue

            h, w = img.shape[:2]
            boxes = self.get_boxes(img_path, w, h)
            view = self.draw(img, boxes)

            help_text = (
                f"{self.idx + 1}/{len(self.image_paths)}  {img_path.name} | "
                "Draw: LMB drag | u:undo | c:clear | s:save | n:next | p:prev | q:quit"
            )
            cv2.putText(
                view,
                help_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, view)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u") and boxes:
                boxes.pop()
            elif key == ord("c"):
                boxes.clear()
            elif key == ord("s"):
                self.save_boxes(img_path, boxes, w, h)
                print(f"Saved labels: {img_path.with_suffix('.txt').name} ({len(boxes)} box(es))")
            elif key == ord("n"):
                self.save_boxes(img_path, boxes, w, h)
                self.idx = (self.idx + 1) % len(self.image_paths)
            elif key == ord("p"):
                self.save_boxes(img_path, boxes, w, h)
                self.idx = (self.idx - 1) % len(self.image_paths)

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label gun bounding boxes in IR images and save YOLO .txt labels."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parent / "ir_gun",
        help="Directory with gun images to label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    print(f"Loaded {len(image_paths)} image(s) from {images_dir}")
    print("Controls: LMB drag box | u undo | c clear | s save | n next | p prev | q quit")
    LabelSession(image_paths).run()


if __name__ == "__main__":
    main()
