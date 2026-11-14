"""
AI metal / no-metal check on one IR frame (the teammate's ThermalCNN, experimental).

Used by the dashboard's "Capture IR" button. It is a separate opinion next to the rule-based
detector (ir_metal_size.py): it says metal / no metal with a score, but not where or how big,
and it does not change the threat level.

Model file: ir_metal_model.pth or ir_knife_model.pth next to this file (or PEEKIR_IR_AI_MODEL).
It was trained by train_ir_model.py with labels ["no_knife", "knife"]; here "knife" means metal.
Input = the Pi's colourised IR JPEG (/api/ir/frame) -> grayscale -> 160x120 -> 0..1, exactly like
train_ir_model.py / predict_ir.py. If the training images were made a different way, results
will be poor.

Needs: pip install torch pillow numpy   (CPU-only torch is enough)

Command line (no sklearn needed, unlike predict_ir.py):
  python ir_ai_metal.py some_ir_image.png [more.png ...]
  python ir_ai_metal.py --eval knife_dataset      # folders knife|metal / no_knife|no_metal
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POSITIVE = ("metal", "knife")                 # label names that mean "metal present"
_model_lock = threading.Lock()
_cached: dict = {}


def default_model_path() -> Path:
    env = os.getenv("PEEKIR_IR_AI_MODEL", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for name in ("ir_metal_model.pth", "ir_knife_model.pth"):
        if (HERE / name).is_file():
            return HERE / name
    return HERE / "ir_metal_model.pth"


def _build_model():
    from torch import nn

    class ThermalCNN(nn.Module):                # identical to train_ir_model.py
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
            self.classifier = nn.Sequential(
                nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

        def forward(self, x):
            return self.classifier(self.features(x))

    return ThermalCNN()


def load_model(path: str | Path | None = None):
    """(model, labels, size, path). Cached; reloaded if the file changes."""
    import torch

    path = Path(path or default_model_path())
    if not path.is_file():
        raise FileNotFoundError(f"AI model not found: {path} (put ir_knife_model.pth next to the dashboard "
                                "or set PEEKIR_IR_AI_MODEL)")
    key = (str(path), path.stat().st_mtime)
    with _model_lock:
        if _cached.get("key") != key:
            # weights_only=True: loads tensors/lists/strings only, never runs code from the file
            ck = torch.load(path, map_location="cpu", weights_only=True)
            model = _build_model()
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            size = tuple(ck.get("image_size", (160, 120)))
            _cached.update(key=key, model=model, labels=list(ck.get("labels", ["no_knife", "knife"])), size=size)
        return _cached["model"], _cached["labels"], _cached["size"], path


def _pos_index(labels: list[str]) -> int:
    for i, lab in enumerate(labels):
        if lab.lower() in POSITIVE:
            return i
    return len(labels) - 1


def predict_image(pil_image, model_path=None) -> dict:
    """Metal score for one PIL image, preprocessed exactly like train_ir_model.py."""
    import torch

    model, labels, size, path = load_model(model_path)
    gray = pil_image.convert("L").resize(size)
    x = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)[None, None]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].tolist()
    pi = _pos_index(labels)
    score = float(probs[pi])
    metal = score >= 0.5
    return {"metal_detected": metal, "metal_score": round(score, 4),
            "label": "metal" if metal else "no metal", "raw_label": labels[int(np.argmax(probs))],
            "model": path.name}


def predict_jpeg(jpeg_bytes: bytes, model_path=None) -> dict:
    """Dashboard: Pi IR JPEG -> result + annotated PNG (base64)."""
    from PIL import Image, ImageDraw

    pil = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    r = predict_image(pil, model_path)
    big = pil.resize((pil.width * 4, pil.height * 4), Image.NEAREST) if pil.width <= 320 else pil.copy()
    d = ImageDraw.Draw(big)
    pct = r["metal_score"] * 100
    text = f"AI (experimental): {'METAL' if r['metal_detected'] else 'NO METAL'}  {pct:.0f}% metal score"
    d.rectangle([0, 0, big.width, 26], fill=(0, 0, 0))
    d.text((8, 7), text, fill=(255, 107, 107) if r["metal_detected"] else (61, 204, 122))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    r.update(ok=True, annotated_image_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
             summary=f"AI check (experimental): {'METAL' if r['metal_detected'] else 'no metal'} · "
                     f"metal score {pct:.0f}% · {r['model']}",
             gun_detected=False, detections=[])
    return r


def _main(argv: list[str]) -> int:
    from PIL import Image

    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--eval":
        root = Path(argv[1] if len(argv) > 1 else "knife_dataset")
        truth = {"knife": True, "metal": True, "no_knife": False, "no_metal": False, "nometal": False}
        n = ok = 0
        cm = {(t, p): 0 for t in (True, False) for p in (True, False)}
        for folder, is_metal in truth.items():
            for f in sorted((root / folder).glob("*.png")) if (root / folder).is_dir() else []:
                with Image.open(f) as im:
                    p = predict_image(im)["metal_detected"]
                cm[(is_metal, p)] += 1
                n += 1
                ok += p == is_metal
        if not n:
            print(f"No images in {root}/(knife|metal|no_knife|no_metal)/*.png")
            return 1
        print(f"{ok}/{n} correct ({ok / n:.0%})")
        print(f"  metal:    {cm[(True, True)]} found, {cm[(True, False)]} missed")
        print(f"  no metal: {cm[(False, False)]} correct, {cm[(False, True)]} false alarms")
        print("Note: images the model was trained on score too well; judge it on new captures.")
        return 0
    for a in argv:
        with Image.open(a) as im:
            r = predict_image(im)
        print(f"{Path(a).name}: {r['label'].upper()}  metal score {r['metal_score'] * 100:.1f}%  ({r['model']})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
