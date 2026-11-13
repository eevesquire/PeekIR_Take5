"""
Shared IR preprocessing. Imported by train_ir.py, predict_ir.py and import_legacy_pngs.py,
so training and inference always prepare frames the same way.

Frame formats (stored as `ir_format` in each clip's JSON):
  y16      raw 16-bit Lepton counts (centikelvin on a radiometric Lepton 3.5), from pi_camera_server
  bgr8     8-bit colorized frames from pi_camera_server with IR_RAW_Y16=0
  legacy8  8-bit values recovered from the old knife/no_knife PNGs (see decode_legacy_png)

Every format ends up as a float frame in [0, 1] via to_unit(), then as a model tensor via
model_input(). "percentile" normalization works for every format; "kelvin" only for y16.
"""
from __future__ import annotations

import numpy as np

PREPROCESS_VERSION = 1
RAW_FORMATS = {"y16"}
EIGHT_BIT_FORMATS = {"bgr8", "gray8", "legacy8"}
IMAGENET_MEAN, IMAGENET_STD = 0.45, 0.225


# --------------------------------------------------------------------------- normalization

def _to_gray(frames: np.ndarray) -> np.ndarray:
    """(N,H,W) or (N,H,W,3 BGR) -> float32 (N,H,W)."""
    x = frames.astype(np.float32)
    if x.ndim == 4:
        x = x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
    return x


def to_unit(frames: np.ndarray, ir_format: str, mode: str = "percentile",
            t_lo: float = 15.0, t_hi: float = 40.0) -> np.ndarray:
    """Raw frames (N,H,W[,3]) -> float32 (N,H,W) roughly in [0,1]."""
    if frames.ndim == 2 or (frames.ndim == 3 and frames.shape[-1] == 3 and ir_format == "bgr8"):
        frames = frames[None]
    x = _to_gray(frames)
    if mode == "kelvin":
        if ir_format not in RAW_FORMATS:
            raise ValueError(f"kelvin normalization needs raw y16 frames, got {ir_format!r}")
        c = x / 100.0 - 273.15
        return np.clip((c - t_lo) / (t_hi - t_lo), 0.0, 1.0)
    if mode != "percentile":
        raise ValueError(f"unknown normalization {mode!r}")
    lo = np.percentile(x, 1, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, 99, axis=(1, 2), keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)


def model_input(unit_frames: np.ndarray, size: int):
    """float (N,H,W) in [0,1] -> torch float tensor (N,1,size,size), standardized."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(unit_frames, dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    x = F.interpolate(x[:, None], size=(size, size), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


# --------------------------------------------------------------------------- legacy PNGs

_INFERNO_LUT = None


def _inferno_lut() -> np.ndarray:
    global _INFERNO_LUT
    if _INFERNO_LUT is None:
        import cv2
        _INFERNO_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                         cv2.COLORMAP_INFERNO).reshape(256, 3).astype(np.int32)
    return _INFERNO_LUT


def decode_legacy_png(path, expected_shape=(120, 160)) -> tuple[np.ndarray | None, dict]:
    """
    Recover an 8-bit thermal value image from a colorized PNG made by the old scripts.

    Returns (values uint8 HxW or None, info). info["reject"] is set when the image should
    not be used:
      wrong_size     not a 160x120 frame (e.g. matplotlib figures from the MLX90640 script)
      text_overlay   labels/text burned into the frame (live_capture.py)
      unreadable
    info["palette"]: "inferno" (OpenCV inferno, inverted exactly), "board" (the PureThermal's
    own palette, decoded by luminance, which rises monotonically along it), or "gray".
    """
    import cv2
    info: dict = {"reject": None, "palette": None}
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        info["reject"] = "unreadable"
        return None, info
    if img.shape[:2] != tuple(expected_shape):
        info["reject"] = "wrong_size"
        info["shape"] = list(img.shape[:2])
        return None, info

    b, g, r = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    if np.abs(b - g).max() <= 2 and np.abs(g - r).max() <= 2:
        info["palette"] = "gray"
        return img[..., 1].copy(), info

    lut = _inferno_lut()
    px = img.reshape(-1, 1, 3).astype(np.int32)
    d = ((px - lut[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    err = np.sqrt(d[np.arange(len(idx)), idx])
    if float(np.median(err)) < 4.0:
        info["palette"] = "inferno"
        # Pixels far from every inferno color are text or graphics drawn on the frame.
        off = float((err > 25).mean())
        info["off_palette_frac"] = round(off, 5)
        if off > 0.002:
            info["reject"] = "text_overlay"
        return idx.reshape(img.shape[:2]).astype(np.uint8), info

    info["palette"] = "board"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # White/near-white text on the board palette would sit far above the brightest real pixels
    # in a tiny cluster; flag frames whose brightest 0.1% are pure white with sharp edges.
    white = (img.min(axis=2) >= 250)
    if white.mean() > 0.002:
        edges = cv2.Laplacian(white.astype(np.uint8) * 255, cv2.CV_16S)
        if np.abs(edges).mean() > 2.0:
            info["reject"] = "text_overlay"
    return gray, info
