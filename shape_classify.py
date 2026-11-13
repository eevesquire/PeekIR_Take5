"""
Shape of a detected metal piece -> "gun" / "knife" / "suspicious object" (no training needed).

Used by ir_metal_size.py and mmwave_metal_size.py on every piece they outline. Works at any
rotation, from measurements of the piece's outline in real units (cm), or in IR pixels when
the IR camera is not calibrated:

  length, width  smallest rotated rectangle around the piece (so a diagonal knife is still long)
  fill           piece area / that rectangle's area (solid plate ~0.9-1.0, L-shape ~0.3-0.6)
  excess         length - width. Blur from the sensors adds about the same to both, so this stays
                 close to the true value (knife 5x1 in: ~10 cm, small 3x1 in rectangle: ~5 cm)
  corner         best share of the piece lying along two ADJACENT sides of a box around it, over
                 all box angles (an L lies entirely along two sides; a U, ring or blob does not)

  L-shape  (gun_min_fill <= fill <= gun_max_fill, corner >= gun_min_corner, long side >= gun_min_length_cm)  -> gun
  solid, long and thin  (excess >= knife_min_excess_cm, length/width >= knife_min_elongation,
                         width <= knife_max_width_cm)                                          -> knife
                        (IR needs calibration for this: without cm, only very thin pieces count)
  solid, length/width <= square_max_elongation                                                -> square
  any other solid piece                                                                        -> rectangle
  anything else                                                                                -> irregular
Weapons (gun / knife) raise the threat level to at least `weapon_level` (default HIGH).
Everything that is not a weapon is shown as "suspicious object" (the exact shape is still saved).

Defaults are set for the demo props: L panel 8 x 5 in with 1 in arms (20.3 x 12.7 cm, 2.5 cm wide),
small rectangle 3 x 1 in (7.6 x 2.5 cm), "knife" = two small rectangles taped end to end
(about 12.7-15 x 2.5 cm).

Settings: ir_metal_state/shape_settings.json (shared by IR and mmWave).
  python shape_classify.py settings                      # show
  python shape_classify.py settings --knife-min-excess-cm 8 --weapon-level MEDIUM
  python shape_classify.py settings --off                # turn shape labels off (size only)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]
DISPLAY = {"L": "gun", "long_rect": "knife"}          # everything else -> "suspicious object"
SHAPE_NAMES = {"L": "L-shape", "long_rect": "long rectangle", "rect": "rectangle", "square": "square",
               "irregular": "irregular"}


@dataclass
class ShapeSettings:
    enabled: bool = True
    weapon_level: str = "HIGH"             # a gun/knife raises the threat level to at least this
    knife_min_excess_cm: float = 7.5       # length - width: knife 5x1 in ~10 cm, 6x1 in ~13 cm; small
                                           # 3x1 in rectangle ~5 cm (blur adds equally to both sides)
    knife_min_elongation: float = 2.0      # length / width (blur widens thin pieces: a 5x1 in knife can
                                           # measure only ~2.1-2.7:1 in IR)
    knife_max_width_cm: float = 9.0        # wider than this is a plate, not a knife
    knife_uncal_min_elongation: float = 4.5  # uncalibrated IR (no cm): length/width alone must be at least this
    solid_min_fill: float = 0.58           # "solid" piece: fills at least this much of its rectangle
                                           # (thin diagonal pieces look jagged at low resolution)
    square_max_elongation: float = 1.35
    gun_max_fill: float = 0.62             # L panel fills ~0.3 of its 8x5 in rectangle (more when blurred)
    gun_min_fill: float = 0.15             # below this it's scattered bits, not a solid L
    gun_min_corner: float = 0.9            # share of the piece along two adjacent sides (L ~0.95-1.0,
                                           # U-shape / ring ~0.6-0.88)
    gun_min_length_cm: float = 10.0        # long side of the L (20.3 cm)
    min_width_cells: float = 2.0           # thinner than this many image cells: shape is a guess (note)


def _state_dir(state_dir=None) -> Path:
    d = Path(state_dir or os.getenv("PEEKIR_METAL_STATE", str(Path(__file__).resolve().parent / "ir_metal_state")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(state_dir=None) -> ShapeSettings:
    p = _state_dir(state_dir) / "shape_settings.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return ShapeSettings(**{k: v for k, v in data.items() if k in ShapeSettings.__dataclass_fields__})
        except Exception:
            pass
    return ShapeSettings()


def save_settings(s: ShapeSettings, state_dir=None) -> Path:
    p = _state_dir(state_dir) / "shape_settings.json"
    p.write_text(json.dumps(asdict(s), indent=2))
    return p


def _corner_share(u: np.ndarray, v: np.ndarray, W: float, H: float, band: float) -> float:
    """Largest share of points lying within `band` of two adjacent sides of the W x H box."""
    best = 0.0
    for su in (-1, 1):
        for sv in (-1, 1):
            near = (np.abs(u - su * W / 2) <= band) | (np.abs(v - sv * H / 2) <= band)
            best = max(best, float(near.mean()))
    return best


def _best_l_fit(xy: np.ndarray, area: float, cell_w: float, cell_h: float, step_deg: float = 3.0):
    """Try box angles 0-90 deg; return (best corner share, box fill, box long side, arm width) at that angle."""
    best = (0.0, 1.0, 0.0, 0.0)
    c = xy.mean(axis=0)
    d0 = xy - c
    pad = max(cell_w, cell_h) / 2.0
    for a in np.deg2rad(np.arange(0.0, 90.0, step_deg)):
        e1, e2 = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        u, v = d0 @ e1, d0 @ e2
        umin, umax, vmin, vmax = u.min() - pad, u.max() + pad, v.min() - pad, v.max() + pad
        W, H = umax - umin, vmax - vmin
        box_fill = min(1.0, area / max(W * H, 1e-9))
        disc = max((W + H) ** 2 - 4 * area, 0.0)
        t = max(((W + H) - np.sqrt(disc)) / 2.0, min(cell_w, cell_h))
        # arm width of an L that fits this box: t^2 - (W+H) t + area = 0
        uc, vc = u - (umin + umax) / 2, v - (vmin + vmax) / 2
        share = _corner_share(uc, vc, W, H, band=max(1.3 * t, 1.5 * max(cell_w, cell_h)))
        if share > best[0] or (share == best[0] and box_fill < best[1]):
            best = (share, box_fill, max(W, H), t)
    return best


def classify_points(xy: np.ndarray, cell_w: float, cell_h: float, in_cm: bool,
                    s: ShapeSettings | None = None) -> dict | None:
    """
    xy: (N, 2) centres of the piece's image cells, in cm (in_cm=True) or in IR pixels (False).
    cell_w/cell_h: cell size in the same units. Returns the shape dict (see module doc).
    """
    import cv2
    s = s or load_settings()
    if not s.enabled or xy is None or len(xy) < 3:
        return None
    xy = np.asarray(xy, np.float64)
    hw, hh = cell_w / 2.0, cell_h / 2.0
    corners = np.concatenate([xy + [dx, dy] for dx in (-hw, hw) for dy in (-hh, hh)]).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(corners)
    W, H = max(float(rw), 1e-6), max(float(rh), 1e-6)
    area = len(xy) * cell_w * cell_h
    length, width = max(W, H), min(W, H)
    elong = length / width
    fill = min(1.0, area / (W * H))
    hull = cv2.convexHull(corners)
    solidity = min(1.0, area / max(float(cv2.contourArea(hull)), 1e-6))
    corner, l_fill, l_len, l_t = _best_l_fit(xy, area, cell_w, cell_h)
    thin = width < s.min_width_cells * max(cell_w, cell_h)

    excess = length - width
    if s.gun_min_fill <= l_fill <= s.gun_max_fill and corner >= s.gun_min_corner and (not in_cm or l_len >= s.gun_min_length_cm) \
            and l_t >= 1.5 * min(cell_w, cell_h):
        shape = "L"
    elif fill >= s.solid_min_fill:
        long_enough = (excess >= s.knife_min_excess_cm and elong >= s.knife_min_elongation
                       and width <= s.knife_max_width_cm) if in_cm \
            else elong >= s.knife_uncal_min_elongation
        if long_enough:
            shape = "long_rect"
        elif elong <= s.square_max_elongation:
            shape = "square"
        else:
            shape = "rect"
    else:
        shape = "irregular"
    unit = "cm" if in_cm else "px"
    return {
        "shape": shape,
        "shape_name": SHAPE_NAMES[shape],
        "weapon": DISPLAY.get(shape),                      # "gun" / "knife" / None
        "display": DISPLAY.get(shape, "suspicious object"),
        "length": round(length, 1), "width": round(width, 1), "unit": unit,
        "elongation": round(elong, 2), "excess": round(excess, 1), "fill": round(fill, 2), "solidity": round(solidity, 2),
        "corner": round(corner, 2), "arm_width": round(float(l_t), 1),
        "uncertain": bool(thin),
    }


def classify_mask(mask: np.ndarray, cell_w: float, cell_h: float, in_cm: bool, origin=(0.0, 0.0),
                  s: ShapeSettings | None = None) -> dict | None:
    """Same as classify_points for a boolean (rows, cols) mask with the given cell size/origin."""
    rr, cc = np.nonzero(mask)
    xy = np.stack([origin[0] + cc * cell_w, origin[1] + rr * cell_h], axis=1)
    return classify_points(xy, cell_w, cell_h, in_cm, s)


def raise_level(level: str, shapes: list[dict | None], s: ShapeSettings | None = None) -> str:
    """Threat level after weapons: at least s.weapon_level if any piece is a gun or knife."""
    s = s or load_settings()
    if not s.enabled or not any(sh and sh.get("weapon") for sh in shapes):
        return level
    return max(level, s.weapon_level, key=LEVELS.index)


def weapons_of(shapes: list[dict | None]) -> list[str]:
    """['gun', 'knife', ...] in order of first appearance, no duplicates."""
    out: list[str] = []
    for sh in shapes:
        if sh and sh.get("weapon") and sh["weapon"] not in out:
            out.append(sh["weapon"])
    return out


def describe(shape: dict | None) -> str:
    if not shape:
        return ""
    txt = shape["display"]
    if shape.get("uncertain"):
        txt += "?"
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("settings", help="show / change shape settings")
    p.add_argument("--on", action="store_true", help="turn shape labels on")
    p.add_argument("--off", action="store_true", help="turn shape labels off (threat from size only)")
    p.add_argument("--weapon-level", choices=["LOW", "MEDIUM", "HIGH"])
    for f in ("knife-min-excess-cm", "knife-min-elongation", "knife-max-width-cm", "knife-uncal-min-elongation",
              "solid-min-fill", "square-max-elongation", "gun-max-fill", "gun-min-fill", "gun-min-corner",
              "gun-min-length-cm"):
        p.add_argument("--" + f, type=float)
    p.add_argument("--reset", action="store_true", help="back to the defaults")
    args = ap.parse_args()
    s = ShapeSettings() if args.reset else load_settings(args.state_dir)
    if args.on:
        s.enabled = True
    if args.off:
        s.enabled = False
    if args.weapon_level:
        s.weapon_level = args.weapon_level
    for k in ("knife_min_excess_cm", "knife_min_elongation", "knife_max_width_cm", "knife_uncal_min_elongation",
              "solid_min_fill", "square_max_elongation", "gun_max_fill", "gun_min_fill", "gun_min_corner",
              "gun_min_length_cm"):
        v = getattr(args, k)
        if v is not None:
            setattr(s, k, v)
    print("saved to", save_settings(s, args.state_dir))
    print(json.dumps(asdict(s), indent=2))


if __name__ == "__main__":
    main()
