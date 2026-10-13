"""Reads the in-game CHARACTER INFO window (the one with LEVEL / JOB / FAME rows).

Layout this is built for (see the example screenshots):
  - left panel: character sprite with a grey name plate underneath, a leaf icon in its top-right corner
  - right of it: pink label cells LEVEL, JOB, FAME, GUILD, each followed by a beige value cell
  - below (optional): a pet block with its own pink TYPE / LEVEL / CLOSENESS / FULLNESS cells (ignored)

Instead of OCR-ing the whole screen, we find the pink label cells by colour (scale-independent), then
derive every region from their size, so it works wherever the window is and at any UI scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .vision import NAME_RE, OCR, match_job, parse_fame, parse_level

# OpenCV HSV (H 0-179). The label pink measures about H=167, S=110-160, V=170-205 in both example screenshots.
PINK_LO = (158, 60, 100)
PINK_HI = (178, 255, 255)

# Region geometry, in multiples of the label cell width (cw). Measured on the example screenshots.
VALUE_PAD = 0.03
VALUE_W = {"level": 1.55, "job": 1.60, "fame": 1.20}  # fame stops before the little up/down arrow buttons
PLATE_TOP = 1.04     # name plate top, below the LEVEL cell top
PLATE_H = 0.27
PANEL_W = 1.72       # sprite panel width to the left of the labels
PANEL_GAP = 0.06
SPRITE_TOP = 1.28    # sprite crop top, above the name plate
SPRITE_HALF_W = 0.62
LEAF = {"x": 0.56, "y_top": 1.42, "y_bottom": 0.80}  # leaf icon box relative to panel right / plate top


@dataclass
class Layout:
    x0: int   # left edge of the label column
    cw: int   # label cell width
    ch: int   # label cell height
    level_y: int
    job_y: int
    fame_y: int


@dataclass
class InfoReading:
    name: str | None
    level: int | None
    job: str | None
    fame: int | None
    level_conf: float
    avatar_png: bytes | None


def find_label_cells(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PINK_LO, PINK_HI)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if 40 <= w <= 500 and 9 <= h <= 70 and 2.2 <= w / h <= 9.5 and area >= 0.40 * w * h:
            cells.append((x, y, w, h))
    return cells


def locate(frame: np.ndarray) -> Layout | None:
    """Find the LEVEL / JOB / FAME label column of the topmost (= character, not pet) block."""
    cells = find_label_cells(frame)
    stacks: list[list[tuple[int, int, int, int]]] = []
    for c in sorted(cells, key=lambda c: c[1]):
        for s in stacks:
            x, y, w, h = s[-1]
            if abs(c[0] - x) <= max(4, 0.04 * w) and abs(c[2] - w) <= max(4, 0.04 * w) and 0 < c[1] - y <= 2.2 * h:
                s.append(c)
                break
        else:
            stacks.append([c])
    # CHARACTER INFO has exactly 4 rows (LEVEL, JOB, FAME, GUILD; GUILD may be too faint to detect).
    # The CHARACTER STAT window has 5+ equal rows (LEVEL, HP, MP, EXP, FAME...), so it is rejected here.
    stacks = [s for s in stacks if 3 <= len(s) <= 4]
    if not stacks:
        return None
    s = min(stacks, key=lambda s: s[0][1])  # topmost stack = character block
    (x, ly, w, h), (_, jy, _, _), (_, fy, _, _) = s[0], s[1], s[2]
    return Layout(x0=x, cw=w, ch=h, level_y=ly, job_y=jy, fame_y=fy)


def close_button(L: Layout) -> tuple[int, int]:
    """Centre of the window's blue X button (top-right of the title bar), from the label geometry."""
    return int(L.x0 + 2.55 * L.cw), int(L.level_y - 0.61 * L.cw)


def _clip(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    H, W = frame.shape[:2]
    xa, ya = max(0, int(round(x0))), max(0, int(round(y0)))
    xb, yb = min(W, int(round(x1))), min(H, int(round(y1)))
    return frame[ya:yb, xa:xb]


def regions(frame: np.ndarray, L: Layout) -> dict[str, tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) of every region we read, in frame coordinates."""
    right = L.x0 + L.cw
    out = {}
    for key, y in (("level", L.level_y), ("job", L.job_y), ("fame", L.fame_y)):
        out[key] = (right + VALUE_PAD * L.cw, y + 1, right + VALUE_W[key] * L.cw, y + L.ch - 1)
    panel_right = L.x0 - PANEL_GAP * L.cw
    plate_y0 = L.level_y + PLATE_TOP * L.cw
    out["name"] = (panel_right - PANEL_W * L.cw, plate_y0, panel_right, plate_y0 + PLATE_H * L.cw)
    cx = (out["name"][0] + out["name"][2]) / 2
    out["sprite"] = (max(panel_right - PANEL_W * L.cw + 3, cx - SPRITE_HALF_W * L.cw),
                     plate_y0 - SPRITE_TOP * L.cw,
                     min(panel_right - 2, cx + SPRITE_HALF_W * L.cw),
                     plate_y0 - 2)
    return out


def crop_sprite(frame: np.ndarray, L: Layout, max_side: int = 192) -> bytes | None:
    r = regions(frame, L)
    x0, y0, x1, y1 = r["sprite"]
    crop = _clip(frame, x0, y0, x1, y1).copy()
    if crop.shape[0] < 24 or crop.shape[1] < 24:
        return None

    # Paint over the leaf icon (top-right of the panel) with the panel's background colour.
    panel_right = L.x0 - PANEL_GAP * L.cw
    plate_y0 = L.level_y + PLATE_TOP * L.cw
    lx0 = int(panel_right - LEAF["x"] * L.cw - x0)
    ly0 = int(plate_y0 - LEAF["y_top"] * L.cw - y0)
    ly1 = int(plate_y0 - LEAF["y_bottom"] * L.cw - y0)
    bg = np.median(crop[:, :3].reshape(-1, 3), axis=0).astype(np.uint8)
    crop[max(0, ly0):max(0, ly1), max(0, lx0):] = bg

    h, w = crop.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else None



def _bbox(mask: np.ndarray, pad: int, shape: tuple[int, int]):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad))


def tight_value(crop: np.ndarray) -> np.ndarray:
    """Dark text on the beige value cell: crop to the text so the OCR sees only the characters."""
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    box = _bbox(gray < 120, 3, gray.shape)
    if box is None:
        return crop
    x0, y0, x1, y1 = box
    return crop[y0:y1, x0:x1]


def tight_name(crop: np.ndarray) -> np.ndarray:
    """White text on the mid-grey name plate: find the plate, then crop to the bright text inside it."""
    if crop.size == 0:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    plate = ((gray > 70) & (gray < 170) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(plate, connectivity=8)
    if n <= 1:
        return crop
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    px, py, pw, ph = (int(v) for v in stats[i, :4])
    if pw < 20 or ph < 8:
        return crop
    inner = gray[py + 4: py + ph - 4, px + 6: px + pw - 6]
    box = _bbox(inner > 190, 2, inner.shape)
    if box is None:
        return crop[py:py + ph, px:px + pw]
    x0, y0, x1, y1 = box
    return crop[py + 4 + y0: py + 4 + y1, px + 6 + x0: px + 6 + x1]


def _variants(crop: np.ndarray, dark_text: bool = True) -> list[np.ndarray]:
    """Different renderings of the same crop. Reading all of them and requiring agreement avoids
    saving a misread number."""
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (otsu == 0).mean() > 0.5:  # make the (minority) text dark on light
        otsu = cv2.bitwise_not(otsu)
    if not dark_text:
        gray = cv2.bitwise_not(gray)
    sharp = cv2.filter2D(crop, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    return [crop, cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR), sharp]


def _majority(values: list):
    """Value read by at least 2 of the variants, else None."""
    vals = [v for v in values if v is not None]
    for v in vals:
        if vals.count(v) >= 2:
            return v
    return None


def read_info_window(frame: np.ndarray, ocr: OCR) -> InfoReading | None:
    """Returns None if no CHARACTER INFO window is visible. A field that can't be read consistently
    comes back as None (and the caller must then not trust the reading)."""
    L = locate(frame)
    if L is None:
        return None
    r = regions(frame, L)

    def crop(key: str) -> np.ndarray:
        x0, y0, x1, y1 = r[key]
        raw = _clip(frame, x0, y0, x1, y1)
        return tight_name(raw) if key == "name" else tight_value(raw)

    def reads(key: str, dark_text: bool = True) -> list[tuple[str, float]]:
        return [ocr.read_text(v) for v in _variants(crop(key), dark_text)]

    # Name: white text on the grey plate. Take the most common cleaned reading.
    names = []
    for t, _c in reads("name", dark_text=False):
        t = t.replace(" ", "")
        names.append(t if NAME_RE.match(t) else None)
    name = _majority(names) or next((n for n in names[:1] if n), None)

    level_reads = reads("level")
    level = _majority([parse_level(t) for t, _ in level_reads])
    level_conf = max((c for _t, c in level_reads), default=0.0)

    job_reads = [t for t, _ in reads("job")]
    job = _majority([match_job(t, level) for t in job_reads])
    fame = _majority([parse_fame(t) for t, _ in reads("fame")])

    return InfoReading(
        name=name,
        level=level,
        job=job,
        fame=fame,
        level_conf=level_conf,
        avatar_png=crop_sprite(frame, L),
    )


def panel_crop(frame: np.ndarray) -> np.ndarray | None:
    """The part of the CHARACTER INFO window that matters (sprite, name plate, LEVEL/JOB/FAME rows),
    cut so a model can't confuse it with the pet block below or the item list beside it."""
    L = locate(frame)
    if L is None:
        return None
    r = regions(frame, L)
    x0 = r["name"][0] - 4
    y0 = L.level_y - 0.2 * L.cw
    x1 = r["fame"][2] + 0.4 * L.cw
    y1 = r["name"][3] + 4
    crop = _clip(frame, x0, y0, x1, y1)
    return crop if crop.size else None


def plate_crop(frame: np.ndarray) -> np.ndarray | None:
    """Just the grey name plate (name text), for a larger, clearer second look."""
    L = locate(frame)
    if L is None:
        return None
    x0, y0, x1, y1 = regions(frame, L)["name"]
    raw = _clip(frame, x0, y0, x1, y1)
    return raw if raw.size else None
