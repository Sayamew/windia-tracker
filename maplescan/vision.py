from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import cv2
import numpy as np

NAME_RE = re.compile(r"^[A-Za-z0-9]{2,16}$")

KNOWN_JOBS = [
    "Beginner", "Warrior", "Magician", "Bowman", "Thief", "Pirate",
    "Fighter", "Page", "Spearman", "Cleric", "Hunter", "Crossbowman",
    "Assassin", "Bandit", "Brawler", "Gunslinger",
    "Wizard (Fire/Poison)", "Wizard (Ice/Lightning)",
    "Crusader", "White Knight", "Dragon Knight", "Priest", "Ranger", "Sniper",
    "Hermit", "Chief Bandit", "Marauder", "Outlaw",
    "Mage (Fire/Poison)", "Mage (Ice/Lightning)",
]


@dataclass
class NameTag:
    name: str
    confidence: float
    x: int  # tag bounding box in window coords
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2


class OCR:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def read(self, img: np.ndarray) -> list[tuple[str, float]]:
        if img.size == 0:
            return []
        img = cv2.copyMakeBorder(img, 6, 6, 8, 8, cv2.BORDER_REPLICATE)
        big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        result, _ = self.engine(big)
        if not result:
            return []
        return [(str(text).strip(), float(score)) for _box, text, score in result]

    def detect(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        """Whole-frame text detection + reading: [((x, y, w, h), text, confidence), ...]."""
        try:
            result, _ = self.engine(frame, text_score=0.3, box_thresh=0.45, unclip_ratio=1.8)
        except TypeError:
            result, _ = self.engine(frame)
        out = []
        for box, text, score in result or []:
            pts = np.array(box, dtype=np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            out.append(((int(x0), int(y0), int(x1 - x0), int(y1 - y0)), str(text).strip(), float(score)))
        return out

    def read_text(self, img: np.ndarray) -> tuple[str, float]:
        parts = self.read(img)
        if not parts:
            return "", 0.0
        text = "".join(t for t, _ in parts)
        conf = min(s for _, s in parts)
        return text, conf


def norm_name(n: str) -> str:
    """Compare names the way the font blurs them: l / I / 1 look alike, so do O / 0."""
    return n.lower().replace("i", "l").replace("1", "l").replace("0", "o")


def _blank_zones(frame: np.ndarray, zones: list[list[float]]) -> np.ndarray:
    """Grey out UI areas (minimap, quest helper, chat, hotbar) so their text is never read as a player."""
    H, W = frame.shape[:2]
    out = frame.copy()
    for x0, y0, x1, y1 in zones:
        out[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 128
    return out


def _is_yellow(frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """NPC and shop names are drawn in yellow; player names are white."""
    roi = frame[max(0, y):y + h, max(0, x):x + w].astype(np.int16)
    if roi.size == 0:
        return False
    bright = roi[roi.min(axis=2) > 150]
    if len(bright) < 3:
        bright = roi.reshape(-1, 3)[roi.reshape(-1, 3).max(axis=1) > 170]
    if len(bright) < 3:
        return False
    b, g, r = bright[:, 0].mean(), bright[:, 1].mean(), bright[:, 2].mean()
    return (r + g) / 2 - b > 55


def _tag_like(frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """A name tag is light text with a dark outline/backing. Rejects speech bubbles (dark text on a mostly
    white box), clouds and other bright scenery."""
    roi = frame[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = float((gray > 190).mean())
    dark = float((gray < 70).mean())
    return 0.05 <= bright <= 0.50 and dark >= 0.05


def _signature(crop: np.ndarray) -> np.ndarray:
    """Background-independent fingerprint of a tag's text, so an unchanged tag isn't sent to Gemini twice."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray > 190).astype(np.float32)
    return cv2.resize(mask, (24, 8), interpolation=cv2.INTER_AREA)


class TagCache:
    """Remembers what Gemini read for a given tag image for a while (default 10 minutes)."""

    def __init__(self, ttl: float = 600.0) -> None:
        self.ttl = ttl
        self.items: list[tuple[np.ndarray, str | None, float]] = []

    def get(self, sig: np.ndarray):
        import time
        now = time.time()
        self.items = [i for i in self.items if now - i[2] < self.ttl]
        for s, val, _t in self.items:
            if float(np.abs(s - sig).mean()) < 0.06:
                return True, val
        return False, None

    def put(self, sig: np.ndarray, val: str | None) -> None:
        import time
        self.items.append((sig, val, time.time()))
        self.items = self.items[-300:]


def read_name_tags(frame: np.ndarray, ocr: OCR, cfg, namer=None, cache: "TagCache | None" = None) -> list[NameTag]:
    """Finds every player name on screen. The local OCR locates the text boxes; when a Gemini `namer` is
    given, each box is cropped and Gemini reads the actual name (the local reader often drops letters).
    Pets and NPCs are mostly filtered out here (spaces, yellow text); any that get through are dropped
    later when double-clicking them doesn't open a player's window."""
    H = frame.shape[0]
    s = H / 1080.0
    own = norm_name(str(cfg.get("own_character", "")))
    zones = cfg.get("ignore_zones", [])
    lines = ocr.detect(_blank_zones(frame, zones) if zones else frame)

    cands: list[tuple[tuple[int, int, int, int], str, float]] = []
    for (x, y, w, h), text, conf in lines:
        if not (7 * s <= h <= 28 * s and w >= 10 * s and w <= 190 * s):
            continue
        raw = text.strip()
        if namer is None:
            if " " in raw or not NAME_RE.match(raw) or len(raw) < 3 or conf < cfg["min_name_confidence"]:
                continue
        if _is_yellow(frame, x, y, w, h) or not _tag_like(frame, x, y, w, h):
            continue
        cands.append(((x, y, w, h), raw, conf))

    names: list[str | None] = []
    if namer is not None:
        cands = cands[: int(cfg.get("max_tag_crops", 30))]
        crops, sigs, todo = [], [], []
        resolved: list = [None] * len(cands)
        have = [False] * len(cands)
        for i, ((x, y, w, h), _t, _c) in enumerate(cands):
            crop = frame[max(0, y - 2):y + h + 2, max(0, x - 3):x + w + 3]
            sig = _signature(crop)
            sigs.append(sig)
            hit, val = cache.get(sig) if cache else (False, None)
            if hit:
                resolved[i], have[i] = val, True
            else:
                todo.append(i)
                crops.append(crop)
        if crops:
            got = namer(crops)
            for j, i in enumerate(todo):
                if got is not None:
                    resolved[i] = got[j]
                    if cache:
                        cache.put(sigs[i], got[j])
                else:  # Gemini unavailable: fall back to the local reading
                    raw = cands[i][1]
                    resolved[i] = raw if NAME_RE.match(raw) else None
        names = resolved
    else:
        names = [raw for _b, raw, _c in cands]

    tags: list[NameTag] = []
    for ((x, y, w, h), _t, conf), name in zip(cands, names):
        if not name:
            continue
        name = name.strip()
        if " " in name or not NAME_RE.match(name) or len(name) < 3 or norm_name(name) == own:
            continue
        tags.append(NameTag(name, 1.0 if namer is not None else conf, x, y, w, h))

    best: dict[str, NameTag] = {}
    for t in tags:
        k = norm_name(t.name)
        if k not in best or t.confidence > best[k].confidence:
            best[k] = t
    return list(best.values())


def crop_avatar(frame: np.ndarray, tag: NameTag, box: dict) -> bytes | None:
    """Crop the character sprite (above its name tag) and return PNG bytes."""
    x0 = tag.center_x + box["dx"]
    y0 = tag.y + box["dy"]
    x1, y1 = x0 + box["w"], y0 + box["h"]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    ok, buf = cv2.imencode(".png", frame[y0:y1, x0:x1])
    return buf.tobytes() if ok else None


def parse_level(text: str) -> int | None:
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    level = int(digits)
    return level if 1 <= level <= 300 else None


_LETTERS = re.compile(r"[^a-z]")
_SIMPLE = {_LETTERS.sub("", j.lower()): j for j in KNOWN_JOBS if "(" not in j}
_SIMPLE.update({
    "crossbowmen": "Crossbowman", "crossbow": "Crossbowman", "xbowman": "Crossbowman",
    "dk": "Dragon Knight", "wk": "White Knight", "dragonknights": "Dragon Knight",
    "gunslingers": "Gunslinger", "bowmen": "Bowman", "pages": "Page", "clerics": "Cleric",
    "fighters": "Fighter", "spearmen": "Spearman", "magicians": "Magician", "pirates": "Pirate",
})


WIZARD_MIN_LEVEL_3RD = 70  # 3rd job (Mage) starts here; 2nd job (Wizard) is below it


def match_job(text: str, level: int | None = None) -> str | None:
    """Maps whatever the JOB row (or a model) says to one of our job names, or None.
    Handles variants ('F/P Wizard', 'Fire/Poison Mage', 'Crossbow man') and TRUNCATED text: the info
    window cuts long names off ('I/L Mag', 'F/P Wiz', 'Ice/Lightn', 'Crossbowma', 'Dragon Kni'). When the
    text shows the element but not whether it is a Wizard or a Mage, the character's level decides."""
    raw = text.strip()
    if not raw:
        return None
    low = raw.lower()
    letters = _LETTERS.sub("", low)
    if not letters or "archmage" in letters or "bishop" in letters:
        return None  # 4th job isn't tracked yet
    tokens = [t for t in re.split(r"[^a-z]+", low) if t]

    # ---- Fire/Poison and Ice/Lightning wizards and mages (the long, often cut-off names)
    fp_words = ("fire", "poison")
    il_words = ("ice", "lightning")
    is_fp = any(t in ("f", "p", "fp") for t in tokens) or any(len(t) >= 3 and w.startswith(t) for t in tokens for w in fp_words)
    is_il = any(t in ("i", "l", "il") for t in tokens) or any(len(t) >= 3 and w.startswith(t) for t in tokens for w in il_words)
    if is_fp != is_il or (is_fp and is_il):
        has_class_word = any(
            (len(t) >= 2 and ("wizard".startswith(t) or "mage".startswith(t) or t in ("wizard", "mage")))
            for t in tokens
        ) or "wizard" in letters or "mage" in letters
        element_only = all(t in ("f", "p", "fp", "i", "l", "il") or any(len(t) >= 3 and w.startswith(t) for w in fp_words + il_words)
                           for t in tokens)
        if has_class_word or element_only:
            if is_fp and is_il:  # ambiguous 'i/l' vs 'f/p' read: give up rather than guess
                return None
            tier = None
            last = tokens[-1] if tokens else ""
            if "wizard" in letters or ("wizard".startswith(last) and last[:1] == "w"):
                tier = "Wizard"
            elif "mage" in letters or ("mage".startswith(last) and last[:1] == "m" and last != "m" or last == "ma"):
                tier = "Mage"
            if tier is None:  # cut off before the class word: use the level
                if level is None:
                    return None
                tier = "Mage" if level >= WIZARD_MIN_LEVEL_3RD else "Wizard"
            return f"{tier} ({'Fire/Poison' if is_fp else 'Ice/Lightning'})"

    if letters in _SIMPLE:
        return _SIMPLE[letters]
    # truncated plain jobs: a unique prefix of a job name ('Crossbowma', 'Dragon Kni', 'Chief Band')
    if len(letters) >= 4:
        pref = {v for k, v in _SIMPLE.items() if k.startswith(letters)}
        if len(pref) == 1:
            return next(iter(pref))
    scored = sorted(((difflib.SequenceMatcher(None, letters, k).ratio(), v) for k, v in _SIMPLE.items()), reverse=True)
    best = scored[0]
    rival = next((sc for sc, v in scored if v != best[1]), 0.0)
    if best[0] >= 0.82 and best[0] - rival > 0.08:
        return best[1]
    return None


def parse_fame(text: str) -> int | None:
    """Fame can be negative; keep a leading minus sign if OCR found one."""
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    m = re.search(r"-?\s*\d[\d,]*", text)
    if not m:
        return None
    try:
        value = int(re.sub(r"[\s,]", "", m.group()))
    except ValueError:
        return None
    return value if -100000 <= value <= 1000000 else None
