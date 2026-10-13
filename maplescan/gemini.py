"""Gemini 2.5 Flash readers: name-tag crops (batched, one request) and the cropped CHARACTER INFO panel."""
from __future__ import annotations

import base64
import json
import logging
import re

import cv2
import numpy as np
import requests

from .vision import NAME_RE, match_job

log = logging.getLogger("gemini")

from .vision import KNOWN_JOBS

PROMPT = """You are given up to two images from the MapleStory CHARACTER INFO window:
1. the panel: the character sprite with its grey name plate on the left, and the rows LEVEL, JOB, FAME, GUILD on the right
2. (optional) a larger crop of just the grey name plate.
Return a JSON object: {"name": string|null, "level": integer|null, "job_text": string|null, "job": string|null, "fame": integer|null}
- name: the text on the grey name plate. Read it letter by letter. Take extra care with capital I versus lowercase l,
  the digit 1, and O versus 0; judge by the glyph shapes. Names are one word, letters and digits only, 4-12 characters.
- level: the number in the LEVEL row
- job_text: the JOB row text exactly as written
- job: the SAME text mapped to exactly one of these valid values, or null if none fits:
  __JOBS__
  (e.g. "F/P Wizard" -> "Wizard (Fire/Poison)", "I/L Mage" -> "Mage (Ice/Lightning)"). Do not pick a different job
  that merely looks similar; Cleric and Priest, Page and White Knight, Hunter and Ranger are different jobs.
- fame: the number in the FAME row (it can be negative, e.g. -7)
Ignore the leaf icon and anything else. Use null for anything you cannot read clearly. Do not guess."""
PROMPT = PROMPT.replace("__JOBS__", " | ".join(KNOWN_JOBS))


def _png_part(img: np.ndarray, scale: float) -> dict | None:
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", big)
    if not ok:
        return None
    return {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.tobytes()).decode()}}


def read_panel(panel_bgr: np.ndarray, api_key: str, model: str = "gemini-2.5-flash",
               plate_bgr: np.ndarray | None = None) -> dict | None:
    parts = [_png_part(panel_bgr, 3)]
    if plate_bgr is not None and plate_bgr.size:
        parts.append(_png_part(plate_bgr, 8))
    parts = [p for p in parts if p]
    if not parts:
        return None
    body = {
        "contents": [{"parts": parts + [{"text": PROMPT}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{re.sub(r'[^A-Za-z0-9._-]', '', model)}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            data=json.dumps(body), timeout=30,
        )
        r.raise_for_status()
        text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        d = json.loads(text)
    except Exception as exc:  # network, quota, bad JSON...
        log.warning("Gemini fallback failed: %s", exc)
        return None

    name = d.get("name").replace(" ", "") if isinstance(d.get("name"), str) else None
    level = d.get("level") if isinstance(d.get("level"), int) and 1 <= d["level"] <= 300 else None
    fame = d.get("fame") if isinstance(d.get("fame"), int) and -100000 <= d["fame"] <= 1000000 else None
    # Prefer the model's mapped job, then map the raw row text ourselves; both go through match_job.
    job = None
    for k in ("job", "job_text"):
        if isinstance(d.get(k), str):
            job = match_job(d[k])
            if job:
                break
    return {"name": name if name and NAME_RE.match(name) else None, "level": level, "job": job, "fame": fame,
            "job_text": d.get("job_text") if isinstance(d.get("job_text"), str) else None}


NAMES_PROMPT = """Each image is a tiny crop of one in-game text label from MapleStory (white text with a dark outline,
usually a player's name tag). Return a JSON array of strings, one entry per image in the same order, containing
exactly the text in that image, case-sensitive, nothing added. Player names are one word of letters and digits
(e.g. "Tira", "xPey2", "Sphinctus"). If an image has no readable text, use null. Do not guess missing letters."""


def _post(body: dict, api_key: str, model: str):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{re.sub(r'[^A-Za-z0-9._-]', '', model)}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        data=json.dumps(body), timeout=30,
    )
    r.raise_for_status()
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


def read_names(crops: list[np.ndarray], api_key: str, model: str = "gemini-2.5-flash") -> list[str | None] | None:
    """Reads every name-tag crop in ONE request. Returns a list the same length as crops, or None on failure."""
    if not crops:
        return []
    parts = []
    for c in crops:
        c = cv2.copyMakeBorder(c, 4, 4, 6, 6, cv2.BORDER_REPLICATE)
        big = cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", big)
        if not ok:
            return None
        parts.append({"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.tobytes()).decode()}})
    parts.append({"text": NAMES_PROMPT})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    try:
        arr = json.loads(_post(body, api_key, model))
    except Exception as exc:
        log.warning("Gemini name read failed: %s", exc)
        return None
    if not isinstance(arr, list) or len(arr) != len(crops):
        log.warning("Gemini returned %s entries for %d crops", len(arr) if isinstance(arr, list) else "bad", len(crops))
        return None
    return [a.strip() if isinstance(a, str) and a.strip() else None for a in arr]
