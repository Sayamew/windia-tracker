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

PROMPT = """This is a small crop of the MapleStory CHARACTER INFO window: the character sprite with its grey name plate
on the left, and the rows LEVEL, JOB, FAME (and GUILD) on the right.
Return a JSON object: {"name": string|null, "level": integer|null, "job": string|null, "fame": integer|null}
- name: the text on the grey name plate under the sprite
- level: the number in the LEVEL row
- job: the text in the JOB row
- fame: the number in the FAME row (it can be negative, e.g. -7)
Ignore the leaf icon and anything else. Use null for anything you cannot read clearly. Do not guess."""


def read_panel(panel_bgr: np.ndarray, api_key: str, model: str = "gemini-2.5-flash") -> dict | None:
    big = cv2.resize(panel_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", big)
    if not ok:
        return None
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.tobytes()).decode()}},
            {"text": PROMPT},
        ]}],
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
    job = match_job(d["job"]) if isinstance(d.get("job"), str) else None
    return {"name": name if name and NAME_RE.match(name) else None, "level": level, "job": job, "fame": fame}


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
