"""MapleStory Classic World (Windia) scanner.

The character stays where it is. Press F4 to start scanning and F5 to stop. While running, every
3-8 seconds it reads the name tags of the players on screen, reads each new/stale player's level and
job from their info window, and uploads name, level, job and a sprite crop to the ranking site.
"""
from __future__ import annotations

import difflib
import logging
import os
import random
import threading
import time

from maplescan.config import Config, app_dir
from maplescan.inspector import Inspector
from maplescan.screen import Screen, enable_dpi_awareness
from maplescan.uploader import Uploader
from maplescan import gemini
from maplescan.names import NameBook
from maplescan.vision import OCR, TagCache, crop_avatar, norm_name, read_name_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scanner")


def scan_once(screen, ocr, inspector, uploader, cfg, last_inspected: dict[str, float], namer=None, cache=None, book=None) -> None:
    frame = screen.grab()
    tags = read_name_tags(frame, ocr, cfg, namer, cache)
    now = time.time()
    entries: list[dict] = []
    inspected = 0

    cooldown = max(3600.0, float(cfg["min_reinspect_seconds"]))  # default 1 hour, for successful AND failed clicks

    def last_click(key: str) -> float:
        """Most recent click on this player (a slightly misread spelling counts as the same player)."""
        best = last_inspected.get(key, 0)
        for k, v in last_inspected.items():
            if v > best and abs(len(k) - len(key)) <= 1 and difflib.SequenceMatcher(None, k, key).ratio() >= 0.85:
                best = v
        return best

    # Never-clicked players first, then whoever was clicked longest ago. Anyone clicked within the
    # cooldown is skipped, so each cycle goes for a different person (or clicks nobody).
    for tag in sorted(tags, key=lambda t: last_click(norm_name(t.name))):
        key = norm_name(tag.name)
        if now - last_click(key) >= cooldown and inspected < cfg["max_inspect_per_cycle"]:
            last_inspected[key] = time.time()  # counts as clicked even if the window doesn't open
            result = inspector.inspect(tag)
            inspected += 1
            if result:
                # Save under the name shown in the info window (large font, more reliable than the tag).
                avatar = result.avatar_png or crop_avatar(frame, tag, cfg["avatar_box"])
                final = book.vote(result.name) if book else result.name
                entries.append(uploader.make_entry(final, result.level, result.job, avatar, result.fame))
                last_inspected[norm_name(result.name)] = time.time()
                log.info("read %s: level %s, %s, fame %s", final, result.level, result.job, result.fame)
                continue
            log.info("could not open a player window for '%s' (pet/NPC or misread name?)", tag.name)
        # Seen but not (re)inspected: heartbeat so "last checked" stays fresh.
        entries.append(uploader.make_entry(tag.name))

    if entries:
        uploader.queue(entries)
    res = uploader.flush()
    log.info("saw %d players, inspected %d%s", len(tags), inspected,
             f", server: {res['upserted']} updated / {res['touched']} touched" if res else "")


def main() -> None:
    os.chdir(app_dir())
    enable_dpi_awareness()
    cfg = Config.load("config.json")

    screen = Screen(cfg["window_title"])
    ocr = OCR()
    inspector = Inspector(screen, ocr, cfg)
    uploader = Uploader(cfg["api_url"], cfg["api_key"], cfg["world"])

    key = cfg.get("gemini_api_key")
    model = cfg.get("gemini_model", "gemini-2.5-flash")
    namer = (lambda crops: gemini.read_names(crops, key, model)) if key else None
    cache = TagCache()
    book = NameBook(app_dir() / "names.json")
    if not namer:
        log.warning("No Gemini key set: names are read by the local reader only, which often drops letters.")

    scanning = threading.Event()
    last_inspected: dict[str, float] = {}

    start_key = cfg.get("start_hotkey", "f4")
    stop_key = cfg.get("stop_hotkey", "f5")

    def start() -> None:
        if not scanning.is_set():
            scanning.set()
            log.info(">>> SCANNING STARTED (press %s to stop)", stop_key.upper())

    def stop() -> None:
        if scanning.is_set():
            scanning.clear()
            log.info("<<< SCANNING STOPPED (press %s to start)", start_key.upper())

    try:
        import keyboard

        keyboard.add_hotkey(start_key, start)
        keyboard.add_hotkey(stop_key, stop)
    except Exception as exc:
        raise SystemExit(
            f"Could not register hotkeys ({exc}). On Windows, try running the program as administrator."
        )

    log.info("Ready. Focus the game, then press %s to start and %s to stop. Close this window to quit.",
             start_key.upper(), stop_key.upper())

    lo, hi = cfg["scan_interval_seconds"]
    try:
        while True:
            if not scanning.is_set():
                time.sleep(0.2)
                continue
            try:
                scan_once(screen, ocr, inspector, uploader, cfg, last_inspected, namer, cache, book)
            except SystemExit:
                raise
            except Exception:
                log.exception("scan cycle failed")
            # Sleep in small steps so F5 takes effect immediately.
            deadline = time.time() + random.uniform(lo, hi)
            while time.time() < deadline and scanning.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("bye")


if __name__ == "__main__":
    main()
