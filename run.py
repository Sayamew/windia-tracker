"""MapleStory Classic World (Windia) scanner.

The character stays where it is. Press F4 to start scanning and F5 to stop. While running, every
1-2 seconds it reads the name tags of the players on screen, reads each new/stale player's level and
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


ARRIVAL_WINDOW = 45.0  # seconds a newcomer keeps top priority
GONE_AFTER = 5.0       # not seen for this long = left the screen (so a return counts as an arrival)


class State:
    """What the scanner remembers: who was clicked when, who failed, who just arrived."""

    def __init__(self, cooldown: float, fail_base: float) -> None:
        self.cooldown = max(3600.0, cooldown)  # one hour per player
        self.fail_base = fail_base
        self.clicked: dict[str, float] = {}      # norm name -> last time we clicked and read them
        self.fails: dict[str, tuple[float, int]] = {}  # norm name -> (last failed click, count)
        self.alias: dict[str, str] = {}          # partly hidden tag spelling -> real name
        self.present: dict[str, list[float]] = {}  # norm name -> [first seen, last seen]

    @staticmethod
    def _same(a: str, b: str) -> bool:
        if a == b:
            return True
        short, long_ = sorted((a, b), key=len)
        if len(short) >= 4 and long_.startswith(short):  # name cut off by another player / sprite
            return True
        return abs(len(a) - len(b)) <= 1 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.85

    def last_click(self, key: str) -> float:
        key = self.alias.get(key, key)
        best = 0.0
        for k, v in self.clicked.items():
            if v > best and self._same(k, key):
                best = v
        return best

    def on_cooldown(self, key: str, now: float) -> bool:
        return now - self.last_click(key) < self.cooldown

    def fail_wait(self, key: str, now: float) -> bool:
        t, n = self.fails.get(key, (0.0, 0))
        return n > 0 and now - t < min(self.fail_base * (2 ** (n - 1)), self.cooldown)

    def mark_clicked(self, key: str, real: str, now: float) -> None:
        self.clicked[real] = now
        self.clicked[key] = now
        if key != real:
            self.alias[key] = real
        self.fails.pop(key, None)

    def mark_failed(self, key: str, now: float) -> None:
        self.fails[key] = (now, self.fails.get(key, (0.0, 0))[1] + 1)

    def update_presence(self, keys: set[str], now: float) -> set[str]:
        """Returns the keys that just came into view."""
        arrived = set()
        for k in keys:
            rec = self.present.get(k)
            if rec is None or now - rec[1] > GONE_AFTER:
                self.present[k] = [now, now]
                arrived.add(k)
            else:
                rec[1] = now
        for k in [k for k, r in self.present.items() if now - r[1] > 120]:
            del self.present[k]
        return arrived


def scan_once(screen, ocr, inspector, uploader, cfg, state: State, namer=None, cache=None, book=None) -> int:
    """One look at the screen. Returns how many eligible players are still waiting, so the caller can
    come back quickly instead of sleeping the full interval."""
    frame = screen.grab()
    tags = read_name_tags(frame, ocr, cfg, namer, cache)
    now = time.time()
    entries: list[dict] = []
    inspected = 0
    state.update_presence({norm_name(t.name) for t in tags}, now)

    def priority(t) -> tuple:
        key = norm_name(t.name)
        first = state.present.get(key, [now, now])[0]
        last = state.last_click(key)
        newcomer = last == 0 and now - first <= ARRIVAL_WINDOW
        # 0: someone who just walked in (newest first), 1: anyone never clicked, 2: oldest click first
        return (0 if newcomer else 1 if last == 0 else 2, -first if last == 0 else last)

    def eligible(t) -> bool:
        key = norm_name(t.name)
        return not state.on_cooldown(key, now) and not state.fail_wait(key, now)

    queue = [t for t in sorted(tags, key=priority) if eligible(t)]
    done_keys: set[str] = set()
    for tag in queue:
        key = norm_name(tag.name)
        if inspected >= cfg["max_inspect_per_cycle"]:
            break
        # the same person may have been clicked a moment ago through a neighbouring tag
        if key in done_keys or state.on_cooldown(key, time.time()):
            continue
        def is_target(name: str, key=key) -> bool:
            n = norm_name(name)
            return state._same(n, key) or state._same(n, state.alias.get(key, key))

        def record(res, tag=tag) -> None:
            """Save a reading (the person we aimed at, or a neighbour whose window opened instead)."""
            avatar = (res.avatar_png or crop_avatar(frame, tag, cfg["avatar_box"])) if is_target(res.name) else (res.avatar_png or None)
            final = book.vote(res.name) if book else res.name
            entries.append(uploader.make_entry(final, res.level, res.job, avatar, res.fame))
            state.clicked[norm_name(res.name)] = time.time()
            done_keys.add(norm_name(res.name))
            log.info("read %s: level %s, %s, fame %s", final, res.level, res.job, res.fame)

        result = inspector.inspect(tag, is_target=is_target, on_other=record)
        inspected += 1
        t_now = time.time()
        if result:
            record(result)
            state.mark_clicked(key, norm_name(result.name), t_now)
            done_keys.add(key)
        else:
            state.mark_failed(key, t_now)
            log.info("could not open the window of '%s' (NPC, pet, hidden or misread name)", tag.name)

    # Everyone seen but not inspected this round: heartbeat so "last checked" stays fresh.
    for t in tags:
        if norm_name(t.name) not in done_keys:
            entries.append(uploader.make_entry(t.name))
    if entries:
        uploader.queue(entries)
    res = uploader.flush()
    waiting = sum(1 for t in queue if not state.on_cooldown(norm_name(t.name), time.time())
                  and norm_name(t.name) not in done_keys and not state.fail_wait(norm_name(t.name), time.time()))
    log.info("saw %d players, inspected %d, %d waiting%s", len(tags), inspected, waiting,
             f", server: {res['upserted']} updated / {res['touched']} touched" if res else "")
    return waiting


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
    state = State(float(cfg["min_reinspect_seconds"]), float(cfg.get("fail_retry_seconds", 60)))

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
    lo, hi = min(lo, 1.0), min(hi, 2.5)  # older configs had 3-8 s; newcomers should be caught fast
    try:
        while True:
            if not scanning.is_set():
                time.sleep(0.2)
                continue
            try:
                waiting = scan_once(screen, ocr, inspector, uploader, cfg, state, namer, cache, book)
            except SystemExit:
                raise
            except Exception:
                log.exception("scan cycle failed")
                waiting = 0
            # Someone still waiting: look again almost at once (a short human-like pause);
            # otherwise a quick re-check so newcomers are caught fast. F5 takes effect within 0.1 s.
            deadline = time.time() + (random.uniform(0.4, 0.9) if waiting else random.uniform(lo, hi))
            while time.time() < deadline and scanning.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("bye")


if __name__ == "__main__":
    main()
