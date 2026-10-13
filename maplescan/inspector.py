from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import pyautogui

from . import gemini, infowindow
from .screen import Screen
from .vision import NAME_RE, OCR, NameTag, click_blocked

pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.02

log = logging.getLogger("inspector")


@dataclass
class Inspection:
    name: str          # the name shown in the info window (large font, more reliable than the tag)
    level: int
    job: str | None
    fame: int | None = None
    avatar_png: bytes | None = None


class Inspector:
    """Double-clicks a player, reads their CHARACTER INFO window (LEVEL, JOB, FAME, sprite), closes it.

    The window is found automatically (pink label cells), so there are no pixel regions to calibrate.
    The click lands on the character sprite above the name tag; if nothing opens it retries a little
    higher/lower. If local OCR can't read every field consistently and a Gemini key is set, the cropped
    panel goes to Gemini instead. Anything unverified is dropped, never guessed.
    """

    def __init__(self, screen: Screen, ocr: OCR, cfg):
        self.screen = screen
        self.ocr = ocr
        self.cfg = cfg

    # -- window handling -------------------------------------------------------------------------
    def _close_info(self) -> None:
        """Click the window's X (position derived from the label cells), fall back to Esc."""
        for _ in range(2):
            frame = self.screen.grab()
            L = infowindow.locate(frame)
            if L is None:
                return
            x, y = infowindow.close_button(L)
            pyautogui.click(*self.screen.to_screen(x, y))
            time.sleep(0.2)
            if infowindow.locate(self.screen.grab()) is None:
                return
            pyautogui.press("esc")
            time.sleep(0.2)

    def _read_open_window(self, frame):
        reading = infowindow.read_info_window(frame, self.ocr)
        if reading is None:
            return None
        name, level, job, fame = reading.name, reading.level, reading.job, reading.fame
        local = (reading.name, reading.level, reading.job, reading.fame)
        job_text = None
        key = self.cfg.get("gemini_api_key")
        if key:
            panel = infowindow.panel_crop(frame)
            if panel is not None:
                g = gemini.read_panel(panel, key, self.cfg.get("gemini_model", "gemini-2.5-flash"),
                                      infowindow.plate_crop(frame))
                if g:
                    # Gemini is the primary reader; the local OCR only fills what Gemini couldn't read.
                    name = g["name"] or name
                    level = g["level"] if g["level"] is not None else level
                    job = g["job"] or job
                    fame = g["fame"] if g["fame"] is not None else fame
                    job_text = g.get("job_text")
        if not name or not NAME_RE.match(name) or level is None:
            return None
        if job is None:
            log.warning("job not recognised for %s (row said %r, local OCR %r): saved without a class", name, job_text, local[2])
            self._debug_dump(frame, name)
        return Inspection(name=name, level=level, job=job, fame=fame, avatar_png=reading.avatar_png)

    def _debug_dump(self, frame, name: str) -> None:
        """Keep a picture of anything we couldn't classify so it can be checked/fixed later."""
        try:
            import cv2
            from .config import app_dir
            d = app_dir() / "debug"
            d.mkdir(exist_ok=True)
            panel = infowindow.panel_crop(frame)
            if panel is not None and len(list(d.glob("*.png"))) < 200:
                cv2.imwrite(str(d / f"{int(time.time())}_{name}.png"), panel)
        except Exception:
            pass

    # -- main entry ------------------------------------------------------------------------------
    # (dx, dy) in pixels at 1080p from the middle of the name tag; dy is negative = up the body
    SPOTS = [(0, -50), (0, -36), (0, -64), (-14, -46), (14, -46), (-16, -60), (16, -60),
             (-10, -30), (10, -30), (0, -78), (-22, -48), (22, -48)]

    def inspect(self, tag: NameTag, is_target=None, on_other=None) -> Inspection | None:
        """Double-clicks spots on the tag's character until the window that opens belongs to that
        character. With overlapping sprites the game opens whoever is in front, so a wrong window is
        handed to `on_other` (it is still a valid reading) and the next spot is tried. Returns the
        target's reading, or None if no spot produced it. Without `is_target`, any window counts."""
        spots = self.cfg.get("click_spots") or self.SPOTS
        tries = int(self.cfg.get("max_click_tries", 8))
        wait = float(self.cfg.get("open_wait_seconds", 0.6))
        seen_other: dict[str, int] = {}
        try:
            self._close_info()  # start from a clean screen
            before = self.screen.grab()
            sc = self.screen.rect.height / 1080
            made = 0
            for dx, dy in spots:
                if made >= tries:
                    break
                cx = max(0, tag.center_x + int(dx * sc))
                cy = max(0, tag.y + int(dy * sc))
                if click_blocked(before, cx, cy, sc):  # a game box / balloon is over this spot
                    continue
                made += 1
                pyautogui.moveTo(*self.screen.to_screen(cx, cy), duration=random.uniform(0.06, 0.14))
                pyautogui.doubleClick(interval=0.07)
                time.sleep(wait)
                result = self._read_open_window(self.screen.grab())
                if result is not None:
                    if is_target is None or is_target(result.name):
                        return result
                    seen_other[result.name] = seen_other.get(result.name, 0) + 1
                    if seen_other[result.name] >= 3:
                        return result  # the same person keeps opening: the tag was probably misread
                    if on_other and seen_other[result.name] == 1:
                        on_other(result)
                    log.info("opened %s instead of %s, trying another spot", result.name, tag.name)
                self._close_info()  # wrong person or nothing readable: reset and retry elsewhere
            return None
        finally:
            self._close_info()
            # park the mouse over empty UI space so hover tooltips don't cover name tags
            h = self.screen.rect.height
            pyautogui.moveTo(*self.screen.to_screen(int(self.screen.rect.width * 0.5), int(h * 0.97)), duration=0.05)
