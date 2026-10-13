from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pyautogui

from . import gemini, infowindow
from .screen import Screen
from .vision import NAME_RE, OCR, NameTag

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
        key = self.cfg.get("gemini_api_key")
        if key:
            panel = infowindow.panel_crop(frame)
            if panel is not None:
                g = gemini.read_panel(panel, key, self.cfg.get("gemini_model", "gemini-2.5-flash"))
                if g:
                    # Gemini is the primary reader; the local OCR only fills what Gemini couldn't read.
                    name = g["name"] or name
                    level = g["level"] if g["level"] is not None else level
                    job = g["job"] or job
                    fame = g["fame"] if g["fame"] is not None else fame
        if not name or not NAME_RE.match(name) or level is None:
            return None
        return Inspection(name=name, level=level, job=job, fame=fame, avatar_png=reading.avatar_png)

    # -- main entry ------------------------------------------------------------------------------
    def inspect(self, tag: NameTag) -> Inspection | None:
        """Returns the reading for the player whose window opened, or None if no player window opened.
        (If a neighbouring player was clicked instead, that player's reading is returned; the caller
        saves it under the name in the window.)"""
        offsets = self.cfg.get("click_offsets", [-50, -36, -64])
        wait = float(self.cfg.get("open_wait_seconds", 0.6))
        try:
            self._close_info()  # start from a clean screen
            for dy in offsets:
                cx = tag.center_x
                cy = max(0, tag.y + int(dy * self.screen.rect.height / 1080))
                pyautogui.moveTo(*self.screen.to_screen(cx, cy), duration=0.08)
                pyautogui.doubleClick(interval=0.07)
                time.sleep(wait)
                result = self._read_open_window(self.screen.grab())
                if result is not None:
                    return result
                self._close_info()  # something opened but unreadable, or nothing: reset and retry
            return None
        finally:
            self._close_info()
            # park the mouse over empty UI space so hover tooltips don't cover name tags
            h = self.screen.rect.height
            pyautogui.moveTo(*self.screen.to_screen(int(self.screen.rect.width * 0.5), int(h * 0.97)), duration=0.05)
