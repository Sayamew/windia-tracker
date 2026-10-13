from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import mss
import numpy as np


def enable_dpi_awareness() -> None:
    """Make screen coordinates match real pixels on scaled Windows displays."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int


def find_game_window(title_fragment: str) -> WindowRect:
    import pygetwindow as gw

    matches = [w for w in gw.getWindowsWithTitle(title_fragment) if w.width > 200 and w.height > 200]
    if not matches:
        raise RuntimeError(f"No window with '{title_fragment}' in its title was found.")
    w = matches[0]
    return WindowRect(w.left, w.top, w.width, w.height)


class Screen:
    """Captures the game window as a BGR numpy array and maps window coords to screen coords."""

    def __init__(self, title_fragment: str):
        self.title = title_fragment
        self.sct = mss.mss()
        self.rect = find_game_window(title_fragment)

    def refresh_rect(self) -> None:
        self.rect = find_game_window(self.title)

    def grab(self) -> np.ndarray:
        self.refresh_rect()
        r = self.rect
        shot = self.sct.grab({"left": r.left, "top": r.top, "width": r.width, "height": r.height})
        frame = np.array(shot)  # BGRA
        return frame[:, :, :3].copy()

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        return self.rect.left + int(x), self.rect.top + int(y)
