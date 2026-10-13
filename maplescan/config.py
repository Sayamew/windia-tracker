from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def app_dir() -> Path:
    """Folder containing the .exe (when frozen) or the scripts; config.json lives here."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "Config":
        p = Path(path)
        if not p.is_absolute():
            p = app_dir() / p
        if not p.exists():
            raise SystemExit(f"{p} not found. Copy config.example.json to config.json and edit it.")
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
