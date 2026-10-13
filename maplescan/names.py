"""Remembers every spelling the bot has read for a player and settles on the most common one.

The game font makes capital I / lowercase l / 1 and O / 0 look alike, so a name can be read slightly
differently from scan to scan. Readings that differ only in those characters are treated as the same
player; the spelling seen most often wins (ties keep the first spelling seen).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .vision import norm_name


class NameBook:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.seen: dict[str, Counter] = {}
        if path and path.exists():
            try:
                for k, v in json.loads(path.read_text(encoding="utf-8")).items():
                    self.seen[k] = Counter(v)
            except Exception:
                self.seen = {}

    def vote(self, name: str) -> str:
        """Record a reading and return the spelling to save under."""
        c = self.seen.setdefault(norm_name(name), Counter())
        c[name] += 1  # Counter keeps insertion order, so max() below keeps the first spelling on ties
        best = max(c.items(), key=lambda kv: kv[1])[0]
        self._save()
        return best

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.write_text(json.dumps({k: dict(v) for k, v in self.seen.items()}), encoding="utf-8")
        except Exception:
            pass
