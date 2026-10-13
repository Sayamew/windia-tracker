from __future__ import annotations

import base64
import logging
import time

import requests

log = logging.getLogger("uploader")

MAX_BUFFER = 500


class Uploader:
    """Posts batches to the ranking site; buffers in memory if the site is unreachable."""

    def __init__(self, api_url: str, api_key: str, world: str):
        self.url = api_url
        self.world = world
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.pending: list[dict] = []

    @staticmethod
    def make_entry(name: str, level: int | None = None, job: str | None = None,
                   avatar_png: bytes | None = None, fame: int | None = None) -> dict:
        entry: dict = {"name": name}
        if level is not None:
            entry["level"] = level
            if job:
                entry["job"] = job
            if fame is not None:
                entry["fame"] = fame
            if avatar_png:
                entry["avatar_b64"] = base64.b64encode(avatar_png).decode("ascii")
        return entry

    def queue(self, entries: list[dict]) -> None:
        self.pending.extend(entries)
        if len(self.pending) > MAX_BUFFER:
            self.pending = self.pending[-MAX_BUFFER:]

    def flush(self) -> dict | None:
        """Send up to 100 pending entries. Returns the server response or None on failure."""
        if not self.pending:
            return None
        batch = self.pending[:100]
        for attempt in range(3):
            try:
                r = self.session.post(self.url, json={"world": self.world, "players": batch}, timeout=15)
                if r.status_code == 401:
                    raise SystemExit("Ranking site rejected the API key (401). Check api_key in config.json.")
                r.raise_for_status()
                self.pending = self.pending[len(batch):]
                data = r.json()
                if data.get("errors"):
                    log.warning("server rejected entries: %s", data["errors"])
                return data
            except requests.RequestException as exc:
                log.warning("upload failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return None
