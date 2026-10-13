"""Setup window (becomes MapleSetup.exe): writes config.json next to the exe, tests the connection."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk

import requests

from maplescan.config import app_dir

CONFIG = app_dir() / "config.json"
EXAMPLE = app_dir() / "config.example.json"

DEFAULTS = {
    "api_url": "https://ranking.sudocat.dev/api/ingest",
    "api_key": "",
    "world": "windia",
    "window_title": "MapleStory",
    "own_character": "",
    "gemini_api_key": "",
}


def load() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    if EXAMPLE.exists():
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))
    return dict(DEFAULTS)


class App(tk.Tk):
    FIELDS = [
        ("api_url", "Ranking site ingest URL"),
        ("api_key", "Ingest API key"),
        ("window_title", "Game window title contains"),
        ("own_character", "Your bot character's name"),
        ("gemini_api_key", "Gemini API key (optional fallback)"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title("MapleStory Scanner Setup (Windia)")
        self.resizable(False, False)
        self.cfg = load()
        self.vars: dict[str, tk.StringVar] = {}

        frm = ttk.Frame(self, padding=16)
        frm.grid()
        for i, (key, label) in enumerate(self.FIELDS):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=str(self.cfg.get(key, DEFAULTS.get(key, ""))))
            show = "*" if key in ("api_key", "gemini_api_key") else ""
            ttk.Entry(frm, textvariable=var, width=46, show=show).grid(row=i, column=1, padx=(12, 0), pady=4)
            self.vars[key] = var

        ttk.Label(frm, text="World: Windia   ·   Start: F4   ·   Stop: F5", foreground="#555").grid(
            row=len(self.FIELDS), column=0, columnspan=2, pady=(8, 4), sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=len(self.FIELDS) + 1, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(btns, text="Test connection", command=self.test).pack(side="left", padx=4)
        ttk.Button(btns, text="Save", command=self.save).pack(side="left", padx=4)

    def values(self) -> dict:
        return {k: v.get().strip() for k, v in self.vars.items()}

    def save(self) -> None:
        v = self.values()
        if not v["api_key"] or not v["own_character"]:
            messagebox.showwarning("Missing info", "API key and your character's name are required.")
            return
        cfg = {**self.cfg, **v, "world": "windia"}
        CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        messagebox.showinfo("Saved", f"Saved to {CONFIG}\n\nNext: run MapleCalibrate.exe, then MapleScanner.exe.")

    def test(self) -> None:
        v = self.values()
        try:
            r = requests.post(v["api_url"], json={"players": [{"name": "ConnectionTest"}]},
                              headers={"Authorization": f"Bearer {v['api_key']}"}, timeout=10)
        except requests.RequestException as exc:
            messagebox.showerror("Connection failed", str(exc))
            return
        if r.status_code == 401:
            messagebox.showerror("Rejected", "The site rejected the API key (401).")
        elif r.ok:
            messagebox.showinfo("Connected", "The ranking site accepted the request.")
        else:
            messagebox.showerror("Error", f"Server answered {r.status_code}.")


if __name__ == "__main__":
    App().mainloop()
