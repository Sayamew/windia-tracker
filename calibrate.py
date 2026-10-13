"""Calibration helper.

  MapleCalibrate.exe          Saves debug_screen.png with the detected player names boxed and labelled
                              (green box = a name the bot will double-click, yellow dot = where it clicks).
  MapleCalibrate.exe --test   Really double-clicks the first detected player and reads their info window,
                              so you can check the whole loop once before pressing F4.
  MapleCalibrate.exe --info   Open a player's CHARACTER INFO window by hand first. Draws the regions the
                              bot reads (yellow = LEVEL/JOB/FAME, green = name plate, orange = sprite photo),
                              prints what was read, and saves debug_info.png and info_sprite.png.

The CHARACTER INFO window is found automatically, so there are no pixel regions to set.
"""
from __future__ import annotations

import argparse
import os

import cv2

from maplescan import infowindow
from maplescan.config import Config, app_dir
from maplescan.screen import Screen, enable_dpi_awareness
from maplescan import gemini
from maplescan.inspector import Inspector
from maplescan.vision import OCR, TagCache, read_name_tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", action="store_true", help="check the CHARACTER INFO window reading")
    ap.add_argument("--test", action="store_true", help="double-click the first detected player and read their window")
    args = ap.parse_args()

    os.chdir(app_dir())
    enable_dpi_awareness()
    cfg = Config.load("config.json")
    screen = Screen(cfg["window_title"])
    frame = screen.grab()
    out = frame.copy()

    if args.info:
        layout = infowindow.locate(frame)
        if layout is None:
            print("No CHARACTER INFO window found. Open one (the window with the pink LEVEL / JOB / FAME labels) and try again.")
            return
        colours = {"level": (0, 255, 255), "job": (0, 255, 255), "fame": (0, 255, 255),
                   "name": (0, 255, 0), "sprite": (255, 128, 0)}
        for key, (x0, y0, x1, y1) in infowindow.regions(frame, layout).items():
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), colours[key], 1)
        cv2.imwrite("debug_info.png", out)

        ocr = OCR()
        reading = infowindow.read_info_window(frame, ocr)
        print(f"name  : {reading.name}")
        print(f"level : {reading.level}")
        print(f"job   : {reading.job}")
        print(f"fame  : {reading.fame}")
        if reading.avatar_png:
            with open("info_sprite.png", "wb") as f:
                f.write(reading.avatar_png)
        missing = [k for k in ("name", "level", "job", "fame") if getattr(reading, k) is None]
        if missing:
            print(f"\nCould not read consistently: {', '.join(missing)}. "
                  + ("Gemini is on, so it will read these while scanning." if cfg.get("gemini_api_key")
                     else "Set gemini_api_key in config.json to let Gemini cover these."))
        print("\nWrote debug_info.png and info_sprite.png")
        return

    ocr = OCR()
    key = cfg.get("gemini_api_key")
    namer = (lambda crops: gemini.read_names(crops, key, cfg.get("gemini_model", "gemini-2.5-flash"))) if key else None
    tags = read_name_tags(frame, ocr, cfg, namer, TagCache())
    scale = frame.shape[0] / 1080
    for t in tags:
        cv2.rectangle(out, (t.x, t.y), (t.x + t.w, t.y + t.h), (0, 255, 0), 1)
        cv2.putText(out, f"{t.name} {t.confidence:.2f}", (t.x, t.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for dy in cfg.get("click_offsets", [-50]):
            cv2.circle(out, (t.center_x, t.y + int(dy * scale)), 3, (0, 255, 255), -1)
        print(f"{t.name}  conf={t.confidence:.2f}")
    cv2.imwrite("debug_screen.png", out)
    print(f"\nFound {len(tags)} player name(s). Wrote debug_screen.png (green = names, yellow dots = click points).")
    if not tags:
        print("No names found. Make sure the game is 1920x1080 and 'own_character' in config.json is your bot's name.")

    if args.test and tags:
        t = tags[0]
        print(f"\nTesting on '{t.name}': double-clicking in 3 seconds. Keep hands off the mouse.")
        import time
        time.sleep(3)
        res = Inspector(screen, ocr, cfg).inspect(t)
        if res is None:
            print("No player window opened. Check debug_screen.png: do the yellow dots sit on the character? "
                  "If not, adjust click_offsets in config.json. (It may also be a pet.)")
        else:
            print(f"Opened: {res.name}  level {res.level}  job {res.job}  fame {res.fame}")
            if res.avatar_png:
                with open("info_sprite.png", "wb") as f:
                    f.write(res.avatar_png)
                print("Saved info_sprite.png (the ranking photo).")


if __name__ == "__main__":
    main()
