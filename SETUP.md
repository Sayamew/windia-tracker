# Bot setup (Windows, 1920x1080)

The bot stands in front of the **Free Market portal in Henesys, Channel 10**. Every 3-8 seconds it finds the players
on screen, **double-clicks** each one to open their Character Info window, reads name / level / job / fame, copies the
sprite as the ranking photo, and uploads it. Anyone imported is skipped for 10 minutes.

**F4 = start, F5 = stop.** Automating the game likely breaks Nexon's rules; use an account you can afford to lose.

## 1. Get the exes
GitHub builds them for you: open the repo's **Releases** page (right side) -> **Latest build** -> download
`MapleScanner-windows.zip` and unzip it anywhere. (If there is no release yet: **Actions** -> *Build Windows EXEs* -> *Run workflow*, wait ~5 minutes.)

You get `MapleSetup.exe`, `MapleCalibrate.exe`, `MapleScanner.exe` and `config.example.json`. Keep them in one folder.

## 2. Setup (once)
Run **MapleSetup.exe** and fill in:
- Ingest URL: `https://ranking.sudocat.dev/api/ingest`
- Ingest API key: the `ingest_api_key` from the website's `app/config.php`
- Game window title: `MapleStory` unless your window title differs
- Your bot character's name (the bot ignores itself)
- Gemini key (optional): backup reader if the text reader is unsure

Press **Test connection**, then **Save**.

## 3. Check it (once)
1. Game at **1920x1080**, windowed or borderless, bot character in Henesys CH10 with players around. Close the Character Info window.
2. Run **MapleCalibrate.exe**. Open `debug_screen.png`: green boxes are the player names it found, yellow dots are where it will double-click (they should sit on the characters).
3. Run **MapleCalibrate.exe --test** (from Command Prompt in that folder). It double-clicks the first player found, reads their Character Info window, closes it, and prints name / level / job / fame. It also saves `info_sprite.png`.
4. If the dots miss the characters, change `click_offsets` in `config.json` (pixels above the name; default `-50, -36, -64`).
   If real players are missing, check `own_character` and `ignore_zones`.

## 4. Run
1. Park the character in front of the FM portal, Channel 10.
2. Run **MapleScanner.exe** as administrator (right-click). Click the game, press **F4**. **F5** stops. Slam the mouse into a screen corner to abort.

## Notes
- Pets and NPCs get tried once or twice, then ignored for a while (30 s, 1 min, 2 min ... up to 30 min).
- Fame and sprite come from the Character Info window, whatever the player's look is.
- A value the bot can't read reliably is dropped, never guessed.
- If clicks do nothing in game, run the exe as administrator.
