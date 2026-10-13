@echo off
REM Builds MapleSetup.exe, MapleCalibrate.exe and MapleScanner.exe into dist\
REM Requires Python 3.10+ on Windows.
python -m pip install -r requirements.txt pyinstaller || exit /b 1

python -m PyInstaller --noconfirm --onefile --windowed --name MapleSetup setup_gui.py || exit /b 1
python -m PyInstaller --noconfirm --onefile --name MapleCalibrate --collect-all rapidocr_onnxruntime calibrate.py || exit /b 1
python -m PyInstaller --noconfirm --onefile --name MapleScanner --collect-all rapidocr_onnxruntime --hidden-import keyboard run.py || exit /b 1

copy /Y config.example.json dist\config.example.json >nul
echo.
echo Done. Copy the dist folder anywhere. Run MapleSetup.exe first, then MapleCalibrate.exe, then MapleScanner.exe (F4 start, F5 stop).
echo MapleScanner.exe should be run as administrator so the F4/F5 hotkeys work while the game is focused.
