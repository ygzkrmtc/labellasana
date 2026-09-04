@echo off
REM ============================================================
REM  YOLOv8 Etiketleme Istasyonu - .exe uretir
REM
REM  Kullanim (sanal ortam AKTIFKEN):
REM     build_exe.bat            -> tek dosya
REM     build_exe.bat onedir     -> klasor (daha guvenilir DLL yuklemesi)
REM
REM  Eski cikti kilitliyse (virus tarayici / acik Gezgin penceresi) betik
REM  durmaz, ciktiyi yeni bir klasore yazar.
REM ============================================================
setlocal enabledelayedexpansion

set MODE=--onefile
set MODEADI=tek dosya
set ALTKLASOR=
if /i "%~1"=="onedir" (
    set MODE=--onedir
    set MODEADI=klasor
    set ALTKLASOR=YOLO_Etiketleyici\
)

echo.
echo === Derleme bicimi: %MODEADI% ===

echo.
echo === PyInstaller kontrol ediliyor ===
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller kurulu degil, kuruluyor...
    python -m pip install pyinstaller || goto :error
)

echo.
echo === Istege bagli paketler denetleniyor ===
set EXTRA=
python -c "import onnxruntime" >nul 2>&1
if not errorlevel 1 (
    echo   onnxruntime bulundu - exe'ye dahil edilecek
    set EXTRA=!EXTRA! --collect-all onnxruntime
) else (
    echo   onnxruntime yok - yapay zeka menusu exe'de pasif olacak
)
python -c "import cv2" >nul 2>&1
if not errorlevel 1 (
    echo   OpenCV bulundu - exe'ye dahil edilecek
    set EXTRA=!EXTRA! --collect-binaries cv2
)

echo.
echo === Calisan surum varsa sonlandiriliyor ===
taskkill /F /IM "YOLO_Etiketleyici.exe" >nul 2>&1
if not errorlevel 1 (
    echo   Acik olan YOLO_Etiketleyici.exe kapatildi.
    ping -n 4 127.0.0.1 >nul
)

echo.
echo === Onceki cikti temizleniyor ===
if exist build rmdir /s /q build >nul 2>&1

set DISTPATH=dist
set TRY=0
:temizle
set /a TRY+=1
if exist "%DISTPATH%" rmdir /s /q "%DISTPATH%" >nul 2>&1
if not exist "%DISTPATH%" goto :temiz
if !TRY! GEQ 3 (
    REM Kilidi kirmak yerine yeni bir klasore yaziyoruz: derleme her halukarda ilerlesin
    set DISTPATH=dist_!RANDOM!
    echo   Eski dist klasoru kilitli ^(virus tarayici veya acik Gezgin penceresi^).
    echo   Cikti !DISTPATH! klasorune yazilacak; eski dist'i sonra elle silebilirsiniz.
    goto :temiz
)
echo   Kilitli, 3 saniye sonra tekrar denenecek... ^(!TRY!/3^)
ping -n 4 127.0.0.1 >nul
goto :temizle
:temiz

echo.
echo === Derleniyor (birkac dakika surebilir) ===
set ICON=
if exist icon.ico set ICON=--icon=icon.ico

python -m PyInstaller ^
    %MODE% ^
    --windowed ^
    --name "YOLO_Etiketleyici" ^
    --distpath "!DISTPATH!" ^
    --clean ^
    --noconfirm ^
    %ICON% ^
    !EXTRA! ^
    --runtime-hook rthook_preload.py ^
    --exclude-module torch ^
    --exclude-module ultralytics ^
    --exclude-module matplotlib ^
    --exclude-module pandas ^
    --exclude-module tkinter ^
    --exclude-module PySide2 ^
    --exclude-module PySide6 ^
    --exclude-module PyQt6 ^
    main.py || goto :error

echo.
echo === TAMAM ===
echo Cikti: !DISTPATH!\!ALTKLASOR!YOLO_Etiketleyici.exe
if /i "%~1"=="onedir" echo Dagitirken KLASORUN TAMAMINI kopyalayin.
echo.
dir /b "!DISTPATH!"
goto :eof

:error
echo.
echo === HATA: derleme basarisiz ===
exit /b 1
