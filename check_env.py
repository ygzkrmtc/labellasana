#!/usr/bin/env python3
"""
Ortam denetçisi — kurulum sonrası "bende çalışmadı" durumlarını teşhis eder.

    python check_env.py

Yalnızca standart kütüphaneyi kullanır, dolayısıyla hiçbir şey kurulmamışken de
çalışır. Her sorun için ne yapılacağını da yazar.

Çıkış kodu: 0 = çekirdek çalışır durumda, 1 = uygulama açılmaz.
"""

import importlib
import importlib.util
import os
import platform
import subprocess
import sys

# Windows'ta varsayılan çıktı kodlaması cp1252'dir ve 'ş', 'ı', 'ğ' harflerini
# kodlayamaz; UTF-8'e çeviriyoruz ki denetçi kendi mesajları yüzünden çökmesin.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MIN_PYTHON = (3, 8)
RECOMMENDED_PYTHON = "3.10 - 3.12"

GREEN = "  [ TAMAM ]"
YELLOW = "  [ EKSIK ]"
RED = "  [ HATA  ]"


def _print_header(text):
    print("\n" + text)
    print("-" * len(text))


def _try_import(name):
    """(başarılı mı, sürüm ya da hata metni)"""
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "?")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_python():
    _print_header("Python")
    print(f"  Sürüm      : {platform.python_version()}")
    print(f"  Yorumlayıcı: {sys.executable}")
    print(f"  İşletim s. : {platform.system()} {platform.release()} ({platform.machine()})")

    if sys.version_info < MIN_PYTHON:
        print(RED, f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ gerekli.")
        return False
    if sys.version_info[:2] >= (3, 13):
        print(YELLOW, f"Çok yeni bir Python. Bazı paketler (onnxruntime, torch) "
                      f"bu sürüm için gecikmeli paket yayınlar.")
        print("         Sorun yaşarsanız önerilen aralık:", RECOMMENDED_PYTHON)
    else:
        print(GREEN, "Sürüm uygun.")

    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print(YELLOW, "Sanal ortam aktif görünmüyor. Paketleri sistem Python'una "
                      "kurmak karışıklık yaratır.")
    else:
        print(GREEN, "Sanal ortam aktif.")
    return True


def check_core():
    _print_header("Çekirdek (zorunlu)")
    ok, info = _try_import("PyQt5.QtWidgets")
    if ok:
        try:
            from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
            print(GREEN, f"PyQt5 {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})")
        except Exception:
            print(GREEN, "PyQt5 yüklendi.")
        return True
    print(RED, f"PyQt5 yüklenemedi -> {info}")
    print("         Çözüm: pip install -r requirements.txt")
    return False


def check_optional():
    _print_header("İsteğe bağlı özellikler")
    results = {}
    for label, module, package, feature in (
            ("numpy", "numpy", "numpy",
             "görüntü iyileştirme, kenar katmanı, manyetik kutu"),
            ("OpenCV", "cv2", "opencv-python-headless", "CLAHE (yerel kontrast)"),
            ("onnxruntime", "onnxruntime", "onnxruntime",
             "model destekli ön-etiketleme")):
        ok, info = _try_import(module)
        results[module] = ok
        if ok:
            print(GREEN, f"{label} {info} — {feature} açık")
        else:
            print(YELLOW, f"{label} yok — {feature} kapalı")
            print(f"         {info}")
            print(f"         Çözüm: pip install {package}")
    return results


def check_opencv_conflict():
    try:
        import cv2
    except Exception:
        return
    qt_dir = os.path.join(os.path.dirname(cv2.__file__), "qt")
    if os.path.isdir(qt_dir):
        _print_header("UYARI: OpenCV'nin tam sürümü kurulu")
        print(YELLOW, "'opencv-python' kendi Qt eklentilerini taşır ve PyQt5 ile "
                      "çakışıp arayüzü bozabilir.")
        print("         Çözüm: pip uninstall opencv-python")
        print("                pip install opencv-python-headless")


def check_load_order():
    """
    Qt yüklendikten SONRA onnxruntime yüklenince Windows'ta DLL hatası
    çıkabiliyor. main.py bunu önlemek için sırayı tersine çeviriyor; burada
    kullanıcının sisteminde gerçekten böyle bir sorun var mı diye bakıyoruz.
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return
    if importlib.util.find_spec("PyQt5") is None:
        return          # PyQt5 yokken bu denetim anlamsız (yanlış alarm verir)
    _print_header("Yükleme sırası denetimi (Qt + onnxruntime)")
    code = "import PyQt5.QtWidgets, onnxruntime; print('OK')"
    try:
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=180)
    except Exception as exc:
        print(YELLOW, f"Denetim çalıştırılamadı: {exc}")
        return

    if result.returncode == 0:
        print(GREEN, "Qt'den sonra onnxruntime sorunsuz yükleniyor.")
        return

    print(YELLOW, "Qt'den SONRA onnxruntime yüklenemiyor (bilinen çakışma).")
    print("         Uygulama bunu zaten önlüyor: main.py, onnxruntime'ı Qt'den "
          "önce yükler.")
    print("         Kendi betiğinizde kullanacaksanız aynı sırayı uygulayın.")
    tail = (result.stderr or "").strip().splitlines()
    if tail:
        print("         Hata:", tail[-1][:160])
    if "DLL load failed" in (result.stderr or "") and os.name == "nt":
        print("         Devam ederse Visual C++ Redistributable kurun:")
        print("         https://aka.ms/vs/17/release/vc_redist.x64.exe")


def check_linux_libraries():
    if platform.system() != "Linux":
        return
    _print_header("Linux sistem kütüphaneleri")
    print("  Qt için gerekli paketler eksikse uygulama açılmaz. Ubuntu/Debian:")
    print("    sudo apt install libgl1 libegl1 libxkbcommon-x11-0 \\")
    print("                     libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \\")
    print("                     libxcb-randr0 libxcb-render-util0 libxcb-xinerama0")


def main() -> int:
    print("=" * 62)
    print(" YOLOv8 Etiketleme İstasyonu — ortam denetimi")
    print("=" * 62)

    python_ok = check_python()
    core_ok = check_core()
    check_optional()
    check_opencv_conflict()
    check_load_order()
    check_linux_libraries()

    _print_header("Sonuç")
    if python_ok and core_ok:
        print(GREEN, "Uygulama çalışabilir:  python main.py")
        print("         Tüm testler için  :  python -m tests.run_all")
        return 0
    print(RED, "Uygulama bu haliyle açılmaz. Yukarıdaki çözümleri uygulayın.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
