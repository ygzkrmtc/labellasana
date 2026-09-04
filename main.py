#!/usr/bin/env python3
"""
YOLOv8 Etiketleme Aracı - giriş noktası.

Çalıştırma:
    python main.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# YÜKLEME SIRASI ÖNEMLİ: onnxruntime'ın yerel (C++) kütüphanesi, Qt yüklendikten
# SONRA açılmaya çalışıldığında bazı Windows kurulumlarında
#     "DLL load failed ... A dynamic link library (DLL) initialization
#      routine failed"
# hatası verir; aynı içe aktarma Qt'siz bir kabukta sorunsuz çalışır. Sorun
# paketin kendisinde değil, Qt'nin DLL arama yolunu değiştirmesindedir.
# Bu yüzden yerel kütüphaneleri Qt'den ÖNCE yüklüyoruz.
#
# Hata burada yutulur; gerçek sebep app/inference.py tarafından kaydedilir ve
# "Yapay Zekâ → Kurulum Durumu" penceresinde gösterilir.
# (cv2 bilerek burada değil: opencv-python'un tam sürümü kendi Qt eklentilerini
#  taşır ve Qt'den önce yüklenirse arayüzü bozabilir.)
# ---------------------------------------------------------------------------
for _preload in ("numpy", "onnxruntime"):
    try:
        __import__(_preload)
    except Exception:
        pass

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

# HiDPI ayarları QApplication oluşturulmadan ÖNCE verilmelidir.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Betik başka bir dizinden çalıştırılsa da 'app' paketi bulunabilsin.
# (PyInstaller ile paketlendiğinde modüller arşivin içindedir, ek yol gerekmez.)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Etiketleyici")
    app.setOrganizationName("YOLO Labeler")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # Komut satırından klasör verilmişse doğrudan aç:  python main.py C:\resimler
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        window.load_folder(sys.argv[1])

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
