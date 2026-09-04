"""
PyInstaller çalışma zamanı kancası (runtime hook).

NEDEN: onnxruntime'ın yerel kütüphanesi, Qt yüklendikten SONRA açılmaya
çalışıldığında Windows'ta
    "DLL load failed ... A dynamic link library (DLL) initialization
     routine failed"
hatası verir. Kaynaktan çalışırken main.py bunu, onnxruntime'ı PyQt5'ten önce
içe aktararak çözer. Ancak PyInstaller ile paketlendiğinde PyInstaller'ın kendi
başlangıç kancaları (Qt eklenti yollarını kuranlar dahil) main.py'den ÖNCE
çalışır; bu yüzden aynı önyüklemenin kanca aşamasında yapılması gerekir.

Bu dosya derlemeye `--runtime-hook rthook_preload.py` ile eklenir ve diğer
her şeyden önce çalışır.

Hatalar bilerek yutulur: paket yoksa uygulama yine açılmalı, ilgili menü
"Kurulum Durumu" penceresinde durumu açıklar.
"""

for _module in ("numpy", "onnxruntime"):
    try:
        __import__(_module)
    except Exception:
        pass
