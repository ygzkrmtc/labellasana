#!/usr/bin/env python3
"""
Tüm testleri tek komutla çalıştırır.

    python -m tests.run_all

Qt kurulu değilse arayüz testleri atlanır, mantık testleri yine çalışır.
"""

import os
import sys
import unittest

# Windows konsolu/CI boru hattı varsayılan olarak cp1252 kullanır ve Türkçe
# 'ş', 'ı', 'ğ' harfleri o kod sayfasında yoktur: yazdırma anında
# UnicodeEncodeError alınır, testler geçse bile çıkış kodu 1 olur.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # Python < 3.7 veya yönlendirilmiş akış
        pass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULES = [
    "tests.test_yolo_io",
    "tests.test_dataset",
    "tests.test_imaging",
    "tests.test_inference",
    "tests.test_audit_converters",
    "tests.test_project_review",
    "tests.test_training",
    "tests.test_gui",
]


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in MODULES:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as exc:            # modül hiç yüklenemediyse görünür olsun
            print(f"[!] {name} yüklenemedi: {exc}")
            return 2

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 60)
    print(f"Toplam: {result.testsRun}   Hata: {len(result.errors)}   "
          f"Başarısız: {len(result.failures)}   Atlanan: {len(result.skipped)}")
    if result.skipped:
        reasons = {str(reason) for _, reason in result.skipped}
        for reason in sorted(reasons):
            print(f"  atlandı: {reason}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
