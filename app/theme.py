"""
Açık ve karanlık tema paletleri.

Saatlerce ekrana bakan bir etiketleyici için karanlık tema konfor değil,
göz sağlığı meselesidir. Fusion stili üzerinden QPalette ile uygulanır;
böylece işletim sistemi temasından bağımsız ve tutarlıdır.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

DARK = "dark"
LIGHT = "light"

# Tuval arka planı temayla birlikte değişir
CANVAS_BACKGROUND = {DARK: "#1e1f22", LIGHT: "#9aa0a6"}
HINT_COLOR = {DARK: "#9aa0a6", LIGHT: "#3c4043"}


def apply(app, mode: str = DARK) -> None:
    app.setStyle("Fusion")
    app.setPalette(_dark_palette() if mode == DARK else _light_palette())
    app.setStyleSheet(_stylesheet(mode))


def _dark_palette() -> QPalette:
    palette = QPalette()
    window = QColor("#2b2d31")
    base = QColor("#232428")
    text = QColor("#e3e5e8")
    highlight = QColor("#3b82f6")

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, QColor("#2a2b30"))
    palette.setColor(QPalette.ToolTipBase, QColor("#3a3d44"))
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, window)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, QColor("#ff5252"))
    palette.setColor(QPalette.Link, highlight)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#7a7d84"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7a7d84"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#7a7d84"))
    return palette


def _light_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f2f3f5"))
    palette.setColor(QPalette.WindowText, QColor("#1f2023"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f7f8fa"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText, QColor("#1f2023"))
    palette.setColor(QPalette.Text, QColor("#1f2023"))
    palette.setColor(QPalette.Button, QColor("#f2f3f5"))
    palette.setColor(QPalette.ButtonText, QColor("#1f2023"))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor("#1a73e8"))
    palette.setColor(QPalette.Highlight, QColor("#1a73e8"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#9aa0a6"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9aa0a6"))
    return palette


def _stylesheet(mode: str) -> str:
    if mode == DARK:
        return """
            QToolTip { color: #e3e5e8; background-color: #3a3d44; border: 1px solid #4a4d54; }
            QGroupBox { border: 1px solid #3a3d44; border-radius: 4px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QListWidget::item:selected { background: #3b82f6; }
        """
    return """
        QToolTip { color: #1f2023; background-color: #ffffff; border: 1px solid #c8ccd0; }
        QGroupBox { border: 1px solid #d7dade; border-radius: 4px; margin-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
    """
