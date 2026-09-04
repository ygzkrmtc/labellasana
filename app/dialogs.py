"""Yardımcı diyaloglar ve küçük arayüz bileşenleri."""

from typing import Optional, Sequence

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog,
                             QComboBox, QCompleter, QDialog, QDialogButtonBox,
                             QFormLayout, QFrame, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QRadioButton, QSlider,
                             QTableWidget, QTableWidgetItem, QToolButton,
                             QVBoxLayout, QWidget)

SHORTCUTS = [
    ("— Etiketleme —", ""),
    ("W", "Yeni kutu çizme modunu aç/kapat"),
    ("E", "Yeni sınıf ekle"),
    ("C", "Seçili kutunun sınıfını değiştir"),
    ("1 … 9", "Sınıf seç (kutu seçiliyken ona atar)"),
    ("Çift tıklama / sağ tık", "Kutunun sınıfını değiştir"),
    ("Delete / Backspace", "Seçili kutuyu sil"),
    ("Ctrl + Shift + Delete", "Bu resimdeki tüm kutuları sil"),
    ("Ctrl + Z", "Son işlemi geri al"),
    ("S", "Seçili kutuyu kenarlara oturt (manyetik)"),
    ("M", "Manyetik kutu modunu aç/kapat"),
    ("Ctrl + L", "Seçili kutuyu kilitle / kilidi aç"),
    ("Ctrl + A", "Tüm kutuları seç"),
    ("Ctrl + tık", "Seçime kutu ekle / çıkar"),
    ("Shift + boşlukta sürükle", "Lastik kutuyla çoklu seçim"),
    ("Ok tuşları", "Seçili kutuyu 1 px kaydır"),
    ("Shift + ok", "10 px kaydır"),
    ("Alt + ok", "Kutuyu boyutlandır"),
    ("Ctrl + C / Ctrl + V", "Kutuları kopyala / başka resme yapıştır"),

    ("— Gezinme —", ""),
    ("D", "Kaydet ve sonraki resme geç"),
    ("A", "Kaydet ve önceki resme geç"),
    ("Ctrl + Sağ ok", "Sonraki etiketlenmemiş resim"),
    ("Ctrl + S", "Etiketleri kaydet"),
    ("Ctrl + O", "Resim klasörü aç"),

    ("— Görünüm —", ""),
    ("Fare tekerleği", "Yakınlaştır / uzaklaştır (imleç merkezli)"),
    ("Orta tuş veya Alt + sürükle", "Resmi kaydır (pan)"),
    ("Boşluğa sol tuşla sürükle", "Resmi kaydır (düzenleme modunda)"),
    ("Ctrl + 0 / Ctrl + 1", "Ekrana sığdır / gerçek boyut"),
    ("Ctrl + + / Ctrl + -", "Yakınlaştır / uzaklaştır"),
    ("H", "Kutuları gizle / göster"),
    ("G", "Kenar katmanını aç/kapat"),
    ("I", "Görüntü iyileştirme şeridini aç/kapat"),
    ("L", "Büyüteci aç/kapat"),

    ("— Araçlar —", ""),
    ("Ctrl + I", "Veri seti istatistikleri"),
    ("Ctrl + E", "Veri setini YOLOv8 formatında dışa aktar"),
    ("F5", "Kalite denetimi"),
    ("Ctrl + R", "Model ile tahmin et"),
    ("Enter", "Model önerilerini kabul et"),
    ("Esc", "Seçimi bırak / çizimi iptal et / önerileri sil"),
    ("F1", "Bu rehber"),
    ("Ctrl + Q", "Çıkış"),
]


def color_icon(color: str, size: int = 14) -> QIcon:
    """Sınıf listesinde kullanılan renk kutucuğu."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


class ShortcutsDialog(QDialog):
    """F1 ile açılan kısayol rehberi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Klavye Kısayolları")
        self.setMinimumSize(560, 620)

        layout = QVBoxLayout(self)
        info = QLabel("Kısayollar ana pencere odaktayken geçerlidir.")
        layout.addWidget(info)

        table = QTableWidget(len(SHORTCUTS), 2, self)
        table.setHorizontalHeaderLabels(["Kısayol", "İşlev"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.setFocusPolicy(Qt.NoFocus)

        for row, (key, description) in enumerate(SHORTCUTS):
            key_item = QTableWidgetItem(key)
            if not description:                      # bölüm başlığı
                font = key_item.font()
                font.setBold(True)
                key_item.setFont(font)
                table.setSpan(row, 0, 1, 2)
            else:
                key_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, QTableWidgetItem(description))
            table.setItem(row, 0, key_item)

        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.resizeRowsToContents()
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class ClassDialog(QDialog):
    """Sınıf ekleme / düzenleme (ad + renk)."""

    def __init__(self, parent=None, name: str = "", color: str = "#e6194b",
                 title: str = "Sınıf Ekle"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(360)
        self._color = color

        form = QFormLayout()
        self.name_edit = QLineEdit(name, self)
        self.name_edit.setPlaceholderText("örn: aku")
        form.addRow("Sınıf adı:", self.name_edit)

        color_row = QHBoxLayout()
        self.color_button = QPushButton(self)
        self.color_button.setFixedSize(64, 24)
        self.color_button.clicked.connect(self._pick_color)
        self._refresh_color_button()
        color_row.addWidget(self.color_button)
        color_row.addStretch(1)
        form.addRow("Renk:", color_row)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        note = QLabel("Boşluklar alt çizgiye çevrilir (YOLO classes.txt kuralı).")
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name_edit.setFocus()

    def _refresh_color_button(self):
        self.color_button.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #555; border-radius: 3px;")

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Sınıf Rengi Seç")
        if color.isValid():
            self._color = color.name()
            self._refresh_color_button()

    def _accept(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Eksik bilgi", "Sınıf adı boş olamaz.")
            return
        self.accept()

    def values(self):
        return self.name_edit.text().strip(), self._color


class LabelPickerDialog(QDialog):
    """
    Kutu çizildikten sonra açılan sınıf seçici.

    Listeden seçilir ya da yeni bir ad yazılır. Yazım hatasıyla kazara sınıf
    oluşmasın diye, listede olmayan bir ad için onay istenir.
    """

    def __init__(self, parent, class_manager, initial: int = -1):
        super().__init__(parent)
        self.setWindowTitle("Kutunun sınıfı")
        self.setMinimumWidth(380)
        self._classes = class_manager

        layout = QVBoxLayout(self)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Sınıf adı yazın veya aşağıdan seçin")
        names = class_manager.names()
        if names:
            completer = QCompleter(names, self)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.name_edit.setCompleter(completer)
        layout.addWidget(self.name_edit)

        self.list = QListWidget(self)
        self.list.setIconSize(QSize(14, 14))
        for index, cls in enumerate(class_manager):
            self.list.addItem(QListWidgetItem(color_icon(cls.color),
                                              f"{index}: {cls.name}"))
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemDoubleClicked.connect(lambda _: self._accept())
        layout.addWidget(self.list)

        if 0 <= initial < len(class_manager):
            self.list.setCurrentRow(initial)
        elif len(class_manager):
            self.list.setCurrentRow(0)

        hint = QLabel("Listede olmayan bir ad yazarsanız (onayınızla) yeni sınıf "
                      "olarak eklenir. İptal ederseniz yeni çizilen kutu silinir.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.name_edit.returnPressed.connect(self._accept)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _on_row(self, row: int):
        if 0 <= row < len(self._classes):
            self.name_edit.setText(self._classes[row].name)

    def _accept(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Eksik bilgi",
                                "Bir sınıf adı yazın veya listeden seçin.")
            return
        if self._classes.index_of(name.replace(" ", "_")) < 0:
            reply = QMessageBox.question(
                self, "Yeni sınıf",
                f"'{name}' listede yok. Yeni bir sınıf olarak oluşturulsun mu?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def chosen_name(self) -> str:
        return self.name_edit.text().strip().replace(" ", "_")


class DeleteClassDialog(QDialog):
    """
    Kullanılan bir sınıf silinirken ne olacağını net biçimde sorar.

    Eski tasarımda "sadece sınıfı sil" seçeneği veri setini tutarsız
    bırakabiliyordu; artık iki seçenek de veri setini tutarlı tutar.
    """

    DELETE_BOXES = "delete"
    MOVE_BOXES = "move"

    def __init__(self, parent, class_name: str, usage: int,
                 other_classes: Sequence[str]):
        super().__init__(parent)
        self.setWindowTitle("Sınıfı sil")
        self.setMinimumWidth(480)
        self._other_classes = list(other_classes)

        layout = QVBoxLayout(self)
        headline = QLabel(f"<b>'{class_name}'</b> sınıfına ait <b>{usage}</b> kutu var. "
                          f"Bu kutulara ne yapılsın?")
        headline.setWordWrap(True)
        layout.addWidget(headline)

        self.radio_delete = QRadioButton(f"Kutuları da sil ({usage} kutu silinecek)", self)
        self.radio_move = QRadioButton("Kutuları başka bir sınıfa taşı:", self)
        self.radio_delete.setChecked(True)
        layout.addWidget(self.radio_delete)

        move_row = QHBoxLayout()
        move_row.addWidget(self.radio_move)
        self.target_combo = QComboBox(self)
        for name in self._other_classes:
            self.target_combo.addItem(name)
        self.target_combo.setEnabled(False)
        move_row.addWidget(self.target_combo, 1)
        layout.addLayout(move_row)

        if not self._other_classes:
            self.radio_move.setEnabled(False)
            self.radio_move.setToolTip("Taşınacak başka sınıf yok.")
        self.radio_move.toggled.connect(self.target_combo.setEnabled)

        note = QLabel(
            "YOLO'da sınıf kimliği = classes.txt satır numarasıdır. Silme "
            "işleminden sonra diskteki tüm etiket dosyaları otomatik olarak "
            "yeniden numaralandırılır; veri setiniz tutarlı kalır.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Sil")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        if self.radio_move.isChecked() and self._other_classes:
            return {"action": self.MOVE_BOXES,
                    "target_name": self.target_combo.currentText()}
        return {"action": self.DELETE_BOXES, "target_name": None}


class NoticeBar(QFrame):
    """
    Tuvalin üstünde beliren uyarı şeridi.

    Kullanıcıyı sıkışmış bir durumdan çıkarmak için kullanılır: sahipsiz
    kutular, onay bekleyen model etiketleri gibi.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setVisible(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self._buttons = []
        self._layout = layout

        self.close_button = QToolButton(self)
        self.close_button.setText("✕")
        self.close_button.setAutoRaise(True)
        self.close_button.clicked.connect(lambda: self.setVisible(False))
        layout.addWidget(self.close_button)

    def show_notice(self, text: str, actions=(), tone: str = "warning") -> None:
        for button in self._buttons:
            self._layout.removeWidget(button)
            button.deleteLater()
        self._buttons = []

        colors = {"warning": ("#4a3b00", "#ffd479"),
                  "error": ("#4a1414", "#ff9a9a"),
                  "info": ("#0f3350", "#9ecbff")}
        background, foreground = colors.get(tone, colors["warning"])
        self.setStyleSheet(
            f"QFrame {{ background: {background}; border: 1px solid {foreground}; "
            f"border-radius: 4px; }} QLabel {{ color: {foreground}; }}")
        self.label.setText(text)

        for caption, callback in actions:
            button = QPushButton(caption, self)
            button.clicked.connect(callback)
            self._layout.insertWidget(self._layout.count() - 1, button)
            self._buttons.append(button)
        self.setVisible(True)

    def hide_notice(self) -> None:
        self.setVisible(False)


class EnhancementBar(QWidget):
    """
    Görüntü iyileştirme şeridi.

    Yalnızca EKRANI etkiler: dosya da etiket koordinatları da değişmez.
    Karanlık, ters ışıklı fotoğraflarda nesne sınırını görünür kılar.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None, backend: str = "numpy"):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.brightness = self._slider(-100, 100, 0)
        self.contrast = self._slider(-100, 100, 0)
        self.gamma = self._slider(20, 300, 100)      # 100 = 1.00
        self.clahe = QCheckBox("CLAHE", self)
        self.clahe.setToolTip("Yerel kontrast eşitleme: karanlık bölgelerdeki "
                              "ayrıntıyı ortaya çıkarır.")

        for caption, widget in (("Parlaklık", self.brightness),
                                ("Kontrast", self.contrast),
                                ("Gama", self.gamma)):
            layout.addWidget(QLabel(caption))
            layout.addWidget(widget, 1)
        layout.addWidget(self.clahe)

        self.reset_button = QPushButton("Sıfırla", self)
        self.reset_button.clicked.connect(self.reset)
        layout.addWidget(self.reset_button)

        self.backend_label = QLabel(backend)
        self.backend_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.backend_label)

        for widget in (self.brightness, self.contrast, self.gamma):
            widget.valueChanged.connect(lambda _: self.changed.emit())
        self.clahe.stateChanged.connect(lambda _: self.changed.emit())

    def _slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setMinimumWidth(90)
        return slider

    def reset(self):
        for widget, value in ((self.brightness, 0), (self.contrast, 0),
                              (self.gamma, 100)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.clahe.blockSignals(True)
        self.clahe.setChecked(False)
        self.clahe.blockSignals(False)
        self.changed.emit()

    def values(self) -> dict:
        return {"brightness": self.brightness.value(),
                "contrast": self.contrast.value(),
                "gamma": self.gamma.value() / 100.0,
                "clahe": self.clahe.isChecked()}

    def set_values(self, brightness: int = 0, contrast: int = 0,
                   gamma: float = 1.0, clahe: bool = False) -> None:
        for widget, value in ((self.brightness, int(brightness)),
                              (self.contrast, int(contrast)),
                              (self.gamma, int(round(gamma * 100)))):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.clahe.blockSignals(True)
        self.clahe.setChecked(bool(clahe))
        self.clahe.blockSignals(False)


def ask_choice(parent, title: str, text: str,
               options: Sequence[str]) -> Optional[int]:
    """Küçük bir seçim penceresi; seçilen seçeneğin indeksini döndürür."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(text)
    label.setWordWrap(True)
    layout.addWidget(label)

    radios = []
    for index, option in enumerate(options):
        radio = QRadioButton(option, dialog)
        radio.setChecked(index == 0)
        radios.append(radio)
        layout.addWidget(radio)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec_() != QDialog.Accepted:
        return None
    for index, radio in enumerate(radios):
        if radio.isChecked():
            return index
    return None
