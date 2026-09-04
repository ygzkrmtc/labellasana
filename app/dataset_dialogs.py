"""Veri seti dışa aktarma sihirbazı ve istatistik paneli."""

import os
from typing import Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from .dataset import HEATMAP_GRID
from .dialogs import color_icon


class ExportDialog(QDialog):
    """train/val/test bölmesi + data.yaml üretimi için ayarlar."""

    def __init__(self, parent, default_dir: str, image_count: int):
        super().__init__(parent)
        self.setWindowTitle("Veri Setini Dışa Aktar")
        self.setMinimumWidth(520)
        self._image_count = image_count

        layout = QVBoxLayout(self)

        intro = QLabel(
            f"{image_count} resim YOLOv8'in beklediği klasör yapısına kopyalanacak "
            f"ve <code>data.yaml</code> üretilecek. Kaynak dosyalarınıza dokunulmaz.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ---- hedef klasör
        target_box = QGroupBox("Hedef klasör", self)
        target_layout = QHBoxLayout(target_box)
        self.dir_edit = QLineEdit(default_dir, self)
        browse = QPushButton("Gözat...", self)
        browse.clicked.connect(self._browse)
        target_layout.addWidget(self.dir_edit, 1)
        target_layout.addWidget(browse)
        layout.addWidget(target_box)

        # ---- bölme oranları
        split_box = QGroupBox("Bölme oranları (%)", self)
        form = QFormLayout(split_box)
        self.train_spin = self._spin(70)
        self.val_spin = self._spin(20)
        self.test_spin = self._spin(10)
        form.addRow("Eğitim (train):", self.train_spin)
        form.addRow("Doğrulama (val):", self.val_spin)
        form.addRow("Test:", self.test_spin)

        self.sum_label = QLabel()
        form.addRow("Toplam:", self.sum_label)

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip(
            "Aynı tohum her zaman aynı bölmeyi üretir; sonuçlarınız tekrarlanabilir olur.")
        form.addRow("Rastgelelik tohumu:", self.seed_spin)
        layout.addWidget(split_box)

        for spin in (self.train_spin, self.val_spin, self.test_spin):
            spin.valueChanged.connect(self._update_sum)

        # ---- seçenekler
        options = QGroupBox("Seçenekler", self)
        option_layout = QVBoxLayout(options)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Kopyala (güvenli)", "copy")
        self.mode_combo.addItem("Sabit bağ / hardlink (aynı diskte yer kaplamaz)", "hardlink")
        option_layout.addWidget(self.mode_combo)

        self.include_unlabeled = QCheckBox(
            "Etiketsiz resimleri de dahil et (arka plan örneği olarak)", self)
        self.include_unlabeled.setToolTip(
            "YOLO, nesne içermeyen resimleri boş .txt ile arka plan örneği sayar. "
            "Az miktarda arka plan hatalı pozitifleri azaltır.")
        option_layout.addWidget(self.include_unlabeled)
        layout.addWidget(options)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Dışa Aktar")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_sum()

    def _spin(self, value: int) -> QSpinBox:
        spin = QSpinBox(self)
        spin.setRange(0, 100)
        spin.setSuffix(" %")
        spin.setValue(value)
        return spin

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Veri setinin yazılacağı klasör", self.dir_edit.text())
        if folder:
            self.dir_edit.setText(folder)

    def _update_sum(self):
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        ok = total == 100
        self.sum_label.setText(f"{total} %" + ("" if ok else "   ← 100 olmalı"))
        self.sum_label.setStyleSheet(
            "color: #2e7d32; font-weight: bold;" if ok else "color: #c62828; font-weight: bold;")
        approx = [int(round(self._image_count * s.value() / 100.0))
                  for s in (self.train_spin, self.val_spin, self.test_spin)]
        self.note.setText(
            f"Yaklaşık dağılım: {approx[0]} eğitim · {approx[1]} doğrulama · {approx[2]} test. "
            "Bölme, sınıf bileşimleri korunacak şekilde katmanlı yapılır.")

    def _accept(self):
        if self.train_spin.value() + self.val_spin.value() + self.test_spin.value() != 100:
            QMessageBox.warning(self, "Oranlar", "Oranların toplamı 100 olmalı.")
            return
        if self.train_spin.value() == 0:
            QMessageBox.warning(self, "Oranlar", "Eğitim payı sıfır olamaz.")
            return
        folder = self.dir_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "Hedef", "Bir hedef klasör seçin.")
            return
        if os.path.isdir(folder) and os.listdir(folder):
            reply = QMessageBox.question(
                self, "Klasör boş değil",
                "Seçilen klasörde başka dosyalar var. Aynı adlı dosyalar üzerine "
                "yazılacak. Devam edilsin mi?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def values(self) -> dict:
        return {
            "out_dir": self.dir_edit.text().strip(),
            "ratios": (self.train_spin.value(), self.val_spin.value(), self.test_spin.value()),
            "seed": self.seed_spin.value(),
            "mode": self.mode_combo.currentData(),
            "include_unlabeled": self.include_unlabeled.isChecked(),
        }


class StatsDialog(QDialog):
    """Veri setinin sağlık raporu: sınıf dağılımı, kutu boyutları, konum ısı haritası."""

    def __init__(self, parent, stats: dict, class_names: Sequence[str],
                 class_colors: Sequence[str]):
        super().__init__(parent)
        self.setWindowTitle("Veri Seti İstatistikleri")
        self.setMinimumSize(760, 560)
        self._class_names = list(class_names)

        layout = QVBoxLayout(self)
        layout.addWidget(self._summary_widget(stats))

        columns = QHBoxLayout()
        columns.addWidget(self._class_table(stats, class_names, class_colors), 3)
        columns.addWidget(self._right_column(stats), 2)
        layout.addLayout(columns, 1)

        warning = self._warning_text(stats)
        if warning:
            label = QLabel(warning)
            label.setWordWrap(True)
            label.setStyleSheet(
                "background: #4a3b00; color: #ffd479; padding: 8px; border-radius: 4px;")
            layout.addWidget(label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- parçalar
    def _summary_widget(self, stats: dict) -> QWidget:
        box = QGroupBox("Özet", self)
        grid = QGridLayout(box)
        boxes_per_image = stats["boxes_per_image"]
        average = (sum(boxes_per_image) / len(boxes_per_image)) if boxes_per_image else 0.0
        cells = [
            ("Resim", stats["images"]),
            ("Etiketli", stats["labeled"]),
            ("Boş (arka plan)", stats["empty"]),
            ("Etiketsiz", stats["missing"]),
            ("Toplam kutu", stats["total_boxes"]),
            ("Resim başına ort.", f"{average:.1f}"),
        ]
        for column, (title, value) in enumerate(cells):
            value_label = QLabel(str(value))
            value_label.setStyleSheet("font-size: 20px; font-weight: bold;")
            value_label.setAlignment(Qt.AlignCenter)
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #888; font-size: 11px;")
            title_label.setAlignment(Qt.AlignCenter)
            grid.addWidget(value_label, 0, column)
            grid.addWidget(title_label, 1, column)
        return box

    def _class_table(self, stats: dict, class_names, class_colors) -> QWidget:
        box = QGroupBox("Sınıf dağılımı", self)
        layout = QVBoxLayout(box)
        per_class = stats["per_class"]
        total = max(1, stats["total_boxes"])

        table = QTableWidget(len(class_names), 4, self)
        table.setHorizontalHeaderLabels(["Sınıf", "Kutu", "Pay", "Ort. alan"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)

        for row, name in enumerate(class_names):
            count = per_class.get(row, 0)
            share = 100.0 * count / total
            mean_area = stats["mean_area"].get(row, 0.0)
            name_item = QTableWidgetItem(f"{row}: {name}")
            if row < len(class_colors):
                name_item.setIcon(color_icon(class_colors[row], 12))
            table.setItem(row, 0, name_item)
            for column, text in ((1, str(count)), (2, f"%{share:.1f}"),
                                 (3, f"%{mean_area * 100:.2f}")):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, column, cell)

        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(table)

        note = QLabel("Ort. alan, kutunun resim alanına oranıdır.")
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)
        return box

    def _right_column(self, stats: dict) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        size_box = QGroupBox("Nesne büyüklüğü", self)
        size_layout = QVBoxLayout(size_box)
        buckets = stats["size_buckets"]
        total = max(1, sum(buckets.values()))
        for key, explanation in (("küçük", "resmin %0.25'inden küçük"),
                                 ("orta", "%0.25 – %2.25 arası"),
                                 ("büyük", "%2.25'ten büyük")):
            count = buckets.get(key, 0)
            row = QLabel(f"<b>{key.capitalize()}</b>: {count}  "
                         f"(%{100.0 * count / total:.1f})<br>"
                         f"<span style='color:#888; font-size:11px;'>{explanation}</span>")
            size_layout.addWidget(row)
        layout.addWidget(size_box)

        heat_box = QGroupBox("Nesne konumu ısı haritası", self)
        heat_layout = QVBoxLayout(heat_box)
        heat_label = QLabel()
        heat_label.setAlignment(Qt.AlignCenter)
        heat_label.setPixmap(self._heatmap_pixmap(stats["heatmap"]))
        heat_label.setFrameShape(QFrame.StyledPanel)
        heat_layout.addWidget(heat_label)
        hint = QLabel("Nesneler hep aynı bölgede toplanıyorsa veri setiniz "
                      "o konuma göre yanlıdır.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        heat_layout.addWidget(hint)
        layout.addWidget(heat_box)
        layout.addStretch(1)
        return container

    def _heatmap_pixmap(self, heatmap) -> QPixmap:
        image = QImage(HEATMAP_GRID, HEATMAP_GRID, QImage.Format_RGB32)
        peak = max((max(row) for row in heatmap), default=0)
        for y in range(HEATMAP_GRID):
            for x in range(HEATMAP_GRID):
                value = heatmap[y][x]
                if peak <= 0 or value == 0:
                    color = QColor(28, 30, 38)
                else:
                    t = value / peak
                    # mavi -> camgöbeği -> sarı -> kırmızı
                    hue = (1.0 - t) * 0.66
                    color = QColor.fromHsvF(max(0.0, min(0.66, hue)), 0.85,
                                            0.35 + 0.65 * t)
                image.setPixel(x, y, color.rgb())
        return QPixmap.fromImage(image).scaled(
            240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _warning_text(self, stats: dict) -> str:
        messages = []
        per_class = stats["per_class"]
        counts = [v for v in per_class.values() if v > 0]
        if len(counts) > 1 and max(counts) / min(counts) >= 10:
            messages.append(
                f"Sınıf dengesizliği yüksek (en kalabalık sınıf en seyrekten "
                f"{max(counts) / min(counts):.0f} kat fazla). Model seyrek sınıfı "
                f"öğrenmekte zorlanabilir.")
        if stats["missing"] > 0:
            messages.append(f"{stats['missing']} resim hiç etiketlenmemiş; "
                            f"dışa aktarımda arka plan sayılırlar.")
        if stats["unknown_ids"]:
            messages.append(f"classes.txt'te karşılığı olmayan sınıf kimlikleri: "
                            f"{stats['unknown_ids']}")
        unused = [f"{i}: {name}" for i, name in enumerate(self._class_names)
                  if per_class.get(i, 0) == 0]
        if unused:
            messages.append("Hiç kullanılmayan sınıflar: " + ", ".join(unused) +
                            ". Bunlar data.yaml'a yine de yazılır; gereksizse silin.")
        return "  ".join(messages)
