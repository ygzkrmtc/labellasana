"""Model destekli ön-etiketleme arayüzü."""

import os
from typing import Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
                             QRadioButton, QVBoxLayout)


class PredictDialog(QDialog):
    """
    Tahmin ayarları.

    Güven eşiği yüksekse az ama isabetli, düşükse çok ama gürültülü kutu gelir.
    Etiketlemede genelde 0.25-0.40 arası iyi bir başlangıçtır: fazla kutu silmek,
    eksik kutuyu fark etmekten kolaydır.
    """

    SCOPE_CURRENT = "current"
    SCOPE_UNLABELED = "unlabeled"
    SCOPE_ALL = "all"

    def __init__(self, parent, model_path: str, model_classes: Sequence[str],
                 project_classes: Sequence[str], mapping: dict,
                 unlabeled_count: int, total_count: int,
                 conf: float = 0.30, iou: float = 0.45):
        super().__init__(parent)
        self.setWindowTitle("Model ile Ön-Etiketleme")
        self.setMinimumWidth(540)

        layout = QVBoxLayout(self)

        model_box = QGroupBox("Model", self)
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(QLabel(f"<b>{os.path.basename(model_path)}</b>"))
        if model_classes:
            matched = len(mapping)
            unmatched = [name for index, name in enumerate(model_classes)
                         if index not in mapping]
            text = (f"Model sınıfları: {', '.join(model_classes[:8])}"
                    f"{'...' if len(model_classes) > 8 else ''}<br>"
                    f"Projeyle eşleşen: <b>{matched}/{len(model_classes)}</b>")
            if unmatched:
                text += ("<br><span style='color:#ef6c00;'>Eşleşmeyen "
                         f"({', '.join(unmatched[:6])}) sınıfların kutuları "
                         "atlanacak. Aynı adla sınıf eklerseniz eşleşirler."
                         "</span>")
        else:
            text = ("Model sınıf adı taşımıyor; kutular <b>aktif sınıfa</b> "
                    "atanacak.")
        info = QLabel(text)
        info.setWordWrap(True)
        model_layout.addWidget(info)
        layout.addWidget(model_box)

        scope_box = QGroupBox("Kapsam", self)
        scope_layout = QVBoxLayout(scope_box)
        self.scope_group = QButtonGroup(self)
        self.radio_current = QRadioButton("Sadece açık olan resim", self)
        self.radio_unlabeled = QRadioButton(
            f"Etiketlenmemiş resimler ({unlabeled_count})", self)
        self.radio_all = QRadioButton(f"Tüm resimler ({total_count})", self)
        self.radio_current.setChecked(True)
        for index, radio in enumerate((self.radio_current, self.radio_unlabeled,
                                       self.radio_all)):
            self.scope_group.addButton(radio, index)
            scope_layout.addWidget(radio)
        self.radio_unlabeled.setEnabled(unlabeled_count > 0)
        layout.addWidget(scope_box)

        settings_box = QGroupBox("Eşikler", self)
        form = QFormLayout(settings_box)
        self.conf_spin = QDoubleSpinBox(self)
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(conf)
        form.addRow("Güven eşiği:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox(self)
        self.iou_spin.setRange(0.10, 0.90)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(iou)
        self.iou_spin.setToolTip("Üst üste binen kutuların ne kadarında birinin "
                                 "eleneceği (NMS eşiği).")
        form.addRow("Örtüşme (IoU) eşiği:", self.iou_spin)
        layout.addWidget(settings_box)

        note = QLabel(
            "Açık resimde sonuçlar <b>kesik çizgili öneri</b> olarak gösterilir; "
            "siz onaylamadan hiçbir şey diske yazılmaz.<br>"
            "Toplu tahminde sonuçlar diske yazılır ama o resimler listede "
            "<b>?</b> ile işaretlenir; açıp onaylayana kadar gözden geçirilmemiş "
            "sayılırlar.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Tahmin Et")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        if self.radio_all.isChecked():
            scope = self.SCOPE_ALL
        elif self.radio_unlabeled.isChecked():
            scope = self.SCOPE_UNLABELED
        else:
            scope = self.SCOPE_CURRENT
        return {"scope": scope,
                "conf": float(self.conf_spin.value()),
                "iou": float(self.iou_spin.value())}
