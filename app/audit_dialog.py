"""Kalite denetimi sonuç paneli — bulguya tıklayınca ilgili resme atlar."""

import os
from typing import List, Sequence

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QDialog,
                             QDialogButtonBox, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from . import audit

_SEVERITY_COLORS = {
    audit.ERROR: "#c62828",
    audit.WARNING: "#ef6c00",
    audit.INFO: "#546e7a",
}


class AuditDialog(QDialog):
    """Bulgular tablosu. Satıra çift tıklamak ilgili resmi açar."""

    findingActivated = pyqtSignal(str, int)     # (resim yolu, kutu indeksi)

    def __init__(self, parent, findings: Sequence[audit.Finding]):
        super().__init__(parent)
        self.setWindowTitle("Kalite Denetimi")
        self.setMinimumSize(860, 560)
        self._all: List[audit.Finding] = list(findings)

        layout = QVBoxLayout(self)

        summary = audit.summarize(self._all)
        header = QLabel(
            f"<span style='color:{_SEVERITY_COLORS[audit.ERROR]};'><b>"
            f"{summary.get(audit.ERROR, 0)} hata</b></span> &nbsp;·&nbsp; "
            f"<span style='color:{_SEVERITY_COLORS[audit.WARNING]};'><b>"
            f"{summary.get(audit.WARNING, 0)} uyarı</b></span> &nbsp;·&nbsp; "
            f"<span style='color:{_SEVERITY_COLORS[audit.INFO]};'>"
            f"{summary.get(audit.INFO, 0)} bilgi</span>")
        header.setStyleSheet("font-size: 14px;")
        layout.addWidget(header)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Göster:"))
        self.show_errors = QCheckBox("Hatalar", self)
        self.show_warnings = QCheckBox("Uyarılar", self)
        self.show_info = QCheckBox("Bilgi", self)
        for box in (self.show_errors, self.show_warnings, self.show_info):
            box.setChecked(True)
            box.stateChanged.connect(self._refresh)
            filters.addWidget(box)
        filters.addStretch(1)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Ağırlık", "Tür", "Resim", "Açıklama"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(self._activate)
        layout.addWidget(self.table, 1)

        hint = QLabel("Bir satıra çift tıklayın ya da seçip 'Resme Git' deyin: "
                      "ilgili resim açılır ve kutu seçilir.")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self.goto_button = QPushButton("Resme Git", self)
        self.goto_button.clicked.connect(self._activate_current)
        buttons.addButton(self.goto_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()

    # ---------------------------------------------------------------- iç
    def _visible_findings(self) -> List[audit.Finding]:
        allowed = set()
        if self.show_errors.isChecked():
            allowed.add(audit.ERROR)
        if self.show_warnings.isChecked():
            allowed.add(audit.WARNING)
        if self.show_info.isChecked():
            allowed.add(audit.INFO)
        return [f for f in self._all if f.severity in allowed]

    def _refresh(self):
        findings = self._visible_findings()
        self.table.setRowCount(len(findings))
        for row, finding in enumerate(findings):
            severity_item = QTableWidgetItem(finding.severity)
            severity_item.setForeground(QColor(_SEVERITY_COLORS.get(
                finding.severity, "#888888")))
            self.table.setItem(row, 0, severity_item)
            self.table.setItem(row, 1, QTableWidgetItem(finding.category))
            image_item = QTableWidgetItem(finding.image_name)
            image_item.setToolTip(finding.image or "")
            self.table.setItem(row, 2, image_item)
            message_item = QTableWidgetItem(finding.message)
            message_item.setToolTip(finding.message)
            self.table.setItem(row, 3, message_item)

        header = self.table.horizontalHeader()
        for column in (0, 1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

    def _row_finding(self, row: int):
        findings = self._visible_findings()
        if 0 <= row < len(findings):
            return findings[row]
        return None

    def _activate(self, item: QTableWidgetItem):
        self._emit_for_row(item.row())

    def _activate_current(self):
        self._emit_for_row(self.table.currentRow())

    def _emit_for_row(self, row: int):
        finding = self._row_finding(row)
        if finding is None or not finding.image:
            return
        if not os.path.isfile(finding.image):
            return
        self.findingActivated.emit(finding.image, finding.box_index)
