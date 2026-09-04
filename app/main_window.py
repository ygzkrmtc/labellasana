"""Ana pencere: menüler, paneller, kısayollar ve tüm iş akışları."""

import os
import time
from typing import Dict, List, Optional, Set

from PyQt5.QtCore import QPointF, QRectF, QSettings, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QImageReader, QKeySequence
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QActionGroup,
                             QApplication, QCheckBox, QFileDialog, QHBoxLayout,
                             QInputDialog, QLabel, QListWidget, QListWidgetItem,
                             QMainWindow, QMenu, QMessageBox, QProgressDialog,
                             QPushButton, QShortcut, QSplitter, QVBoxLayout,
                             QWidget)

from . import (audit, converters, dataset, imaging, inference, project, review,
               theme, training, yolo_io)
from .ai_dialogs import PredictDialog
from .train_dialog import TrainDialog
from .audit_dialog import AuditDialog
from .canvas import Canvas, Mode, qimage_to_array
from .dataset_dialogs import ExportDialog, StatsDialog
from .dialogs import (ClassDialog, DeleteClassDialog, EnhancementBar,
                      LabelPickerDialog, NoticeBar, ShortcutsDialog, color_icon)
from .models import ClassManager, color_for_index

APP_TITLE = "YOLOv8 Etiketleme Aracı"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 900)
        self.setMinimumSize(980, 620)

        # ---- durum
        self.classes = ClassManager()
        self.image_dir: Optional[str] = None
        self.label_dir: Optional[str] = None
        self.images: List[str] = []
        self.current_index: int = -1
        self._loading = False
        self._checkbox_just_toggled = False
        self._clipboard: List[tuple] = []
        self._clipboard_size = (0, 0)
        self._pending_review: Set[str] = set()
        self._session_start = time.time()
        self._session_boxes = 0
        self._detector = None
        self._model_path = ""
        self._conf = 0.30
        self._iou = 0.45
        self._theme = theme.DARK
        self.settings = QSettings("YOLO Labeler", "YOLO Etiketleyici")

        # ---- widget'lar
        self.canvas = Canvas(self.classes, self)
        self.notice = NoticeBar(self)
        self.enhance_bar = EnhancementBar(self, imaging.backend_name())
        self.enhance_bar.setVisible(False)

        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        center_layout.addWidget(self.notice)
        center_layout.addWidget(self.canvas, 1)
        center_layout.addWidget(self.enhance_bar)

        self._build_left_panel()
        self._build_right_panel()

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(center)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([260, 940, 280])
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self._build_actions_and_menus()
        self._build_statusbar()
        self._connect_signals()

        self._update_class_list()
        self._update_status()
        self._update_title()
        self._restore_settings()

    # ================================================================ sinyaller
    def _connect_signals(self):
        self.canvas.boxesChanged.connect(self._update_status)
        self.canvas.selectionChanged.connect(self._on_canvas_selection)
        self.canvas.zoomChanged.connect(lambda _: self._update_status())
        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        self.canvas.dirtyChanged.connect(lambda _: self._update_title())
        self.canvas.modeChanged.connect(self._on_mode_changed)
        self.canvas.boxCreated.connect(self._on_box_created)
        self.canvas.boxDoubleClicked.connect(self._on_box_double_clicked)
        self.canvas.contextMenuRequested.connect(self._on_context_menu)
        self.enhance_bar.changed.connect(self._on_enhancement_changed)

    # ================================================================ ayarlar
    def _restore_settings(self):
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self.settings.value("splitter")
        if splitter_state is not None:
            self.splitter.restoreState(splitter_state)

        def flag(key, action, default):
            value = self.settings.value(key, default, type=bool)
            action.setChecked(value)
            return value

        flag("autosave", self.act_autosave, True)
        flag("ask_class", self.act_ask_class, False)
        flag("return_edit", self.act_return_edit, False)
        flag("restore_last", self.act_restore_last, True)
        self.canvas.set_show_fill(flag("show_fill", self.act_show_fill, False))
        self.canvas.magnetic = flag("magnetic", self.act_magnetic, False)
        self.canvas.loupe_enabled = flag("loupe", self.act_loupe, False)

        self._conf = float(self.settings.value("conf", 0.30, type=float))
        self._iou = float(self.settings.value("iou", 0.45, type=float))
        self._model_path = self.settings.value("model_path", "", type=str)

        stored_theme = self.settings.value("theme", theme.DARK, type=str)
        self._apply_theme(stored_theme if stored_theme in (theme.DARK, theme.LIGHT)
                          else theme.DARK)

        if self.act_restore_last.isChecked():
            last = self.settings.value("last_folder", "", type=str)
            if last and os.path.isdir(last) and yolo_io.list_images(last):
                self.load_folder(last)
                saved_label_dir = self.settings.value("last_label_dir", "", type=str)
                if (saved_label_dir and os.path.isdir(saved_label_dir)
                        and saved_label_dir != self.label_dir):
                    self.label_dir = saved_label_dir
                    self._load_classes_from_disk()
                    self._reload_review_queue()
                    self._refresh_image_list_marks()

    def _save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
        self.settings.setValue("autosave", self.act_autosave.isChecked())
        self.settings.setValue("ask_class", self.act_ask_class.isChecked())
        self.settings.setValue("return_edit", self.act_return_edit.isChecked())
        self.settings.setValue("restore_last", self.act_restore_last.isChecked())
        self.settings.setValue("show_fill", self.act_show_fill.isChecked())
        self.settings.setValue("magnetic", self.act_magnetic.isChecked())
        self.settings.setValue("loupe", self.act_loupe.isChecked())
        self.settings.setValue("theme", self._theme)
        self.settings.setValue("conf", self._conf)
        self.settings.setValue("iou", self._iou)
        self.settings.setValue("model_path", self._model_path or "")
        self.settings.setValue("last_folder", self.image_dir or "")
        self.settings.setValue("last_label_dir", self.label_dir or "")

    # ================================================================ paneller
    def _build_left_panel(self):
        self.left_panel = QWidget(self)
        layout = QVBoxLayout(self.left_panel)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Resimler")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self.only_unlabeled = QCheckBox("Sadece etiketlenmemişler", self.left_panel)
        self.only_unlabeled.stateChanged.connect(lambda _: self._apply_image_filter())
        layout.addWidget(self.only_unlabeled)

        self.image_list = QListWidget(self.left_panel)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        layout.addWidget(self.image_list, 1)

        self.image_count_label = QLabel("0 resim")
        self.image_count_label.setStyleSheet("color: #888;")
        layout.addWidget(self.image_count_label)

    def _build_right_panel(self):
        self.right_panel = QWidget(self)
        layout = QVBoxLayout(self.right_panel)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Sınıflar")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self.class_list = QListWidget(self.right_panel)
        self.class_list.setIconSize(QSize(14, 14))
        self.class_list.currentRowChanged.connect(self._on_class_row_changed)
        self.class_list.itemClicked.connect(self._on_class_clicked)
        self.class_list.itemDoubleClicked.connect(lambda _: self.edit_class())
        self.class_list.itemChanged.connect(self._on_class_item_changed)
        layout.addWidget(self.class_list, 1)

        self.btn_add_class = QPushButton("＋  Yeni Sınıf Ekle  (E)")
        self.btn_add_class.setMinimumHeight(32)
        self.btn_add_class.setStyleSheet("font-weight: bold;")
        self.btn_add_class.clicked.connect(self.add_class)
        layout.addWidget(self.btn_add_class)

        buttons = QHBoxLayout()
        self.btn_edit_class = QPushButton("Düzenle")
        self.btn_del_class = QPushButton("Sil")
        self.btn_edit_class.clicked.connect(self.edit_class)
        self.btn_del_class.clicked.connect(self.delete_class)
        for button in (self.btn_edit_class, self.btn_del_class):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        hint = QLabel("Kutu çizince sınıfı sorulur. Var olan bir kutunun sınıfını "
                      "değiştirmek için çift tıklayın, sağ tıklayın, 1-9 tuşlarına "
                      "basın ya da seçiliyken listeden tıklayın. Soldaki kutucuk "
                      "o sınıfın kutularını gizler.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        layout.addWidget(QLabel("Bu resimdeki kutular:"))
        self.box_list = QListWidget(self.right_panel)
        self.box_list.setMaximumHeight(200)
        self.box_list.setIconSize(QSize(12, 12))
        self.box_list.currentRowChanged.connect(self._on_box_row_changed)
        self.box_list.itemDoubleClicked.connect(self._on_box_item_double_clicked)
        layout.addWidget(self.box_list)

    # ================================================================ menüler
    def _build_actions_and_menus(self):
        menubar = self.menuBar()

        # ------------------------------------------------ Dosya
        file_menu = menubar.addMenu("&Dosya")
        self.act_open = self._action("Klasör Aç...", "Ctrl+O", self.open_folder)
        self.act_open_project = self._action("Proje Aç...", None, self.open_project)
        self.act_save_project = self._action("Projeyi Kaydet...", None, self.save_project)
        self.act_label_dir = self._action("Etiket Klasörünü Değiştir...", None,
                                          self.change_label_dir)
        self.act_save = self._action("Kaydet", "Ctrl+S",
                                     lambda: self.save_current(silent=False))
        self.act_autosave = self._checkable("Resim değişince otomatik kaydet", True)
        self.act_quit = self._action("Çıkış", "Ctrl+Q", self.close)

        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_open_project)
        file_menu.addAction(self.act_save_project)
        file_menu.addAction(self.act_label_dir)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_autosave)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        # ------------------------------------------------ Düzen
        edit_menu = menubar.addMenu("Dü&zen")
        self.act_draw = self._checkable("Yeni Kutu Çiz", False, "W")
        self.act_draw.triggered.connect(lambda _: self.canvas.toggle_mode())
        self.act_undo = self._action("Geri Al", "Ctrl+Z", self.undo)
        self.act_delete = self._action("Seçili Kutuyu Sil", QKeySequence(Qt.Key_Delete),
                                       self.delete_selected_box)
        self.act_clear_boxes = self._action("Bu Resimdeki Tüm Kutuları Sil",
                                            "Ctrl+Shift+Delete", self.clear_all_boxes)
        self.act_select_all = self._action("Tüm Kutuları Seç", "Ctrl+A", self.select_all_boxes)
        self.act_lock = self._action("Seçili Kutuyu Kilitle / Aç", "Ctrl+L",
                                     self.toggle_lock)
        self.act_copy_boxes = self._action("Kutuları Kopyala", "Ctrl+C", self.copy_boxes)
        self.act_paste_boxes = self._action("Kutuları Yapıştır", "Ctrl+V", self.paste_boxes)
        self.act_add_class = self._action("Yeni Sınıf Ekle...", "E", self.add_class)
        self.act_change_class = self._action("Seçili Kutunun Sınıfını Değiştir...", "C",
                                             self.change_class_of_selection)
        self.act_snap = self._action("Seçili Kutuyu Kenarlara Oturt", "S",
                                     self.snap_selection)
        self.act_magnetic = self._checkable("Manyetik kutu (çizerken oturt)", False, "M")
        self.act_magnetic.triggered.connect(self._toggle_magnetic)
        self.act_prev = self._action("Önceki Resim (Kaydet)", "A", self.go_prev)
        self.act_next = self._action("Sonraki Resim (Kaydet)", "D", self.go_next)
        self.act_next_unlabeled = self._action("Sonraki Etiketlenmemiş Resim",
                                               "Ctrl+Right", self.go_next_unlabeled)
        self.act_ask_class = self._checkable("Her kutu çiziminde sınıfı sor", False)
        self.act_ask_class.setToolTip(
            "Kapalıyken yeni kutulara aktif sınıf atanır; hiç sınıf yoksa yine sorulur.")
        self.act_return_edit = self._checkable("Her kutudan sonra çizim modundan çık", False)

        for action in (self.act_draw, self.act_undo, self.act_delete,
                       self.act_clear_boxes):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        for action in (self.act_select_all, self.act_lock, self.act_copy_boxes,
                       self.act_paste_boxes):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        for action in (self.act_add_class, self.act_change_class, self.act_snap,
                       self.act_magnetic):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        for action in (self.act_prev, self.act_next, self.act_next_unlabeled):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_ask_class)
        edit_menu.addAction(self.act_return_edit)

        # ------------------------------------------------ Görünüm
        view_menu = menubar.addMenu("&Görünüm")
        self.act_zoom_in = QAction("Yakınlaştır", self)
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl++"), QKeySequence("Ctrl+=")])
        self.act_zoom_in.triggered.connect(lambda: self.canvas.zoom_by(1.25))
        self.act_zoom_out = self._action("Uzaklaştır", "Ctrl+-",
                                         lambda: self.canvas.zoom_by(1 / 1.25))
        self.act_fit = self._action("Ekrana Sığdır", "Ctrl+0", self.canvas.fit_to_window)
        self.act_actual = self._action("Gerçek Boyut (%100)", "Ctrl+1",
                                       self.canvas.zoom_reset_100)
        self.act_toggle_boxes = self._checkable("Kutuları Gizle", False, "H")
        self.act_toggle_boxes.triggered.connect(self._toggle_boxes_visible)
        self.act_show_fill = self._checkable("Kutuların içini boya", False)
        self.act_show_fill.triggered.connect(
            lambda checked: self.canvas.set_show_fill(checked))
        self.act_edges = self._checkable("Kenar katmanı", False, "G")
        self.act_edges.triggered.connect(self._toggle_edges)
        self.act_enhance = self._checkable("Görüntü iyileştirme şeridi", False, "I")
        self.act_enhance.triggered.connect(self._toggle_enhance_bar)
        self.act_loupe = self._checkable("Büyüteç", False, "L")
        self.act_loupe.triggered.connect(self._toggle_loupe)
        self.act_restore_last = self._checkable("Açılışta son klasörü yeniden aç", True)

        self.act_theme_dark = self._checkable("Karanlık tema", True)
        self.act_theme_light = self._checkable("Açık tema", False)
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        theme_group.addAction(self.act_theme_dark)
        theme_group.addAction(self.act_theme_light)
        self.act_theme_dark.triggered.connect(lambda: self._apply_theme(theme.DARK))
        self.act_theme_light.triggered.connect(lambda: self._apply_theme(theme.LIGHT))

        for action in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in (self.act_toggle_boxes, self.act_show_fill, self.act_loupe):
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in (self.act_enhance, self.act_edges):
            view_menu.addAction(action)
        view_menu.addSeparator()
        view_menu.addAction(self.act_theme_dark)
        view_menu.addAction(self.act_theme_light)
        view_menu.addSeparator()
        view_menu.addAction(self.act_restore_last)

        # Eksik kütüphane yüzünden menüyü GRİLEŞTİRMİYORUZ: tıklayınca hiçbir şey
        # olmaması, kullanıcıya "bozuk" gibi görünür. Bunun yerine tıklandığında
        # neyin eksik olduğunu ve tam kurulum komutunu söyleyen bir pencere açılır.
        if not imaging.available():
            for action in (self.act_enhance, self.act_edges, self.act_magnetic,
                           self.act_snap):
                action.setToolTip("Bu özellik için numpy gerekli: pip install numpy")

        # ------------------------------------------------ Araçlar
        tools_menu = menubar.addMenu("&Araçlar")
        self.act_stats = self._action("Veri Seti İstatistikleri...", "Ctrl+I",
                                      self.show_stats)
        self.act_audit = self._action("Kalite Denetimi...", "F5", self.run_audit)
        self.act_export = self._action("Veri Setini Dışa Aktar (YOLOv8)...", "Ctrl+E",
                                       self.export_dataset)
        self.act_coco_export = self._action("COCO JSON Olarak Dışa Aktar...", None,
                                            self.export_coco)
        self.act_coco_import = self._action("COCO JSON İçe Aktar...", None,
                                            self.import_coco)
        self.act_voc_export = self._action("Pascal VOC Olarak Dışa Aktar...", None,
                                           self.export_voc)
        self.act_voc_import = self._action("Pascal VOC İçe Aktar...", None,
                                           self.import_voc)
        self.act_restore_backup = self._action("Bu Resmi Yedekten Geri Al...", None,
                                               self.restore_from_backup)
        for action in (self.act_stats, self.act_audit):
            tools_menu.addAction(action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_export)
        tools_menu.addSeparator()
        for action in (self.act_coco_export, self.act_coco_import,
                       self.act_voc_export, self.act_voc_import):
            tools_menu.addAction(action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_restore_backup)

        # ------------------------------------------------ Yapay Zekâ
        ai_menu = menubar.addMenu("&Yapay Zekâ")
        self.act_load_model = self._action("ONNX Modeli Yükle...", None, self.load_model)
        self.act_predict = self._action("Tahmin Et...", "Ctrl+R", self.predict)
        self.act_accept_proposals = self._action("Önerileri Kabul Et", None,
                                                 self.accept_proposals)
        self.act_discard_proposals = self._action("Önerileri Sil", None,
                                                  self.discard_proposals)
        self.act_train = self._action("Model Eğit...", None, self.train_model)
        self.act_env_status = self._action("Kurulum Durumu...", None,
                                           self.show_environment_status)

        ai_menu.addAction(self.act_train)
        self.models_menu = ai_menu.addMenu("Modeller")
        self.models_menu.aboutToShow.connect(self._rebuild_models_menu)
        ai_menu.addSeparator()
        for action in (self.act_load_model, self.act_predict):
            ai_menu.addAction(action)
        ai_menu.addSeparator()
        ai_menu.addAction(self.act_accept_proposals)
        ai_menu.addAction(self.act_discard_proposals)
        ai_menu.addSeparator()
        ai_menu.addAction(self.act_env_status)

        if not inference.available():
            reason = inference.unavailable_reason()
            for action in (self.act_load_model, self.act_predict):
                action.setToolTip(f"Kullanılamıyor: {reason}")

        # ------------------------------------------------ Yardım
        help_menu = menubar.addMenu("&Yardım")
        self.act_shortcuts = self._action("Kısayollar", "F1", self.show_shortcuts)
        self.act_about = self._action("Hakkında", None, self.show_about)
        help_menu.addAction(self.act_shortcuts)
        help_menu.addAction(self.act_about)

        # ------------------------------------------------ menüsüz kısayollar
        self.sc_backspace = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        self.sc_backspace.activated.connect(self.delete_selected_box)
        self.sc_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.sc_escape.activated.connect(self._on_escape)
        self.sc_return = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.sc_return.activated.connect(self.accept_proposals)
        self.sc_enter = QShortcut(QKeySequence(Qt.Key_Enter), self)
        self.sc_enter.activated.connect(self.accept_proposals)

    def _action(self, caption, shortcut, slot) -> QAction:
        action = QAction(caption, self)
        if shortcut is not None:
            action.setShortcut(shortcut if isinstance(shortcut, QKeySequence)
                               else QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def _checkable(self, caption, checked: bool, shortcut=None) -> QAction:
        action = QAction(caption, self)
        action.setCheckable(True)
        action.setChecked(checked)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        return action

    def _build_statusbar(self):
        self.status = self.statusBar()
        self.lbl_mode = QLabel("Mod: Düzenle")
        self.lbl_active_class = QLabel("Aktif sınıf: -")
        self.lbl_image = QLabel("Resim: -")
        self.lbl_zoom = QLabel("Zoom: -")
        self.lbl_cursor = QLabel("x: -, y: -")
        self.lbl_boxes = QLabel("Kutu: 0")
        self.lbl_session = QLabel("Oturum: 0 kutu")
        for widget in (self.lbl_mode, self.lbl_active_class, self.lbl_image,
                       self.lbl_zoom, self.lbl_cursor, self.lbl_boxes,
                       self.lbl_session):
            self.status.addPermanentWidget(widget)

    # ================================================================ tema
    def _apply_theme(self, mode: str):
        self._theme = mode
        application = QApplication.instance()
        if application is not None:
            theme.apply(application, mode)
        self.canvas.setBackgroundBrush(QColor(theme.CANVAS_BACKGROUND[mode]))
        self.act_theme_dark.setChecked(mode == theme.DARK)
        self.act_theme_light.setChecked(mode == theme.LIGHT)

    # ================================================================ klasör
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Resim klasörü seç", self.image_dir or "")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str) -> bool:
        if self.canvas.is_dirty() and not self._maybe_save_before_leaving():
            return False

        images = yolo_io.list_images(folder)
        valid, broken = [], 0
        for path in images:
            reader = QImageReader(path)
            reader.setDecideFormatFromContent(True)
            if reader.canRead() and not reader.size().isEmpty():
                valid.append(path)
            else:
                broken += 1

        if not valid:
            QMessageBox.warning(self, "Resim yok",
                                "Seçilen klasörde okunabilir resim bulunamadı.")
            return False

        stems: Dict[str, List[str]] = {}
        for path in valid:
            stems.setdefault(os.path.splitext(os.path.basename(path))[0].lower(),
                             []).append(path)
        clashes = {key: value for key, value in stems.items() if len(value) > 1}
        if clashes:
            QMessageBox.warning(
                self, "Aynı adlı dosyalar",
                "Bu klasörde adı aynı, uzantısı farklı dosyalar var "
                f"({', '.join(list(clashes)[:5])}). YOLO etiketleri dosya adına "
                "göre yazıldığı için bunlar aynı .txt dosyasını paylaşır ve "
                "birbirinin etiketlerini ezer. Yeniden adlandırmanız önerilir.")

        self.image_dir = folder
        self.label_dir = os.path.join(folder, "labels")
        self.images = valid
        self.current_index = -1
        self.canvas.clear_all()
        self.notice.hide_notice()

        self._load_classes_from_disk()
        self._reload_review_queue()
        self._populate_image_list()

        message = f"{len(valid)} resim yüklendi."
        if broken:
            message += f" ({broken} bozuk/desteklenmeyen dosya atlandı)"
        message += f"  Etiket klasörü: {self.label_dir}"
        self.status.showMessage(message, 8000)

        self._select_row(0)
        return True

    def change_label_dir(self):
        if not self.image_dir:
            QMessageBox.information(self, "Önce klasör açın",
                                    "Etiket klasörünü değiştirmek için önce bir "
                                    "resim klasörü açın.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Etiketlerin kaydedileceği klasör", self.label_dir or self.image_dir)
        if not folder:
            return
        self.label_dir = folder
        self._load_classes_from_disk()
        self._reload_review_queue()
        self._refresh_image_list_marks()
        if self.current_index >= 0:
            self._load_image_at(self.current_index, force=True)
        self.status.showMessage(f"Etiket klasörü: {folder}", 5000)

    # ================================================================ resim listesi
    def _populate_image_list(self):
        self._loading = True
        self.image_list.clear()
        for path in self.images:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.image_list.addItem(item)
        self._loading = False
        self._refresh_image_list_marks()

    def _refresh_image_list_marks(self):
        if not self.label_dir:
            return
        labeled = empty = pending = 0
        for row, path in enumerate(self.images):
            item = self.image_list.item(row)
            if item is None:
                continue
            name = os.path.basename(path)
            txt = yolo_io.find_existing_label(path, self.label_dir)
            size = 0
            if txt:
                try:
                    size = os.path.getsize(txt)
                except OSError:
                    size = 0
            if name in self._pending_review:
                prefix, pending = "?  ", pending + 1
            elif txt and size > 0:
                prefix, labeled = "✓  ", labeled + 1
            elif txt:
                prefix, empty = "○  ", empty + 1
            else:
                prefix = "      "
            item.setText(prefix + name)

        text = f"{labeled} / {len(self.images)} etiketlendi"
        if empty:
            text += f"  ·  {empty} boş"
        if pending:
            text += f"  ·  {pending} onay bekliyor"
        self.image_count_label.setText(text)
        self._apply_image_filter()

    def _apply_image_filter(self):
        """'Sadece etiketlenmemişler' filtresi: satırları gizler, listeyi bozmaz."""
        if not self.images or not self.label_dir:
            return
        only_unlabeled = self.only_unlabeled.isChecked()
        for row, path in enumerate(self.images):
            item = self.image_list.item(row)
            if item is None:
                continue
            hidden = False
            if only_unlabeled and row != self.current_index:
                hidden = yolo_io.has_labels(path, self.label_dir)
            item.setHidden(hidden)

    def _visible_rows(self) -> List[int]:
        rows = []
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if item is not None and not item.isHidden():
                rows.append(row)
        return rows

    def _select_row(self, row: int):
        if 0 <= row < self.image_list.count():
            self.image_list.setCurrentRow(row)

    def _on_image_row_changed(self, row: int):
        if self._loading or row < 0 or row >= len(self.images):
            return
        self._load_image_at(row)

    def _load_image_at(self, index: int, force: bool = False):
        if index == self.current_index and not force:
            return
        if self.current_index >= 0 and self.canvas.is_dirty():
            if self.act_autosave.isChecked():
                self.save_current(silent=True)
                saved = not self.canvas.is_dirty()
            else:
                saved = self._maybe_save_before_leaving()
            if not saved:
                self._loading = True
                self.image_list.setCurrentRow(self.current_index)
                self._loading = False
                return

        path = self.images[index]
        if not self.canvas.load_image(path):
            QMessageBox.warning(self, "Açılamadı", f"Resim okunamadı:\n{path}")
            return

        self.current_index = index
        self._apply_enhancement_to_canvas()
        self._load_labels_for_current()
        self._refresh_notice()
        self._update_box_list()
        self._update_status()
        self._update_title()
        self._apply_image_filter()

    def _load_labels_for_current(self):
        """Kayıtlı .txt varsa YOLO değerlerini piksele çevirip kutuları geri çizer."""
        path = self.images[self.current_index]
        txt = yolo_io.find_existing_label(path, self.label_dir)
        img_w, img_h = self.canvas.image_size()
        boxes = yolo_io.load_labels(txt, img_w, img_h) if txt else []

        if boxes:
            max_id = max(row[0] for row in boxes)
            before = len(self.classes)
            self.classes.ensure_capacity(max_id)
            if len(self.classes) != before:
                self._update_class_list()
                QMessageBox.information(
                    self, "Tanımsız sınıf kimliği",
                    f"'{os.path.basename(path)}' etiket dosyasında classes.txt'te "
                    f"karşılığı olmayan kimlikler var.\n\n"
                    f"{len(self.classes) - before} yer tutucu sınıf (class_N) "
                    f"oluşturuldu; 'Düzenle' ile doğru adları verebilirsiniz.")

        self.canvas.set_boxes(boxes)
        self.canvas.set_dirty(False)
        if txt and boxes:
            self.status.showMessage(
                f"{len(boxes)} kutu okundu: {os.path.basename(txt)}", 4000)

    # ================================================================ uyarı şeridi
    def _refresh_notice(self):
        orphans = self.canvas.orphan_boxes()
        if orphans:
            self.notice.show_notice(
                f"<b>{len(orphans)} kutunun sınıfı sınıf listesinde yok.</b> "
                f"Bu kutular kaydedilemez; sınıf atayın ya da silin.",
                [("Sınıf ata...", self.fix_orphan_boxes),
                 ("Kutuları sil", self.delete_orphan_boxes)],
                tone="error")
            return

        if self.current_index >= 0:
            name = os.path.basename(self.images[self.current_index])
            if name in self._pending_review:
                self.notice.show_notice(
                    "<b>Bu resmin etiketleri model tarafından üretildi.</b> "
                    "Gözden geçirip onaylayın.",
                    [("Onayla", self.confirm_review),
                     ("Etiketleri sil", self.reject_review)],
                    tone="info")
                return

        proposals = self.canvas.proposals()
        if proposals:
            self.notice.show_notice(
                f"<b>{len(proposals)} model önerisi</b> onay bekliyor. "
                f"Kabul edilene kadar diske yazılmaz.",
                [("Hepsini kabul et (Enter)", self.accept_proposals),
                 ("Hepsini sil", self.discard_proposals)],
                tone="info")
            return

        self.notice.hide_notice()

    def fix_orphan_boxes(self):
        orphans = self.canvas.orphan_boxes()
        if not orphans:
            self._refresh_notice()
            return
        class_id = self._ask_class(-1)
        if class_id is None:
            return
        self.canvas.push_undo()
        for item in orphans:
            item.class_id = class_id
            item.update()
        self.canvas.mark_dirty()
        self.canvas.notify_boxes_changed()
        self._update_box_list()
        self._refresh_notice()
        self.status.showMessage(
            f"{len(orphans)} kutuya '{self.classes.name_of(class_id)}' atandı.", 4000)

    def delete_orphan_boxes(self):
        orphans = self.canvas.orphan_boxes()
        if not orphans:
            self._refresh_notice()
            return
        self.canvas.push_undo()
        for item in orphans:
            self.canvas.remove_box(item)
        self.canvas.mark_dirty()
        self._update_box_list()
        self._refresh_notice()
        self.status.showMessage(f"{len(orphans)} sahipsiz kutu silindi.", 4000)

    def confirm_review(self):
        if self.current_index < 0:
            return
        name = os.path.basename(self.images[self.current_index])
        self._pending_review = review.unmark(self.label_dir, name)
        self._refresh_image_list_marks()
        self._refresh_notice()
        self.status.showMessage("Etiketler onaylandı.", 3000)

    def reject_review(self):
        if self.current_index < 0:
            return
        name = os.path.basename(self.images[self.current_index])
        self.canvas.clear_boxes()
        self.save_current(silent=True)
        self._pending_review = review.unmark(self.label_dir, name)
        self._refresh_image_list_marks()
        self._refresh_notice()
        self.status.showMessage("Model etiketleri silindi.", 3000)

    def _reload_review_queue(self):
        self._pending_review = review.load_pending(self.label_dir or "")

    # ================================================================ kaydetme
    def save_current(self, silent: bool = True) -> bool:
        if self.current_index < 0 or not self.canvas.has_image() or not self.label_dir:
            return False

        if self.canvas.proposals():
            reply = QMessageBox.question(
                self, "Onaylanmamış öneriler",
                f"{len(self.canvas.proposals())} model önerisi var. "
                f"Kaydetmeden önce ne yapılsın?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes)
            if reply == QMessageBox.Cancel:
                return False
            if reply == QMessageBox.Yes:
                self.canvas.accept_proposals()
            else:
                self.canvas.discard_proposals()
            self._refresh_notice()

        if not len(self.classes) and self.canvas.box_count() > 0:
            QMessageBox.warning(self, "Sınıf yok",
                                "Kaydetmeden önce en az bir sınıf tanımlamalısınız.")
            return False

        invalid = self.canvas.orphan_boxes()
        if invalid:
            self._refresh_notice()
            QMessageBox.warning(
                self, "Geçersiz sınıf",
                f"{len(invalid)} kutunun sınıfı sınıf listesinde yok. "
                f"Üstteki uyarı şeridinden sınıf atayın ya da bu kutuları silin.")
            return False

        path = self.images[self.current_index]
        img_w, img_h = self.canvas.image_size()
        boxes = [(b.class_id, b.rect().left(), b.rect().top(),
                  b.rect().right(), b.rect().bottom())
                 for b in self.canvas.real_boxes()]
        txt_path = yolo_io.label_path_for(path, self.label_dir)
        try:
            yolo_io.backup_label(txt_path)
            yolo_io.save_labels(txt_path, boxes, img_w, img_h)
            yolo_io.save_classes(os.path.join(self.label_dir, "classes.txt"),
                                 self.classes.names())
        except OSError as exc:
            QMessageBox.critical(self, "Kaydedilemedi", f"Dosya yazılamadı:\n{exc}")
            return False

        name = os.path.basename(path)
        if name in self._pending_review:
            self._pending_review = review.unmark(self.label_dir, name)
            self._refresh_notice()

        self.canvas.set_dirty(False)
        self._refresh_image_list_marks()
        self.status.showMessage(
            f"Kaydedildi: {txt_path if not silent else os.path.basename(txt_path)}",
            5000 if not silent else 2500)
        self._update_title()
        return True

    def _maybe_save_before_leaving(self) -> bool:
        reply = QMessageBox.question(
            self, "Kaydedilmemiş değişiklikler",
            "Bu resimdeki değişiklikler kaydedilmedi. Kaydedilsin mi?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            return self.save_current(silent=True)
        self.canvas.set_dirty(False)
        return True

    def _save_before_move(self) -> bool:
        self.save_current(silent=True)
        if self.canvas.is_dirty():
            self.status.showMessage("Kaydedilemedi, resim değiştirilmedi.", 4000)
            return False
        return True

    # ================================================================ gezinme
    def go_next(self):
        self._step(+1)

    def go_prev(self):
        self._step(-1)

    def _step(self, direction: int):
        if not self.images or not self._save_before_move():
            return
        rows = self._visible_rows()
        if not rows:
            return
        if self.current_index in rows:
            position = rows.index(self.current_index)
        else:
            position = 0 if direction > 0 else len(rows) - 1
        target = position + direction
        if 0 <= target < len(rows):
            self._select_row(rows[target])
        else:
            self.status.showMessage(
                "Son resimdesiniz." if direction > 0 else "İlk resimdesiniz.", 3000)

    def go_next_unlabeled(self):
        if not self.images or not self.label_dir or not self._save_before_move():
            return
        for offset in range(1, len(self.images) + 1):
            index = (self.current_index + offset) % len(self.images)
            if not yolo_io.has_labels(self.images[index], self.label_dir):
                self._select_row(index)
                self.status.showMessage(
                    f"Etiketlenmemiş: {os.path.basename(self.images[index])}", 4000)
                return
        self.status.showMessage("Tüm resimler etiketlenmiş görünüyor.", 4000)

    def go_to_image(self, path: str, box_index: int = -1):
        """Kalite denetiminden gelen 'resme git' isteği."""
        try:
            index = self.images.index(path)
        except ValueError:
            QMessageBox.information(self, "Bulunamadı",
                                    "Bu resim açık klasörde değil.")
            return
        self.only_unlabeled.setChecked(False)
        self._select_row(index)
        if box_index >= 0:
            boxes = self.canvas.boxes()
            if box_index < len(boxes):
                self.canvas.clear_selection()
                boxes[box_index].setSelected(True)
                self.canvas.notify_selection()
                self.canvas.zoom_to_box(boxes[box_index])

    # ================================================================ sınıflar
    def _load_classes_from_disk(self):
        # Var olan bir classes.txt yetkilidir: BOŞ olsa bile. Aksi halde
        # sınıfları silip kapattığınızda eski bir dosya onları geri getirir.
        names = []
        for folder in (self.label_dir, self.image_dir):
            if not folder:
                continue
            candidate = os.path.join(folder, "classes.txt")
            if os.path.isfile(candidate):
                names = yolo_io.load_classes(candidate)
                break

        self.classes.clear()
        seen, duplicates = set(), 0
        for index, raw in enumerate(names):
            name = raw
            suffix = 2
            while name in seen:
                name = f"{raw}_{suffix}"
                suffix += 1
                duplicates += 1
            seen.add(name)
            self.classes.add(name, color_for_index(index), silent_duplicates=True)
        if duplicates:
            self.status.showMessage(
                f"classes.txt'te {duplicates} yinelenen ad vardı; kimlikler "
                f"korunarak yeniden adlandırıldı.", 8000)
        self._update_class_list()

    def _update_class_list(self):
        self._loading = True
        current = self.class_list.currentRow()
        hidden = self.canvas.hidden_classes()
        self.class_list.clear()
        for index, cls in enumerate(self.classes):
            item = QListWidgetItem(color_icon(cls.color), f"{index}: {cls.name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if index in hidden else Qt.Checked)
            item.setToolTip("Kutucuğu kaldırırsanız bu sınıfın kutuları gizlenir.")
            self.class_list.addItem(item)
        self._loading = False

        if len(self.classes):
            row = current if 0 <= current < len(self.classes) else 0
            self.class_list.setCurrentRow(row)
            self.canvas.current_class = row
        self.canvas.refresh_boxes()
        self._update_active_class_label()
        self._update_box_list()

    def _on_class_item_changed(self, item: QListWidgetItem):
        if self._loading:
            return
        # Kutucuğa tıklanmış demektir: bunu takip eden itemClicked sinyali
        # seçili kutunun sınıfını DEĞİŞTİRMEMELİ.
        self._checkbox_just_toggled = True
        hidden = set()
        for row in range(self.class_list.count()):
            entry = self.class_list.item(row)
            if entry is not None and entry.checkState() == Qt.Unchecked:
                hidden.add(row)
        self.canvas.set_hidden_classes(hidden)
        self._update_box_list()

    def add_class(self):
        dialog = ClassDialog(self, color=color_for_index(len(self.classes)),
                             title="Sınıf Ekle")
        if dialog.exec_() != ClassDialog.Accepted:
            return
        name, color = dialog.values()
        try:
            index = self.classes.add(name, color)
        except ValueError as exc:
            QMessageBox.warning(self, "Eklenemedi", str(exc))
            return
        self._update_class_list()
        self.class_list.setCurrentRow(index)
        self._persist_classes()
        self._refresh_notice()

        # İlk sınıfı ekledikten sonra kullanıcının bir sonraki adımı her zaman
        # kutu çizmektir: W'ye basmasını beklemek yerine çizim modunu açıyoruz.
        if len(self.classes) == 1 and self.canvas.has_image():
            self.canvas.set_mode(Mode.DRAW)
            self.status.showMessage(
                "Çizim modu açıldı — fareyle sürükleyerek kutu çizin. "
                "Kapatmak için W veya Esc.", 8000)
        else:
            self.status.showMessage(
                f"'{self.classes[index].name}' eklendi. Kutu çizmek için W.", 5000)

    def edit_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            self.status.showMessage("Önce listeden bir sınıf seçin.", 3000)
            return
        cls = self.classes[row]
        dialog = ClassDialog(self, name=cls.name, color=cls.color,
                             title="Sınıfı Düzenle")
        if dialog.exec_() != ClassDialog.Accepted:
            return
        name, color = dialog.values()
        try:
            self.classes.rename(row, name)
        except ValueError as exc:
            QMessageBox.warning(self, "Değiştirilemedi", str(exc))
            return
        self.classes.set_color(row, color)
        self._update_class_list()
        self.class_list.setCurrentRow(row)
        self._persist_classes()

    def delete_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            self.status.showMessage("Önce listeden bir sınıf seçin.", 3000)
            return
        name = self.classes[row].name
        usage = self._class_usage_count(row)
        usage += sum(1 for b in self.canvas.boxes() if b.class_id == row)

        move_to = None
        if usage == 0:
            reply = QMessageBox.question(
                self, "Sınıfı sil",
                f"'{name}' sınıfı silinsin mi?\n(Bu sınıfa ait hiç etiket yok.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return
        else:
            others = [c.name for index, c in enumerate(self.classes) if index != row]
            dialog = DeleteClassDialog(self, name, usage, others)
            if dialog.exec_() != DeleteClassDialog.Accepted:
                return
            choice = dialog.values()
            if choice["action"] == DeleteClassDialog.MOVE_BOXES:
                move_to = self.classes.index_of(choice["target_name"])
                if move_to < 0:
                    QMessageBox.warning(self, "Hedef bulunamadı",
                                        "Kutuların taşınacağı sınıf bulunamadı.")
                    return

        self._apply_class_removal(row, move_to)
        self.status.showMessage(f"'{name}' sınıfı silindi.", 4000)

    def _apply_class_removal(self, row: int, move_to: Optional[int]):
        """
        Sınıfı siler ve veri setini TUTARLI bırakır.

        Kutular ya hedef sınıfa taşınır ya da silinir; ardından diskteki tüm
        etiket dosyaları yeni kimliklere göre yeniden numaralandırılır.
        Böylece 'sahipsiz kutu' durumu hiç oluşmaz.
        """
        if move_to is not None:
            self.canvas.reassign_class(row, move_to)
            if self.label_dir:
                yolo_io.reassign_label_files(self.label_dir, row, move_to)

        mapping = self.classes.remove(row)
        removed = self.canvas.apply_class_mapping(mapping)
        if self.label_dir:
            yolo_io.remap_label_files(self.label_dir, mapping)

        # Eski geri-al kayıtları artık eski kimliklere göre; geri alınırlarsa
        # sahipsiz kutu üretirler. Bu yüzden yığın boşaltılır.
        self.canvas.clear_undo()

        self._update_class_list()
        self._persist_classes()
        if removed or move_to is not None:
            self.canvas.mark_dirty()
        self._update_box_list()
        self._refresh_image_list_marks()
        self._refresh_notice()

    def _class_usage_count(self, class_id: int) -> int:
        """Diskteki etiket dosyalarında bu sınıftan kaç kutu var?"""
        count = 0
        if not self.label_dir or not os.path.isdir(self.label_dir):
            return count
        for filename in os.listdir(self.label_dir):
            if not filename.lower().endswith(".txt") or filename.lower() == "classes.txt":
                continue
            for row in yolo_io.load_raw(os.path.join(self.label_dir, filename)):
                if row[0] == class_id:
                    count += 1
        return count

    def _persist_classes(self):
        if self.label_dir:
            try:
                yolo_io.save_classes(os.path.join(self.label_dir, "classes.txt"),
                                     self.classes.names())
            except OSError:
                pass

    def _on_class_row_changed(self, row: int):
        if row >= 0:
            self.canvas.current_class = row
        self._update_active_class_label()

    def _update_active_class_label(self):
        if not len(self.classes):
            self.lbl_active_class.setText("Aktif sınıf: <i>yok</i>")
            return
        class_id = max(0, min(self.canvas.current_class, len(self.classes) - 1))
        self.lbl_active_class.setText(
            f'Aktif sınıf: <span style="color:{self.classes.color_of(class_id)};">■</span> '
            f'{self.classes.name_of(class_id)}')

    def _on_class_clicked(self, item: QListWidgetItem):
        if self._checkbox_just_toggled:
            self._checkbox_just_toggled = False
            return
        row = self.class_list.row(item)
        if row < 0:
            return
        self.canvas.current_class = row
        self._update_active_class_label()
        if self.canvas.assign_class_to_selection(row):
            self._update_box_list()
            self.status.showMessage(
                f"Seçili kutulara '{self.classes[row].name}' atandı.", 2500)

    # ================================================================ sınıf seçme
    def _ask_class(self, initial: int = -1) -> Optional[int]:
        dialog = LabelPickerDialog(self, self.classes, initial)
        if dialog.exec_() != LabelPickerDialog.Accepted:
            return None
        name = dialog.chosen_name()
        index = self.classes.index_of(name)
        if index < 0:
            try:
                index = self.classes.add(name)
            except ValueError as exc:
                QMessageBox.warning(self, "Sınıf eklenemedi", str(exc))
                return None
            self._update_class_list()
            self._persist_classes()
        self.class_list.setCurrentRow(index)
        self.canvas.current_class = index
        self._update_active_class_label()
        return index

    def _on_box_created(self, item):
        self._session_boxes += 1
        must_ask = self.act_ask_class.isChecked() or not len(self.classes)
        if must_ask:
            class_id = self._ask_class(item.class_id if len(self.classes) else -1)
            if class_id is None:
                self.canvas.remove_box(item)
                self.canvas.pop_undo()
                self._session_boxes = max(0, self._session_boxes - 1)
            else:
                item.class_id = class_id
                item.update()
        if self.act_return_edit.isChecked():
            self.canvas.set_mode(Mode.EDIT)
        self._update_box_list()
        self._update_status()
        self._refresh_notice()

    def _on_box_double_clicked(self, item):
        class_id = self._ask_class(item.class_id)
        if class_id is not None and class_id != item.class_id:
            self.canvas.push_undo()
            item.class_id = class_id
            item.update()
            self.canvas.mark_dirty()
            self.canvas.notify_boxes_changed()
        self._update_box_list()
        self._refresh_notice()

    def change_class_of_selection(self):
        item = self.canvas.selected_box()
        if item is None:
            self.status.showMessage("Önce bir kutu seçin.", 2500)
            return
        self._on_box_double_clicked(item)

    def _on_context_menu(self, global_pos, box):
        menu = QMenu(self)
        act_class = act_delete = act_lock = act_snap = None
        if box is not None:
            act_class = menu.addAction("Sınıfını değiştir...")
            act_lock = menu.addAction("Kilidi aç" if box.locked else "Kilitle")
            if imaging.available():
                act_snap = menu.addAction("Kenarlara oturt (S)")
            act_delete = menu.addAction("Kutuyu sil")
            menu.addSeparator()
        act_draw = menu.addAction("Çizim modunu aç/kapat (W)")
        act_new_class = menu.addAction("Yeni sınıf ekle (E)")
        menu.addSeparator()
        act_fit = menu.addAction("Ekrana sığdır")

        chosen = menu.exec_(global_pos)
        if chosen is None:
            return
        if chosen is act_class:
            self._on_box_double_clicked(box)
        elif chosen is act_delete:
            self.delete_selected_box()
        elif chosen is act_lock:
            self.toggle_lock()
        elif chosen is act_snap:
            self.snap_selection()
        elif chosen is act_draw:
            self.canvas.toggle_mode()
        elif chosen is act_new_class:
            self.add_class()
        elif chosen is act_fit:
            self.canvas.fit_to_window()

    # ================================================================ kutu listesi
    def _update_box_list(self):
        self._loading = True
        self.box_list.clear()
        for index, box in enumerate(self.canvas.boxes()):
            rect = box.rect()
            marks = []
            if box.proposal:
                marks.append(f"öneri {box.score:.2f}")
            if box.locked:
                marks.append("kilitli")
            if self.canvas.is_box_hidden(box):
                marks.append("gizli")
            suffix = f"  ({', '.join(marks)})" if marks else ""
            text = (f"{index + 1}. {self.classes.name_of(box.class_id)}  "
                    f"[{int(rect.width())}x{int(rect.height())}]{suffix}")
            item = QListWidgetItem(color_icon(self.classes.color_of(box.class_id), 12),
                                   text)
            self.box_list.addItem(item)
            if box.isSelected():
                self.box_list.setCurrentRow(index)
        self._loading = False

    def _on_box_row_changed(self, row: int):
        if self._loading or row < 0:
            return
        boxes = self.canvas.boxes()
        if row >= len(boxes):
            return
        self.canvas.clear_selection()
        boxes[row].setSelected(True)
        self.canvas.notify_selection()
        self.canvas.ensureVisible(boxes[row], 40, 40)

    def _on_box_item_double_clicked(self, item: QListWidgetItem):
        row = self.box_list.row(item)
        boxes = self.canvas.boxes()
        if 0 <= row < len(boxes):
            self.canvas.zoom_to_box(boxes[row])

    def _on_canvas_selection(self):
        self._update_box_list()
        selected = self.canvas.selected_box()
        if selected is not None and 0 <= selected.class_id < len(self.classes):
            self._loading = True
            self.class_list.setCurrentRow(selected.class_id)
            self._loading = False
            self.canvas.current_class = selected.class_id
            self._update_active_class_label()
        if selected is not None:
            self.canvas.setFocus()

    # ================================================================ eylemler
    def undo(self):
        if self.canvas.undo():
            self._update_box_list()
            self._refresh_notice()
            self.status.showMessage("Geri alındı.", 2000)
        else:
            self.status.showMessage("Geri alınacak işlem yok.", 2000)

    def delete_selected_box(self):
        count = self.canvas.delete_selected()
        if count:
            self._update_box_list()
            self._refresh_notice()
            self.status.showMessage(f"{count} kutu silindi.", 2000)
        elif self.canvas.selected_boxes():
            self.status.showMessage("Seçili kutular kilitli.", 2500)

    def clear_all_boxes(self):
        count = self.canvas.box_count()
        if count == 0:
            return
        reply = QMessageBox.question(
            self, "Tüm kutuları sil",
            f"Bu resimdeki {count} kutunun tamamı silinsin mi? "
            f"(Ctrl+Z ile geri alınabilir)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes and self.canvas.clear_boxes():
            self._update_box_list()
            self._refresh_notice()
            self.status.showMessage(f"{count} kutu silindi.", 3000)

    def select_all_boxes(self):
        count = self.canvas.select_all()
        self.status.showMessage(f"{count} kutu seçildi.", 2000)

    def toggle_lock(self):
        count = self.canvas.toggle_lock_selection()
        if count:
            self._update_box_list()
            self.status.showMessage(f"{count} kutunun kilit durumu değişti.", 2500)
        else:
            self.status.showMessage("Önce bir kutu seçin.", 2500)

    def _require_imaging(self) -> bool:
        if imaging.available():
            return True
        QMessageBox.information(
            self, "Görüntü işleme kurulu değil",
            self._missing_package_message(["numpy", "opencv-python-headless"]))
        return False

    def snap_selection(self):
        if not self._require_imaging():
            return
        count = self.canvas.snap_selection()
        self.status.showMessage(
            f"{count} kutu kenarlara oturtuldu." if count
            else "Oturtulacak belirgin kenar bulunamadı.", 3000)
        self._update_box_list()

    def copy_boxes(self):
        if not self.canvas.has_image() or not self.canvas.real_boxes():
            self.status.showMessage("Kopyalanacak kutu yok.", 2500)
            return
        self._clipboard = [(b.class_id, b.rect().left(), b.rect().top(),
                            b.rect().right(), b.rect().bottom())
                           for b in self.canvas.real_boxes()]
        self._clipboard_size = self.canvas.image_size()
        self.status.showMessage(f"{len(self._clipboard)} kutu kopyalandı.", 3000)

    def paste_boxes(self):
        if not self.canvas.has_image() or not self._clipboard:
            self.status.showMessage("Panoda kutu yok. (Ctrl+C ile kopyalayın)", 3000)
            return
        src_w, src_h = self._clipboard_size
        dst_w, dst_h = self.canvas.image_size()
        if not src_w or not src_h:
            return
        scale_x, scale_y = dst_w / src_w, dst_h / src_h
        self.canvas.push_undo()
        pasted = 0
        for class_id, x1, y1, x2, y2 in self._clipboard:
            if not (0 <= class_id < len(self.classes)):
                continue
            self.canvas.add_box(QRectF(QPointF(x1 * scale_x, y1 * scale_y),
                                       QPointF(x2 * scale_x, y2 * scale_y)), class_id)
            pasted += 1
        self.canvas.mark_dirty()
        self._update_box_list()
        self.status.showMessage(f"{pasted} kutu yapıştırıldı.", 3000)

    # ================================================================ görünüm
    def _toggle_boxes_visible(self, checked: bool):
        self.canvas.set_boxes_visible(not checked)
        self._update_box_list()
        self.status.showMessage(
            "Kutular gizlendi (H ile geri getir)." if checked else "Kutular görünür.",
            2500)

    def _toggle_edges(self, checked: bool):
        if checked and not self._require_imaging():
            self.act_edges.setChecked(False)
            return
        self.canvas.set_edge_overlay(checked)

    def _toggle_magnetic(self, checked: bool):
        if checked and not self._require_imaging():
            self.act_magnetic.setChecked(False)
            return
        self.canvas.magnetic = checked
        self.status.showMessage(
            "Manyetik kutu açık: çizdiğiniz kutu kenarlara oturtulur." if checked
            else "Manyetik kutu kapalı.", 3000)

    def _toggle_loupe(self, checked: bool):
        self.canvas.loupe_enabled = checked
        self.canvas.viewport().update()

    def _toggle_enhance_bar(self, checked: bool):
        if checked and not self._require_imaging():
            self.act_enhance.setChecked(False)
            return
        self.enhance_bar.setVisible(checked)

    def _on_enhancement_changed(self):
        self._apply_enhancement_to_canvas()

    def _apply_enhancement_to_canvas(self):
        if not imaging.available():
            return
        values = self.enhance_bar.values()
        self.canvas.set_enhancement(values["brightness"], values["contrast"],
                                    values["gamma"], values["clahe"])
        if self.act_edges.isChecked():
            self.canvas.set_edge_overlay(True)

    # ================================================================ veri seti
    def _dataset_ready(self) -> bool:
        if not self.images or not self.label_dir:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return False
        if not len(self.classes):
            QMessageBox.information(self, "Sınıf yok",
                                    "Önce en az bir sınıf tanımlayın.")
            return False
        return True

    def show_stats(self):
        if not self._dataset_ready():
            return
        if self.canvas.is_dirty():
            self.save_current(silent=True)
        stats = dataset.collect_stats(self.images, self.label_dir, len(self.classes))
        colors = [self.classes.color_of(i) for i in range(len(self.classes))]
        StatsDialog(self, stats, self.classes.names(), colors).exec_()

    def run_audit(self):
        if not self._dataset_ready():
            return
        if self.canvas.is_dirty():
            self.save_current(silent=True)
        findings = audit.audit(self.images, self.label_dir, self.classes.names())
        if not findings:
            QMessageBox.information(self, "Kalite Denetimi",
                                    "Hiçbir sorun bulunamadı. Veri seti temiz.")
            return
        dialog = AuditDialog(self, findings)
        dialog.findingActivated.connect(self.go_to_image)
        dialog.exec_()

    def export_dataset(self):
        if not self._dataset_ready():
            return
        if self.canvas.is_dirty() and not self._save_before_move():
            return

        parent_dir = os.path.dirname(os.path.abspath(self.image_dir))
        default_out = os.path.join(parent_dir,
                                   os.path.basename(self.image_dir) + "_dataset")
        dialog = ExportDialog(self, default_out, len(self.images))
        if dialog.exec_() != ExportDialog.Accepted:
            return
        options = dialog.values()

        items = []
        for path in self.images:
            txt = yolo_io.find_existing_label(path, self.label_dir)
            signature = dataset.image_signature(txt) if txt else ()
            if not signature and not options["include_unlabeled"]:
                continue
            items.append((path, signature))

        if not items:
            QMessageBox.warning(self, "Dışa aktarılacak resim yok",
                                "Etiketlenmiş resim bulunamadı. Etiketsizleri dahil "
                                "etmek için sihirbazdaki kutuyu işaretleyin.")
            return

        split = dataset.stratified_split(items, options["ratios"], options["seed"])
        progress = QProgressDialog("Veri seti hazırlanıyor...", "İptal", 0,
                                   len(items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(done: int, total: int, name: str) -> bool:
            progress.setMaximum(total)
            progress.setValue(done)
            progress.setLabelText(f"{name}\n({done}/{total})")
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            result = dataset.export_dataset(
                split, self.label_dir, options["out_dir"], self.classes.names(),
                options["mode"], on_progress)
        except OSError as exc:
            progress.close()
            QMessageBox.critical(self, "Dışa aktarılamadı", f"Dosya hatası:\n{exc}")
            return
        progress.close()

        if result.get("cancelled"):
            QMessageBox.information(self, "İptal edildi",
                                    "Dışa aktarma yarıda kesildi; hedef klasörde "
                                    "eksik dosyalar kalmış olabilir.")
            return

        counts = result["counts"]
        QMessageBox.information(
            self, "Veri seti hazır",
            f"<b>{result['total']} resim</b> dışa aktarıldı.<br><br>"
            f"Eğitim: {counts.get('train', 0)}<br>"
            f"Doğrulama: {counts.get('val', 0)}<br>"
            f"Test: {counts.get('test', 0)}<br>"
            f"Boş etiket (arka plan): {result['empty_labels']}<br><br>"
            f"Klasör:<br><code>{result['out_dir']}</code><br><br>"
            f"Eğitmek için:<br><code>yolo detect train "
            f"data=\"{result['yaml']}\" model=yolov8n.pt epochs=100 imgsz=640</code>")
        self.status.showMessage(f"Veri seti dışa aktarıldı: {result['out_dir']}", 10000)

    # ---------------------------------------------------- format dönüştürme
    def _size_of(self, path: str):
        reader = QImageReader(path)
        size = reader.size()
        return (size.width(), size.height()) if size.isValid() else (0, 0)

    def export_coco(self):
        if not self._dataset_ready():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "COCO JSON kaydet",
            os.path.join(self.image_dir or "", "annotations.json"), "JSON (*.json)")
        if not path:
            return
        try:
            result = converters.export_coco(self.images, self.label_dir,
                                            self.classes.names(), self._size_of, path)
        except OSError as exc:
            QMessageBox.critical(self, "Yazılamadı", str(exc))
            return
        QMessageBox.information(
            self, "COCO dışa aktarıldı",
            f"{result['images']} resim, {result['annotations']} kutu yazıldı:\n{path}")

    def import_coco(self):
        if not self.image_dir:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "COCO JSON aç",
                                              self.image_dir or "", "JSON (*.json)")
        if not path:
            return
        if QMessageBox.question(
                self, "İçe aktar",
                f"Etiketler '{self.label_dir}' klasörüne yazılacak ve aynı adlı "
                f"mevcut .txt dosyaları üzerine yazılacak. Devam edilsin mi?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            result = converters.import_coco(path, self.label_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Okunamadı", f"COCO dosyası okunamadı:\n{exc}")
            return
        self._after_import(result, f"{result['annotations']} kutu içe aktarıldı "
                                   f"({result['skipped']} satır atlandı).")

    def export_voc(self):
        if not self._dataset_ready():
            return
        folder = QFileDialog.getExistingDirectory(self, "VOC XML klasörü seç",
                                                  self.image_dir or "")
        if not folder:
            return
        try:
            result = converters.export_voc(self.images, self.label_dir,
                                           self.classes.names(), self._size_of, folder)
        except OSError as exc:
            QMessageBox.critical(self, "Yazılamadı", str(exc))
            return
        QMessageBox.information(
            self, "VOC dışa aktarıldı",
            f"{result['files']} XML dosyası, {result['boxes']} kutu:\n{folder}")

    def import_voc(self):
        if not self.image_dir:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return
        folder = QFileDialog.getExistingDirectory(self, "VOC XML klasörü seç",
                                                  self.image_dir or "")
        if not folder:
            return
        if QMessageBox.question(
                self, "İçe aktar",
                f"Etiketler '{self.label_dir}' klasörüne yazılacak ve aynı adlı "
                f"mevcut .txt dosyaları üzerine yazılacak. Devam edilsin mi?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            result = converters.import_voc(folder, self.label_dir, self.classes.names())
        except OSError as exc:
            QMessageBox.critical(self, "Okunamadı", str(exc))
            return
        self._after_import(result, f"{result['files']} dosya, {result['boxes']} kutu "
                                   f"içe aktarıldı.")

    def _after_import(self, result: dict, message: str):
        self._load_classes_from_disk()
        self._refresh_image_list_marks()
        if self.current_index >= 0:
            self._load_image_at(self.current_index, force=True)
        QMessageBox.information(self, "İçe aktarıldı",
                                f"{message}\n\nSınıflar: "
                                f"{', '.join(result.get('class_names', [])[:10])}")

    def restore_from_backup(self):
        if self.current_index < 0 or not self.label_dir:
            return
        txt_path = yolo_io.label_path_for(self.images[self.current_index], self.label_dir)
        backups = yolo_io.list_backups(txt_path)
        if not backups:
            QMessageBox.information(self, "Yedek yok",
                                    "Bu resim için henüz yedek alınmamış. "
                                    "Yedekler ilk kayıttan sonra oluşur.")
            return
        labels = [os.path.basename(path) for path in backups]
        choice, ok = QInputDialog.getItem(
            self, "Yedekten geri al",
            "Geri yüklenecek sürümü seçin (en yenisi üstte):", labels, 0, False)
        if not ok:
            return
        source = backups[labels.index(choice)]
        img_w, img_h = self.canvas.image_size()
        rows = yolo_io.load_labels(source, img_w, img_h)
        self.canvas.push_undo()
        self.canvas.set_boxes(rows)
        self.canvas.mark_dirty()
        self._update_box_list()
        self.status.showMessage(
            f"{len(rows)} kutu yedekten yüklendi. Kalıcı olması için Ctrl+S, "
            f"vazgeçmek için Ctrl+Z.", 8000)

    # ================================================================ proje
    def save_project(self):
        if not self.image_dir:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Projeyi kaydet",
            os.path.join(self.image_dir, os.path.basename(self.image_dir)
                         + project.EXTENSION),
            f"Etiketleme projesi (*{project.EXTENSION})")
        if not path:
            return
        document = project.build(
            self.image_dir, self.label_dir, list(self.classes), self._model_path,
            {"conf": self._conf, "iou": self._iou, "theme": self._theme})
        try:
            written = project.save(path, document)
        except OSError as exc:
            QMessageBox.critical(self, "Kaydedilemedi", str(exc))
            return
        self.status.showMessage(f"Proje kaydedildi: {written}", 6000)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Proje aç", self.image_dir or "",
            f"Etiketleme projesi (*{project.EXTENSION})")
        if not path:
            return
        try:
            document = project.load(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Açılamadı", f"Proje dosyası okunamadı:\n{exc}")
            return
        if not project.is_usable(document):
            QMessageBox.warning(self, "Klasör bulunamadı",
                                "Projedeki resim klasörü artık yok:\n"
                                f"{document.get('image_dir')}")
            return

        if not self.load_folder(document["image_dir"]):
            return
        label_dir = document.get("label_dir")
        if label_dir and os.path.isdir(label_dir):
            self.label_dir = label_dir

        if document["classes"]:
            self.classes.clear()
            for index, entry in enumerate(document["classes"]):
                self.classes.add(entry["name"],
                                 entry.get("color") or color_for_index(index),
                                 silent_duplicates=True)
            self._update_class_list()
            self._persist_classes()

        settings = document.get("settings") or {}
        self._conf = float(settings.get("conf", self._conf))
        self._iou = float(settings.get("iou", self._iou))
        stored_theme = settings.get("theme")
        if stored_theme in (theme.DARK, theme.LIGHT):
            self._apply_theme(stored_theme)
        model_path = document.get("model_path") or ""
        if model_path and os.path.isfile(model_path) and inference.available():
            self._load_model_path(model_path, silent=True)

        self._reload_review_queue()
        self._refresh_image_list_marks()
        self.status.showMessage(f"Proje açıldı: {os.path.basename(path)}", 6000)

    # ================================================================ yapay zekâ
    # ---------------------------------------------------- model kütüphanesi
    def model_library(self) -> training.ModelLibrary:
        return training.ModelLibrary(self.image_dir or "")

    def _rebuild_models_menu(self):
        """Menü her açıldığında diskten tazelenir; elle yenilemeye gerek kalmaz."""
        self.models_menu.clear()
        if not self.image_dir:
            action = self.models_menu.addAction("(önce bir resim klasörü açın)")
            action.setEnabled(False)
            return

        library = self.model_library()
        models = library.list_models()
        if not models:
            action = self.models_menu.addAction("(kütüphanede model yok)")
            action.setEnabled(False)
            self.models_menu.addSeparator()
            self.models_menu.addAction(
                "Model eğiterek oluştur...", self.train_model)
            return

        for path in models:
            metadata = library.read_metadata(path)
            action = self.models_menu.addAction(training.describe_model(path, metadata))
            action.setCheckable(True)
            action.setChecked(os.path.abspath(path)
                              == os.path.abspath(self._model_path or ""))
            tooltip = [path]
            if metadata.get("created"):
                tooltip.append(f"Eğitim: {metadata['created']}")
            if metadata.get("classes"):
                tooltip.append("Sınıflar: " + ", ".join(metadata["classes"]))
            action.setToolTip("\n".join(tooltip))
            action.triggered.connect(
                lambda _checked=False, target=path: self._select_library_model(target))
        self.models_menu.addSeparator()
        self.models_menu.addAction("Klasörü aç...", self._open_models_folder)

    def _select_library_model(self, path: str):
        if self._load_model_path(path, silent=True):
            self.status.showMessage(
                f"Model yüklendi: {os.path.basename(path)} — Ctrl+R ile tahmin edin.",
                6000)
        else:
            QMessageBox.warning(self, "Yüklenemedi",
                                f"Model açılamadı:\n{path}")

    def _open_models_folder(self):
        folder = self.model_library().ensure()
        if not folder:
            return
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ---------------------------------------------------- model eğitimi
    def _labeled_image_count(self) -> int:
        if not self.label_dir:
            return 0
        return sum(1 for path in self.images
                   if yolo_io.has_labels(path, self.label_dir))

    def _build_training_dataset(self) -> Optional[str]:
        """
        Eğitim penceresi çağırır: mevcut etiketlerden data.yaml üretir.

        Sadece etiketli resimler alınır; %80/%20 katmanlı bölme, sabit tohum.
        """
        if not self._dataset_ready():
            return None
        if self.canvas.is_dirty():
            self.save_current(silent=True)

        items = []
        for path in self.images:
            txt = yolo_io.find_existing_label(path, self.label_dir)
            signature = dataset.image_signature(txt) if txt else ()
            if not signature:
                continue
            items.append((path, signature))

        if len(items) < 4:
            QMessageBox.warning(
                self, "Yeterli etiket yok",
                f"Eğitim için en az 4 etiketli resim gerekir (şu an {len(items)}). "
                f"Pratikte anlamlı bir model için 100-200 resim önerilir.")
            return None

        split = dataset.stratified_split(items, (80, 20, 0), seed=0)
        out_dir = training.dataset_dir(self.image_dir)
        try:
            result = dataset.export_dataset(split, self.label_dir, out_dir,
                                            self.classes.names(), "copy", None)
        except OSError as exc:
            QMessageBox.critical(self, "Veri seti hazırlanamadı", str(exc))
            return None
        return result.get("yaml")

    def train_model(self):
        if not self._dataset_ready():
            return
        dialog = TrainDialog(self, self.image_dir, self.classes.names(),
                             self._labeled_image_count(), self._build_training_dataset)
        dialog.modelReady.connect(self._on_model_trained)
        dialog.exec_()

    def _on_model_trained(self, path: str):
        self._load_model_path(path, silent=True)
        self.status.showMessage(
            f"Eğitilen model yüklendi: {os.path.basename(path)} — Ctrl+R ile "
            f"tahmin alabilirsiniz.", 10000)

    # ---------------------------------------------------- kurulum yardımı
    def _package_versions(self) -> dict:
        """Hangi isteğe bağlı paket kurulu? Sürümleriyle birlikte."""
        import sys as _sys
        versions = {"Python": _sys.version.split()[0],
                    "Çalışan yorumlayıcı": _sys.executable}
        try:
            from PyQt5.QtCore import PYQT_VERSION_STR
            versions["PyQt5"] = PYQT_VERSION_STR
        except Exception:
            versions["PyQt5"] = None
        for label, module in (("numpy", "numpy"), ("OpenCV", "cv2"),
                              ("onnxruntime", "onnxruntime")):
            try:
                imported = __import__(module)
                versions[label] = getattr(imported, "__version__", "kurulu")
            except Exception:
                versions[label] = None
        return versions

    def _missing_package_message(self, packages) -> str:
        import sys as _sys
        errors = dict(imaging.IMPORT_ERRORS)
        errors.update(inference.IMPORT_ERRORS)
        detail = ""
        for package in packages:
            if package in errors:
                detail += (f"<br><b>{package}</b> içe aktarılamadı:<br>"
                           f"<code>{errors[package]}</code><br>")

        return (
            f"Bu özellik için şu paket(ler) gerekli:<br>"
            f"<code>pip install {' '.join(packages)}</code>"
            + (f"<br>{detail}" if detail else "") +
            f"<br><b>Paketleri az önce kurduysanız programı kapatıp yeniden "
            f"açın</b> — Python kütüphaneleri yalnızca açılışta yükler, çalışan "
            f"süreç sonradan kurulanı görmez.<br><br>"
            f"Uygulamanın kullandığı Python:<br><code>{_sys.executable}</code><br>"
            f"Paketleri bu Python'a kurmuş olmanız gerekir (sanal ortam aktifken "
            f"<code>pip install -r requirements.txt</code>).<br><br>"
            f"Kurulum hiç başarılı olmuyorsa Python sürümünüz için henüz hazır "
            f"paket yayınlanmamış olabilir; o durumda Python 3.12 ile ayrı bir "
            f"ortam kurun:<br>"
            f"<code>py -3.12 -m venv .venv312</code><br>"
            f"<code>.venv312\\Scripts\\activate</code><br>"
            f"<code>pip install -r requirements.txt</code><br><br>"
            f"Ayrıntı için: Yapay Zekâ → Kurulum Durumu")

    def show_environment_status(self):
        versions = self._package_versions()
        rows = []
        for label, value in versions.items():
            if value:
                rows.append(f"<tr><td>{label}</td><td><b>{value}</b></td>"
                            f"<td style='color:#2e7d32;'>kurulu</td></tr>")
            else:
                rows.append(f"<tr><td>{label}</td><td>-</td>"
                            f"<td style='color:#c62828;'>kurulu değil</td></tr>")
        features = [
            ("Etiketleme (temel)", True, "PyQt5"),
            ("Görüntü iyileştirme, kenar katmanı, manyetik kutu",
             imaging.available(), "numpy"),
            ("CLAHE (yerel kontrast)", imaging.HAS_CV2, "opencv-python-headless"),
            ("Model destekli ön-etiketleme", inference.available(), "onnxruntime"),
        ]
        train_python = training.find_training_python(self.image_dir or "")
        feature_rows_extra = (
            f"<tr><td>Model eğitimi</td><td>"
            + ("<span style='color:#2e7d32;'>✓ ortam kurulu</span><br>"
               f"<code>{train_python}</code>" if train_python else
               "<span style='color:#ef6c00;'>ortam kurulu değil</span> "
               "(Yapay Zekâ → Model Eğit → Eğitim Ortamını Kur)")
            + "</td></tr>")
        feature_rows = []
        for name, ready, package in features:
            mark = ("<span style='color:#2e7d32;'>✓ açık</span>" if ready
                    else f"<span style='color:#c62828;'>✕ kapalı</span> "
                         f"(<code>pip install {package}</code>)")
            feature_rows.append(f"<tr><td>{name}</td><td>{mark}</td></tr>")

        errors = dict(imaging.IMPORT_ERRORS)
        errors.update(inference.IMPORT_ERRORS)
        error_block = ""
        if errors:
            items = "".join(f"<li><b>{name}</b>: <code>{message}</code></li>"
                            for name, message in errors.items())
            error_block = ("<br><b>İçe aktarma hataları</b><ul>" + items + "</ul>"
                           "Paketleri az önce kurduysanız programı kapatıp "
                           "yeniden açın.")

        QMessageBox.information(
            self, "Kurulum Durumu",
            "<b>Paketler</b><table cellpadding=4>" + "".join(rows) + "</table>"
            "<br><b>Özellikler</b><table cellpadding=4>"
            + "".join(feature_rows) + feature_rows_extra + "</table>" + error_block)

    def load_model(self):
        if not inference.available():
            QMessageBox.information(
                self, "Model desteği kurulu değil",
                self._missing_package_message(["onnxruntime", "numpy"]))
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "ONNX modeli seç", os.path.dirname(self._model_path or ""),
            "ONNX modeli (*.onnx)")
        if path:
            self._load_model_path(path, silent=False)

    def _load_model_path(self, path: str, silent: bool = False) -> bool:
        try:
            self._detector = inference.OnnxDetector(path)
        except Exception as exc:                     # model dosyası her şey olabilir
            self._detector = None
            if not silent:
                QMessageBox.critical(self, "Model yüklenemedi",
                                     f"Bu dosya bir YOLOv8 ONNX modeli olarak "
                                     f"açılamadı:\n{exc}")
            return False
        self._model_path = path
        names = self._detector.class_names
        if not silent:
            QMessageBox.information(
                self, "Model yüklendi",
                f"<b>{os.path.basename(path)}</b><br>"
                f"Giriş boyutu: {self._detector.size}×{self._detector.size}<br>"
                f"Sınıflar: {', '.join(names) if names else '(model adları taşımıyor)'}")
        self.status.showMessage(f"Model: {os.path.basename(path)}", 6000)
        return True

    def predict(self):
        if not inference.available():
            QMessageBox.information(
                self, "Model desteği kurulu değil",
                self._missing_package_message(["onnxruntime", "numpy"]))
            return
        if self._detector is None:
            if not self._model_path or not self._load_model_path(self._model_path, True):
                QMessageBox.information(self, "Model yok",
                                        "Önce Yapay Zekâ → ONNX Modeli Yükle.")
                return
        if not self.images:
            QMessageBox.information(self, "Klasör yok", "Önce bir resim klasörü açın.")
            return

        model_names = self._detector.class_names
        mapping = (inference.map_model_classes(model_names, self.classes.names())
                   if model_names else None)
        unlabeled = [p for p in self.images
                     if not yolo_io.has_labels(p, self.label_dir or "")]

        dialog = PredictDialog(self, self._model_path, model_names,
                               self.classes.names(), mapping or {},
                               len(unlabeled), len(self.images), self._conf, self._iou)
        if dialog.exec_() != PredictDialog.Accepted:
            return
        options = dialog.values()
        self._conf, self._iou = options["conf"], options["iou"]

        if options["scope"] == PredictDialog.SCOPE_CURRENT:
            self._predict_current(mapping)
        else:
            targets = unlabeled if options["scope"] == PredictDialog.SCOPE_UNLABELED \
                else list(self.images)
            self._predict_batch(targets, mapping)

    def _detect_on_path(self, path: str):
        image = QImage(path)
        if image.isNull():
            return None
        array = qimage_to_array(image)
        if array is None:
            return None
        return self._detector.predict(array, self._conf, self._iou)

    def _predict_current(self, mapping):
        if self.current_index < 0 or not self.canvas.has_image():
            return
        array = self.canvas.source_array()
        if array is None:
            QMessageBox.information(self, "numpy gerekli",
                                    "Tahmin için numpy kurulu olmalı.")
            return
        try:
            detections = self._detector.predict(array, self._conf, self._iou)
        except Exception as exc:
            QMessageBox.critical(self, "Tahmin başarısız", str(exc))
            return

        if mapping is None:
            detections = [(self.canvas.current_class, *rest) for _, *rest in detections]
            added = self.canvas.add_proposals(detections, None)
        else:
            added = self.canvas.add_proposals(detections, mapping)

        self._update_box_list()
        self._refresh_notice()
        self.status.showMessage(
            f"{added} öneri eklendi (güven ≥ {self._conf:.2f}). "
            f"Enter ile kabul, Esc ile sil.", 8000)

    def _predict_batch(self, targets: List[str], mapping):
        if not targets:
            return
        if self.canvas.is_dirty() and not self._save_before_move():
            return

        progress = QProgressDialog("Model çalışıyor...", "İptal", 0, len(targets), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        written = boxes_total = failed = 0
        pending = set(self._pending_review)

        for index, path in enumerate(targets, start=1):
            progress.setValue(index - 1)
            progress.setLabelText(f"{os.path.basename(path)}\n({index}/{len(targets)})")
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                detections = self._detect_on_path(path)
            except Exception:
                detections = None
            if detections is None:
                failed += 1
                continue

            rows = []
            for class_id, x1, y1, x2, y2, _score in detections:
                target_class = class_id if mapping is None else mapping.get(class_id)
                if target_class is None:
                    continue
                rows.append((target_class, x1, y1, x2, y2))

            reader = QImageReader(path)
            size = reader.size()
            if not size.isValid():
                failed += 1
                continue
            txt_path = yolo_io.label_path_for(path, self.label_dir)
            try:
                yolo_io.backup_label(txt_path)
                yolo_io.save_labels(txt_path, rows, size.width(), size.height())
            except OSError:
                failed += 1
                continue
            pending.add(os.path.basename(path))
            written += 1
            boxes_total += len(rows)

        progress.close()
        review.save_pending(self.label_dir, pending)
        self._pending_review = pending
        self._refresh_image_list_marks()
        if self.current_index >= 0:
            self._load_image_at(self.current_index, force=True)

        QMessageBox.information(
            self, "Toplu tahmin bitti",
            f"{written} resim işlendi, {boxes_total} kutu yazıldı."
            + (f"<br>{failed} resim atlandı." if failed else "")
            + "<br><br>Bu resimler listede <b>?</b> ile işaretli; açıp "
              "onayladıkça işaret kalkar.")

    def accept_proposals(self):
        count = self.canvas.accept_proposals()
        if count:
            self._update_box_list()
            self._refresh_notice()
            self.status.showMessage(f"{count} öneri kabul edildi.", 3000)

    def discard_proposals(self):
        count = self.canvas.discard_proposals()
        if count:
            self._update_box_list()
            self._refresh_notice()
            self.status.showMessage(f"{count} öneri silindi.", 3000)

    # ================================================================ yardım
    def show_shortcuts(self):
        ShortcutsDialog(self).exec_()

    def show_about(self):
        QMessageBox.about(
            self, "Hakkında",
            f"<b>{APP_TITLE}</b><br><br>"
            "PyQt5 ile yazılmış YOLOv8 uyumlu etiketleme istasyonu.<br>"
            "Etiketler <code>&lt;class_id&gt; &lt;x_center&gt; &lt;y_center&gt; "
            "&lt;width&gt; &lt;height&gt;</code> biçiminde, 0-1 aralığına "
            "normalize edilerek kaydedilir.<br><br>"
            f"Görüntü işleme: {imaging.backend_name()}<br>"
            f"Model çıkarımı: "
            f"{'onnxruntime hazır' if inference.available() else inference.unavailable_reason()}"
            "<br><br>Kısayollar için <b>F1</b>.")

    def _on_escape(self):
        if self.canvas.cancel_drawing():
            self.status.showMessage("Çizim iptal edildi.", 2000)
            return
        if self.canvas.proposals():
            self.discard_proposals()
            return
        if self.canvas.mode == Mode.DRAW:
            self.canvas.set_mode(Mode.EDIT)
        else:
            self.canvas.clear_selection()

    def _on_mode_changed(self, mode: int):
        self.act_draw.setChecked(mode == Mode.DRAW)
        self.lbl_mode.setText("Mod: Çizim (W)" if mode == Mode.DRAW else "Mod: Düzenle")
        if mode == Mode.DRAW and not self.canvas.boxes_visible():
            self.canvas.set_boxes_visible(True)
            self.act_toggle_boxes.setChecked(False)

    # ================================================================ durum
    def _on_cursor_moved(self, x: float, y: float):
        self.lbl_cursor.setText(f"x: {int(x)}, y: {int(y)}")

    def _update_status(self):
        if self.canvas.has_image():
            width, height = self.canvas.image_size()
            self.lbl_image.setText(f"Resim: {width}x{height} px")
            self.lbl_zoom.setText(f"Zoom: %{self.canvas.current_scale() * 100:.0f}")
        else:
            self.lbl_image.setText("Resim: -")
            self.lbl_zoom.setText("Zoom: -")

        proposals = len(self.canvas.proposals())
        text = f"Kutu: {len(self.canvas.real_boxes())}"
        if proposals:
            text += f" (+{proposals} öneri)"
        self.lbl_boxes.setText(text)

        minutes = max(1e-6, (time.time() - self._session_start) / 60.0)
        self.lbl_session.setText(
            f"Oturum: {self._session_boxes} kutu · {self._session_boxes / minutes:.1f}/dk")
        self._update_box_list()

    def _update_title(self):
        if self.current_index < 0 or not self.images:
            self.setWindowTitle(APP_TITLE)
            return
        name = os.path.basename(self.images[self.current_index])
        star = " *" if self.canvas.is_dirty() else ""
        self.setWindowTitle(
            f"{name}{star}  ({self.current_index + 1}/{len(self.images)})  -  {APP_TITLE}")

    # ================================================================ olaylar
    def keyPressEvent(self, event):
        plain = not (event.modifiers()
                     & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
        key = event.key()
        if plain and Qt.Key_1 <= key <= Qt.Key_9:
            index = key - Qt.Key_1
            if index < len(self.classes):
                self.class_list.setCurrentRow(index)
                self.canvas.current_class = index
                self._update_active_class_label()
                if self.canvas.assign_class_to_selection(index):
                    self._update_box_list()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.canvas.is_dirty():
            if self.act_autosave.isChecked():
                self.save_current(silent=True)
            if self.canvas.is_dirty() and not self._maybe_save_before_leaving():
                event.ignore()
                return
        self._save_settings()
        event.accept()
