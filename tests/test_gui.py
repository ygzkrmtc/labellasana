"""
Arayüz (Qt) davranış testleri — ekran gerektirmez.

Qt'nin 'offscreen' sürücüsüyle gerçek MainWindow ve Canvas nesnelerini kurar,
kullanıcı eylemlerini programatik olarak tetikler ve sonucu doğrular.

Çalıştırma:
    python -m tests.test_gui

Bu dosya özellikle şu senaryoları kilitler:
  * sınıf silindiğinde sahipsiz kutu KALMAMALI (bildirilen hata)
  * boş classes.txt yetkili olmalı (silinen sınıf geri gelmemeli)
  * kaydet/oku turunda kutu koordinatları korunmalı
  * kilitli kutular silinmemeli/taşınmamalı
"""

import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtGui import QColor, QImage, QPainter
    from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
    HAS_QT = True
except ImportError:                                    # pragma: no cover
    HAS_QT = False

if HAS_QT:
    from app import main_window as mw
    from app import yolo_io
    from app.canvas import Mode
    from app.dialogs import DeleteClassDialog

APP = None


def setUpModule():
    global APP
    if HAS_QT:
        APP = QApplication.instance() or QApplication([])


class FakeSettings:
    """Testler kullanıcının gerçek ayarlarını kirletmesin."""

    _store = {}

    def __init__(self, *args, **kwargs):
        self._data = {}

    def value(self, key, default=None, type=None):
        value = self._data.get(key, default)
        if type is bool:
            return bool(value)
        if type is float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        if type is str:
            return "" if value is None else str(value)
        return value

    def setValue(self, key, value):
        self._data[key] = value


class FakePicker:
    """LabelPickerDialog yerine: her zaman verilen adı seçer."""

    Accepted = QDialog.Accepted if HAS_QT else 1
    next_name = "aku"
    next_result = None          # None -> Accepted

    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return (FakePicker.next_result if FakePicker.next_result is not None
                else FakePicker.Accepted)

    def chosen_name(self):
        return FakePicker.next_name


class FakeDeleteDialog:
    """DeleteClassDialog yerine: önceden belirlenmiş seçimi döndürür."""

    Accepted = QDialog.Accepted if HAS_QT else 1
    MOVE_BOXES = "move"
    DELETE_BOXES = "delete"
    next_action = "delete"
    next_target = None
    next_result = None

    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return (FakeDeleteDialog.next_result if FakeDeleteDialog.next_result is not None
                else FakeDeleteDialog.Accepted)

    def values(self):
        return {"action": FakeDeleteDialog.next_action,
                "target_name": FakeDeleteDialog.next_target}


class _FakeClassDialog:
    """ClassDialog yerine: verilen adı ve rengi döndürür."""

    Accepted = QDialog.Accepted if HAS_QT else 1
    next_name = "aku"
    next_color = "#ff0000"

    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return _FakeClassDialog.Accepted

    def values(self):
        return _FakeClassDialog.next_name, _FakeClassDialog.next_color


_REAL_CLASS_DIALOG = mw.ClassDialog if HAS_QT else None


def make_image(path, width=200, height=160, with_rectangle=True):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(30, 30, 30))
    if with_rectangle:
        painter = QPainter(image)
        painter.fillRect(int(width * 0.25), int(height * 0.25),
                         int(width * 0.5), int(height * 0.5), QColor(240, 240, 240))
        painter.end()
    image.save(path, "PNG")


@unittest.skipUnless(HAS_QT, "PyQt5 kurulu değil")
class GuiTestCase(unittest.TestCase):
    """Her test için taze bir pencere ve taze bir resim klasörü."""

    IMAGE_COUNT = 3

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.images = []
        for index in range(self.IMAGE_COUNT):
            path = os.path.join(self.dir, f"img{index}.png")
            make_image(path)
            self.images.append(path)
        self.labels = os.path.join(self.dir, "labels")

        # Modal pencereleri sustur
        self._saved = {
            "settings": mw.QSettings,
            "picker": mw.LabelPickerDialog,
            "delete": mw.DeleteClassDialog,
            "question": QMessageBox.question,
            "information": QMessageBox.information,
            "warning": QMessageBox.warning,
            "critical": QMessageBox.critical,
        }
        mw.QSettings = FakeSettings
        mw.LabelPickerDialog = FakePicker
        mw.DeleteClassDialog = FakeDeleteDialog
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
        QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
        QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)

        FakePicker.next_result = None
        FakePicker.next_name = "aku"
        FakeDeleteDialog.next_result = None
        FakeDeleteDialog.next_action = FakeDeleteDialog.DELETE_BOXES
        FakeDeleteDialog.next_target = None

        self.window = mw.MainWindow()
        self.window.resize(900, 600)
        self.window.show()          # offscreen; alt widget'ların görünürlüğü için
        APP.processEvents()

    def tearDown(self):
        self.window.canvas.set_dirty(False)
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()
        mw.QSettings = self._saved["settings"]
        mw.LabelPickerDialog = self._saved["picker"]
        mw.DeleteClassDialog = self._saved["delete"]
        QMessageBox.question = self._saved["question"]
        QMessageBox.information = self._saved["information"]
        QMessageBox.warning = self._saved["warning"]
        QMessageBox.critical = self._saved["critical"]
        shutil.rmtree(self.dir, ignore_errors=True)

    # ------------------------------------------------------------ yardımcılar
    def open_folder(self):
        self.assertTrue(self.window.load_folder(self.dir))
        return self.window

    def add_class(self, name, color="#ff0000"):
        index = self.window.classes.add(name, color)
        self.window._update_class_list()
        self.window._persist_classes()
        return index

    def add_box(self, x1, y1, x2, y2, class_id=0):
        return self.window.canvas.add_box(QRectF(x1, y1, x2 - x1, y2 - y1), class_id)


class TestStartup(GuiTestCase):
    def test_window_builds_without_folder(self):
        self.assertEqual(self.window.image_list.count(), 0)
        self.assertEqual(len(self.window.classes), 0)
        self.assertFalse(self.window.canvas.has_image())

    def test_open_folder_lists_images(self):
        self.open_folder()
        self.assertEqual(self.window.image_list.count(), self.IMAGE_COUNT)
        self.assertTrue(self.window.canvas.has_image())
        self.assertEqual(self.window.current_index, 0)

    def test_broken_file_is_skipped(self):
        with open(os.path.join(self.dir, "bozuk.png"), "wb") as handle:
            handle.write(b"bu bir resim degil")
        self.open_folder()
        self.assertEqual(self.window.image_list.count(), self.IMAGE_COUNT)

    def test_title_shows_progress(self):
        self.open_folder()
        self.assertIn(f"(1/{self.IMAGE_COUNT})", self.window.windowTitle())


class TestFirstClassFlow(GuiTestCase):
    """İlk sınıf eklendikten sonra kullanıcı doğrudan çizmeye başlayabilmeli."""

    def test_adding_first_class_enters_draw_mode(self):
        self.open_folder()
        self.assertEqual(self.window.canvas.mode, Mode.EDIT)
        mw.ClassDialog = _FakeClassDialog
        try:
            _FakeClassDialog.next_name = "aku"
            self.window.add_class()
        finally:
            mw.ClassDialog = _REAL_CLASS_DIALOG
        self.assertEqual(len(self.window.classes), 1)
        self.assertEqual(self.window.canvas.mode, Mode.DRAW,
                         "İlk sınıftan sonra çizim modu açılmalıydı")

    def test_second_class_does_not_force_draw_mode(self):
        self.open_folder()
        self.add_class("aku")
        self.window.canvas.set_mode(Mode.EDIT)
        mw.ClassDialog = _FakeClassDialog
        try:
            _FakeClassDialog.next_name = "msda"
            self.window.add_class()
        finally:
            mw.ClassDialog = _REAL_CLASS_DIALOG
        self.assertEqual(self.window.canvas.mode, Mode.EDIT)


class TestOptionalPackages(GuiTestCase):
    """Eksik kütüphane menüyü griye almamalı; tıklayınca açıklama çıkmalı."""

    def test_ai_actions_stay_clickable(self):
        self.assertTrue(self.window.act_load_model.isEnabled())
        self.assertTrue(self.window.act_predict.isEnabled())

    def test_imaging_actions_stay_clickable(self):
        for action in (self.window.act_enhance, self.window.act_edges,
                       self.window.act_magnetic, self.window.act_snap):
            self.assertTrue(action.isEnabled())

    def test_environment_status_reports_packages(self):
        versions = self.window._package_versions()
        self.assertIn("Python", versions)
        self.assertIn("numpy", versions)
        self.assertIn("onnxruntime", versions)
        self.assertTrue(versions["PyQt5"])

    def test_missing_package_message_has_command(self):
        message = self.window._missing_package_message(["onnxruntime"])
        self.assertIn("pip install onnxruntime", message)


class TestSaveLoadRoundTrip(GuiTestCase):
    def test_box_survives_save_and_reload(self):
        self.open_folder()
        self.add_class("aku")
        self.add_box(20, 30, 120, 110)
        self.assertTrue(self.window.save_current(silent=True))

        txt = os.path.join(self.labels, "img0.txt")
        self.assertTrue(os.path.isfile(txt))
        rows = yolo_io.load_raw(txt)
        self.assertEqual(len(rows), 1)
        for value in rows[0][1:]:
            self.assertTrue(0.0 <= value <= 1.0)

        # Başka resme geçip geri dön
        self.window._select_row(1)
        self.window._select_row(0)
        boxes = self.window.canvas.boxes()
        self.assertEqual(len(boxes), 1)
        rect = boxes[0].rect()
        self.assertAlmostEqual(rect.left(), 20, delta=1)
        self.assertAlmostEqual(rect.top(), 30, delta=1)
        self.assertAlmostEqual(rect.right(), 120, delta=1)
        self.assertAlmostEqual(rect.bottom(), 110, delta=1)

    def test_empty_file_written_when_all_boxes_removed(self):
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        self.window.canvas.clear_boxes()
        self.window.save_current(silent=True)
        txt = os.path.join(self.labels, "img0.txt")
        self.assertTrue(os.path.isfile(txt))
        self.assertEqual(os.path.getsize(txt), 0)

    def test_classes_txt_written(self):
        self.open_folder()
        self.add_class("aku")
        self.add_class("msda")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        names = yolo_io.load_classes(os.path.join(self.labels, "classes.txt"))
        self.assertEqual(names, ["aku", "msda"])

    def test_backup_created_on_second_save(self):
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        self.add_box(60, 60, 90, 90)
        self.window.save_current(silent=True)
        txt = os.path.join(self.labels, "img0.txt")
        self.assertGreaterEqual(len(yolo_io.list_backups(txt)), 1)


class TestClassDeletionBug(GuiTestCase):
    """Bildirilen hata: sınıf silinince isimsiz ('?0') kutular kalıyordu."""

    def test_deleting_only_class_removes_its_boxes(self):
        self.open_folder()
        self.add_class("aku")
        for i in range(4):
            self.add_box(10 + i * 20, 10, 40 + i * 20, 60)
        self.assertEqual(self.window.canvas.box_count(), 4)

        self.window.class_list.setCurrentRow(0)
        FakeDeleteDialog.next_action = FakeDeleteDialog.DELETE_BOXES
        self.window.delete_class()

        self.assertEqual(len(self.window.classes), 0)
        self.assertEqual(self.window.canvas.box_count(), 0,
                         "Sınıf silindi ama kutular sahipsiz kaldı")
        self.assertEqual(self.window.canvas.orphan_boxes(), [])

    def test_deleting_class_can_move_boxes(self):
        self.open_folder()
        self.add_class("aku")
        self.add_class("msda")
        self.add_box(10, 10, 50, 50, class_id=0)
        self.add_box(60, 10, 90, 50, class_id=1)

        self.window.class_list.setCurrentRow(0)
        FakeDeleteDialog.next_action = FakeDeleteDialog.MOVE_BOXES
        FakeDeleteDialog.next_target = "msda"
        self.window.delete_class()

        self.assertEqual(self.window.classes.names(), ["msda"])
        self.assertEqual(self.window.canvas.box_count(), 2)
        for box in self.window.canvas.boxes():
            self.assertEqual(box.class_id, 0)
        self.assertEqual(self.window.canvas.orphan_boxes(), [])

    def test_disk_labels_are_remapped_after_delete(self):
        self.open_folder()
        self.add_class("aku")
        self.add_class("msda")
        self.add_box(10, 10, 50, 50, class_id=0)
        self.add_box(60, 10, 90, 50, class_id=1)
        self.window.save_current(silent=True)

        # img1'e de iki sınıflı etiket yaz
        self.window._select_row(1)
        self.add_box(10, 10, 50, 50, class_id=1)
        self.window.save_current(silent=True)

        self.window._select_row(0)
        self.window.class_list.setCurrentRow(0)
        FakeDeleteDialog.next_action = FakeDeleteDialog.DELETE_BOXES
        self.window.delete_class()          # 'aku' gitti, 'msda' 1 -> 0 olmalı

        rows = yolo_io.load_raw(os.path.join(self.labels, "img1.txt"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 0, "Diskteki kimlikler yeniden numaralanmadı")

    def test_unused_class_deleted_without_dialog(self):
        self.open_folder()
        self.add_class("aku")
        self.add_class("kullanilmayan")
        self.window.class_list.setCurrentRow(1)
        FakeDeleteDialog.next_result = QDialog.Rejected   # açılırsa test düşer
        self.window.delete_class()
        self.assertEqual(self.window.classes.names(), ["aku"])

    def test_deleted_class_does_not_come_back(self):
        """Boş classes.txt yetkili olmalı."""
        self.open_folder()
        self.add_class("gr")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)

        self.window.class_list.setCurrentRow(0)
        FakeDeleteDialog.next_action = FakeDeleteDialog.DELETE_BOXES
        self.window.delete_class()
        self.assertEqual(len(self.window.classes), 0)

        # Klasörü yeniden aç
        self.window.canvas.set_dirty(False)
        self.window.load_folder(self.dir)
        self.assertEqual(len(self.window.classes), 0,
                         "Silinen sınıf klasör yeniden açılınca geri geldi")


class TestOrphanRecovery(GuiTestCase):
    """Sahipsiz kutu oluşursa kullanıcı sıkışmamalı."""

    def _make_orphan(self):
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50, class_id=0)
        box = self.add_box(60, 10, 90, 50, class_id=0)
        box.class_id = 7                     # yapay olarak sahipsiz bırak
        self.window._refresh_notice()
        return box

    def test_orphans_are_detected(self):
        self._make_orphan()
        self.assertEqual(len(self.window.canvas.orphan_boxes()), 1)
        self.assertTrue(self.window.notice.isVisible())

    def test_save_is_blocked_while_orphans_exist(self):
        self._make_orphan()
        self.assertFalse(self.window.save_current(silent=True),
                         "Sahipsiz kutu varken kayıt yapılmamalı")

    def test_assigning_class_fixes_orphans(self):
        self._make_orphan()
        FakePicker.next_name = "aku"
        self.window.fix_orphan_boxes()
        self.assertEqual(self.window.canvas.orphan_boxes(), [])
        self.assertTrue(self.window.save_current(silent=True))

    def test_deleting_orphans_fixes_them(self):
        self._make_orphan()
        self.window.delete_orphan_boxes()
        self.assertEqual(self.window.canvas.orphan_boxes(), [])
        self.assertEqual(self.window.canvas.box_count(), 1)


class TestBoxOperations(GuiTestCase):
    def setUp(self):
        super().setUp()
        self.open_folder()
        self.add_class("aku")
        self.add_class("msda")

    def test_undo_restores_deleted_box(self):
        box = self.add_box(10, 10, 50, 50)
        box.setSelected(True)
        self.window.delete_selected_box()
        self.assertEqual(self.window.canvas.box_count(), 0)
        self.window.undo()
        self.assertEqual(self.window.canvas.box_count(), 1)

    def test_undo_restores_geometry(self):
        box = self.add_box(10, 10, 50, 50)
        box.setSelected(True)
        self.window.canvas.nudge_selection(5, 5)
        self.assertAlmostEqual(self.window.canvas.boxes()[0].rect().left(), 15, delta=0.01)
        self.window.undo()
        self.assertAlmostEqual(self.window.canvas.boxes()[0].rect().left(), 10, delta=0.01)

    def test_nudge_respects_image_bounds(self):
        box = self.add_box(0, 0, 40, 40)
        box.setSelected(True)
        self.window.canvas.nudge_selection(-50, -50)
        rect = self.window.canvas.boxes()[0].rect()
        self.assertGreaterEqual(rect.left(), 0)
        self.assertGreaterEqual(rect.top(), 0)

    def test_locked_box_is_not_deleted_or_moved(self):
        box = self.add_box(10, 10, 50, 50)
        box.setSelected(True)
        self.window.toggle_lock()
        self.assertTrue(self.window.canvas.boxes()[0].locked)

        self.window.delete_selected_box()
        self.assertEqual(self.window.canvas.box_count(), 1, "Kilitli kutu silindi")

        self.window.canvas.nudge_selection(10, 10)
        self.assertAlmostEqual(self.window.canvas.boxes()[0].rect().left(), 10, delta=0.01)

    def test_select_all_and_bulk_class_change(self):
        self.add_box(10, 10, 50, 50, class_id=0)
        self.add_box(60, 10, 90, 50, class_id=0)
        self.window.select_all_boxes()
        self.assertEqual(len(self.window.canvas.selected_boxes()), 2)
        self.window.canvas.assign_class_to_selection(1)
        self.assertTrue(all(b.class_id == 1 for b in self.window.canvas.boxes()))

    def test_class_visibility_filter(self):
        self.add_box(10, 10, 50, 50, class_id=0)
        self.add_box(60, 10, 90, 50, class_id=1)
        canvas = self.window.canvas
        canvas.set_hidden_classes({0})
        visible = [b for b in canvas.boxes() if not canvas.is_box_hidden(b)]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].class_id, 1)
        self.assertFalse(canvas.boxes()[0].isVisible())
        canvas.set_hidden_classes(set())
        self.assertEqual(
            len([b for b in canvas.boxes() if not canvas.is_box_hidden(b)]), 2)

    def test_hide_all_boxes(self):
        canvas = self.window.canvas
        self.add_box(10, 10, 50, 50)
        canvas.set_boxes_visible(False)
        self.assertTrue(canvas.is_box_hidden(canvas.boxes()[0]))
        self.assertFalse(canvas.boxes()[0].isVisible())
        canvas.set_boxes_visible(True)
        self.assertFalse(canvas.is_box_hidden(canvas.boxes()[0]))

    def test_copy_paste_scales_between_sizes(self):
        # 400x320'lik (iki kat) bir resim ekleyip klasörü yeniden yükle
        big = os.path.join(self.dir, "big.png")
        make_image(big, 400, 320)
        self.window.canvas.set_dirty(False)
        self.window.load_folder(self.dir)
        # Klasör yeniden yüklenince sınıflar classes.txt'ten geri gelir;
        # yoksa (henüz kaydedilmemişse) ekle.
        if not len(self.window.classes):
            self.add_class("aku")

        small_index = self.window.images.index(self.images[0])
        self.window._select_row(small_index)
        self.add_box(20, 20, 60, 60)
        self.window.copy_boxes()

        self.window._select_row(self.window.images.index(big))
        self.window.paste_boxes()
        boxes = self.window.canvas.boxes()
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].rect().left(), 40, delta=1)   # 2 kat ölçek
        self.assertAlmostEqual(boxes[0].rect().right(), 120, delta=1)

    def test_clear_all_boxes(self):
        self.add_box(10, 10, 50, 50)
        self.add_box(60, 10, 90, 50)
        self.window.clear_all_boxes()
        self.assertEqual(self.window.canvas.box_count(), 0)

    def test_mode_toggle(self):
        self.assertEqual(self.window.canvas.mode, Mode.EDIT)
        self.window.canvas.toggle_mode()
        self.assertEqual(self.window.canvas.mode, Mode.DRAW)
        self.window._on_escape()
        self.assertEqual(self.window.canvas.mode, Mode.EDIT)


class TestNavigation(GuiTestCase):
    def setUp(self):
        super().setUp()
        self.open_folder()
        self.add_class("aku")

    def test_next_and_prev(self):
        self.window.go_next()
        self.assertEqual(self.window.current_index, 1)
        self.window.go_prev()
        self.assertEqual(self.window.current_index, 0)

    def test_next_stops_at_end(self):
        for _ in range(10):
            self.window.go_next()
        self.assertEqual(self.window.current_index, self.IMAGE_COUNT - 1)

    def test_autosave_on_navigation(self):
        self.add_box(10, 10, 50, 50)
        self.window.act_autosave.setChecked(True)
        self.window.go_next()
        self.assertTrue(os.path.isfile(os.path.join(self.labels, "img0.txt")))

    def test_next_unlabeled_skips_labeled(self):
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)      # img0 etiketlendi
        self.window.go_next_unlabeled()
        self.assertNotEqual(self.window.current_index, 0)

    def test_only_unlabeled_filter_hides_rows(self):
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        self.window._select_row(1)
        self.window.only_unlabeled.setChecked(True)
        self.assertTrue(self.window.image_list.item(0).isHidden())
        self.window.only_unlabeled.setChecked(False)
        self.assertFalse(self.window.image_list.item(0).isHidden())


class TestProposals(GuiTestCase):
    def setUp(self):
        super().setUp()
        self.open_folder()
        self.add_class("aku")

    def test_proposals_are_not_saved_until_accepted(self):
        self.window.canvas.add_proposals([(0, 10, 10, 50, 50, 0.9)], None)
        self.assertEqual(len(self.window.canvas.proposals()), 1)
        self.assertEqual(len(self.window.canvas.real_boxes()), 0)

        # save_current önerileri sorar; testte 'Yes' -> kabul et
        self.window.save_current(silent=True)
        rows = yolo_io.load_raw(os.path.join(self.labels, "img0.txt"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(self.window.canvas.proposals()), 0)

    def test_accept_and_discard(self):
        self.window.canvas.add_proposals(
            [(0, 10, 10, 50, 50, 0.9), (0, 60, 10, 90, 50, 0.5)], None)
        self.assertEqual(self.window.canvas.accept_proposals(), 2)
        self.assertEqual(len(self.window.canvas.real_boxes()), 2)

        self.window.canvas.add_proposals([(0, 10, 60, 40, 90, 0.7)], None)
        self.assertEqual(self.window.canvas.discard_proposals(), 1)
        self.assertEqual(len(self.window.canvas.real_boxes()), 2)

    def test_unmapped_classes_are_skipped(self):
        added = self.window.canvas.add_proposals(
            [(0, 10, 10, 50, 50, 0.9), (3, 60, 10, 90, 50, 0.8)], {0: 0})
        self.assertEqual(added, 1)

    def test_escape_discards_proposals(self):
        self.window.canvas.add_proposals([(0, 10, 10, 50, 50, 0.9)], None)
        self.window._on_escape()
        self.assertEqual(len(self.window.canvas.proposals()), 0)


class TestImagingIntegration(GuiTestCase):
    def setUp(self):
        super().setUp()
        self.open_folder()
        self.add_class("aku")

    def test_enhancement_does_not_move_boxes(self):
        box = self.add_box(20, 20, 80, 80)
        before = QRectF(box.rect())
        self.window.enhance_bar.set_values(brightness=60, contrast=20, gamma=1.5)
        self.window._apply_enhancement_to_canvas()
        after = self.window.canvas.boxes()[0].rect()
        self.assertEqual(before, after, "İyileştirme etiket koordinatlarını değiştirdi")

    def test_enhancement_is_display_only(self):
        path = self.window.images[0]
        before = os.path.getsize(path)
        self.window.enhance_bar.set_values(brightness=80)
        self.window._apply_enhancement_to_canvas()
        self.assertEqual(os.path.getsize(path), before, "Kaynak dosya değiştirildi")

    def test_edge_overlay_toggle(self):
        self.window.act_edges.setChecked(True)
        self.window._toggle_edges(True)
        self.window.act_edges.setChecked(False)
        self.window._toggle_edges(False)      # patlamamalı

    def test_snap_moves_box_towards_edges(self):
        from app import imaging
        if not imaging.available():
            self.skipTest("numpy yok")
        # Test resminde beyaz dikdörtgen 50..150 x 40..120 arasında
        box = self.add_box(44, 34, 156, 126)
        box.setSelected(True)
        self.window.snap_selection()
        rect = self.window.canvas.boxes()[0].rect()
        self.assertLessEqual(abs(rect.left() - 50), 6)
        self.assertLessEqual(abs(rect.top() - 40), 6)


class TestDialogsConstruct(GuiTestCase):
    """Diyaloglar exec_ edilmeden kurulabilmeli (kurulum hatası avı)."""

    def test_shortcuts_dialog(self):
        from app.dialogs import ShortcutsDialog
        dialog = ShortcutsDialog(self.window)
        self.assertTrue(dialog.windowTitle())

    def test_stats_dialog(self):
        from app import dataset
        from app.dataset_dialogs import StatsDialog
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        stats = dataset.collect_stats(self.window.images, self.labels, 1)
        dialog = StatsDialog(self.window, stats, ["aku"], ["#ff0000"])
        self.assertTrue(dialog.windowTitle())

    def test_export_dialog(self):
        from app.dataset_dialogs import ExportDialog
        dialog = ExportDialog(self.window, self.dir, 10)
        values = dialog.values()
        self.assertEqual(sum(values["ratios"]), 100)

    def test_audit_dialog(self):
        from app import audit
        from app.audit_dialog import AuditDialog
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        findings = audit.audit(self.window.images, self.labels, ["aku"])
        dialog = AuditDialog(self.window, findings)
        self.assertTrue(dialog.windowTitle())

    def test_predict_dialog(self):
        from app.ai_dialogs import PredictDialog
        dialog = PredictDialog(self.window, "model.onnx", ["aku"], ["aku"],
                               {0: 0}, 2, 5)
        self.assertIn("scope", dialog.values())

    def test_delete_class_dialog(self):
        dialog = DeleteClassDialog(self.window, "aku", 5, ["msda"])
        self.assertEqual(dialog.values()["action"], DeleteClassDialog.DELETE_BOXES)

    def test_class_dialog(self):
        from app.dialogs import ClassDialog
        dialog = ClassDialog(self.window, "aku", "#ff0000")
        self.assertEqual(dialog.values()[0], "aku")


class TestThemeAndSettings(GuiTestCase):
    def test_theme_switch(self):
        from app import theme
        self.window._apply_theme(theme.LIGHT)
        self.assertEqual(self.window._theme, theme.LIGHT)
        self.window._apply_theme(theme.DARK)
        self.assertEqual(self.window._theme, theme.DARK)

    def test_project_round_trip(self):
        from app import project
        self.open_folder()
        self.add_class("aku", "#123456")
        document = project.build(self.window.image_dir, self.window.label_dir,
                                 list(self.window.classes), "", {"conf": 0.4})
        path = project.save(os.path.join(self.dir, "proje"), document)
        loaded = project.load(path)
        self.assertEqual(loaded["classes"][0]["name"], "aku")
        self.assertEqual(loaded["classes"][0]["color"], "#123456")
        self.assertTrue(project.is_usable(loaded))


class TestTrainingIntegration(GuiTestCase):
    """Model kütüphanesi ve eğitim akışının arayüz tarafı."""

    def _fake_model(self, name="aku_v1"):
        from app import training
        library = self.window.model_library()
        library.ensure()
        source = os.path.join(self.dir, "best.onnx")
        with open(source, "wb") as handle:
            handle.write(b"onnx")
        target = library.register(source, name)
        library.write_metadata(target, {"classes": ["aku"], "map50": 0.77,
                                        "created": "2026-01-01 10:00"})
        return target

    def test_models_menu_without_folder(self):
        self.window._rebuild_models_menu()
        actions = self.window.models_menu.actions()
        self.assertTrue(actions)
        self.assertFalse(actions[0].isEnabled())

    def test_models_menu_lists_library(self):
        self.open_folder()
        self.add_class("aku")
        target = self._fake_model()
        self.window._rebuild_models_menu()
        texts = [a.text() for a in self.window.models_menu.actions()]
        self.assertTrue(any("aku_v1" in text for text in texts),
                        f"Model menüde görünmedi: {texts}")
        self.assertTrue(os.path.isfile(target))

    def test_models_menu_offers_training_when_empty(self):
        self.open_folder()
        self.window._rebuild_models_menu()
        texts = [a.text() for a in self.window.models_menu.actions()]
        self.assertTrue(any("eğit" in text.lower() for text in texts))

    def test_labeled_image_count(self):
        self.open_folder()
        self.add_class("aku")
        self.assertEqual(self.window._labeled_image_count(), 0)
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        self.assertEqual(self.window._labeled_image_count(), 1)

    def test_training_dataset_refuses_when_too_few_labels(self):
        self.open_folder()
        self.add_class("aku")
        self.add_box(10, 10, 50, 50)
        self.window.save_current(silent=True)
        self.assertIsNone(self.window._build_training_dataset())

    def test_training_dataset_builds_data_yaml(self):
        from app import training
        # 6 resim: 4'ten fazla etiketli gerekiyor
        for index in range(3, 8):
            make_image(os.path.join(self.dir, f"img{index}.png"))
        self.window.canvas.set_dirty(False)
        self.open_folder()
        if not len(self.window.classes):
            self.add_class("aku")

        for row in range(len(self.window.images)):
            self.window._select_row(row)
            self.add_box(10, 10, 60, 60)
            self.window.save_current(silent=True)

        data_yaml = self.window._build_training_dataset()
        self.assertIsNotNone(data_yaml)
        self.assertTrue(os.path.isfile(data_yaml))
        with open(data_yaml, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("train: images/train", content)
        self.assertIn("val: images/val", content)
        dataset_root = training.dataset_dir(self.window.image_dir)
        self.assertTrue(os.path.isdir(os.path.join(dataset_root, "images", "train")))

    def test_train_dialog_constructs_without_environment(self):
        from app.train_dialog import TrainDialog
        self.open_folder()
        self.add_class("aku")
        dialog = TrainDialog(self.window, self.window.image_dir,
                             self.window.classes.names(), 0,
                             lambda: None)
        # Ortam yokken başlat düğmesi kapalı, kurulum düğmesi açık olmalı
        self.assertFalse(dialog.start_button.isEnabled())
        self.assertTrue(dialog.name_edit.text().startswith("aku_v"))
        dialog.close()

    def test_environment_status_dialog_runs(self):
        self.window.show_environment_status()      # patlamamalı


class TestReviewQueue(GuiTestCase):
    def test_pending_marks_and_clears(self):
        from app import review
        self.open_folder()
        self.add_class("aku")
        os.makedirs(self.labels, exist_ok=True)
        review.mark(self.labels, "img0.png")
        self.window._reload_review_queue()
        self.window._refresh_image_list_marks()
        self.assertIn("img0.png", self.window._pending_review)
        self.assertTrue(self.window.image_list.item(0).text().startswith("?"))

        self.window._select_row(0)
        self.window.confirm_review()
        self.assertNotIn("img0.png", self.window._pending_review)


if __name__ == "__main__":
    unittest.main(verbosity=2)
