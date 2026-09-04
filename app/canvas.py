"""
Çizim tuvali: QGraphicsView / QGraphicsScene tabanlı.

Neden QGraphicsView? Büyük resimlerde ve çok sayıda kutuda QLabel + paintEvent
yaklaşımı her karede tüm resmi yeniden ölçeklendirir. QGraphicsView, sahne
dönüşümünü verimli uygular, sadece kirlenen bölgeyi yeniden çizer ve zoom/pan'i
doğal olarak destekler.

Koordinat sözleşmesi (tüm hataların kaynağı burasıdır, bu yüzden tek kural):
    BoxItem.pos() daima (0, 0); BoxItem.rect() doğrudan RESİM PİKSELİ
    koordinatlarındadır. Sahne dikdörtgeni de resmin kendisidir.
"""

from typing import List, Optional, Sequence, Set

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QFontMetrics, QImage, QPainter,
                         QPainterPath, QPen, QPixmap)
from PyQt5.QtWidgets import (QFrame, QGraphicsItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsScene, QGraphicsView,
                             QLabel)

from . import imaging
from .models import is_light_color

HANDLE_PX = 9.0          # tutamaç kenarı (ekran pikseli)
MIN_BOX_PX = 3.0         # resim pikselinde en küçük kutu
LABEL_FONT_PX = 11
LOUPE_SIZE = 168         # büyüteç penceresi kenarı (ekran pikseli)
LOUPE_ZOOM = 4.0


class Mode:
    EDIT = 0
    DRAW = 1


# ============================================================ numpy köprüsü
def qimage_to_array(image: QImage):
    """QImage -> (H, W, 3) uint8 RGB numpy dizisi."""
    if not imaging.HAS_NUMPY:
        return None
    converted = image.convertToFormat(QImage.Format_RGB888)
    width, height = converted.width(), converted.height()
    if width <= 0 or height <= 0:
        return None
    pointer = converted.constBits()
    pointer.setsize(converted.byteCount())
    stride = converted.bytesPerLine()
    flat = imaging.np.frombuffer(pointer, dtype=imaging.np.uint8)
    flat = flat[:height * stride].reshape(height, stride)
    return flat[:, :width * 3].reshape(height, width, 3).copy()


def array_to_qimage(array) -> QImage:
    """(H, W, 3) uint8 RGB numpy dizisi -> QImage (veriyi kopyalar)."""
    contiguous = imaging.np.ascontiguousarray(array, dtype=imaging.np.uint8)
    height, width = contiguous.shape[:2]
    image = QImage(contiguous.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


# ============================================================ kutu öğesi
class BoxItem(QGraphicsRectItem):
    """Taşınabilir, 8 tutamaçtan boyutlandırılabilir, kilitlenebilir kutu."""

    TL, TC, TR, ML, MR, BL, BC, BR = range(8)

    CURSORS = {
        TL: Qt.SizeFDiagCursor, BR: Qt.SizeFDiagCursor,
        TR: Qt.SizeBDiagCursor, BL: Qt.SizeBDiagCursor,
        TC: Qt.SizeVerCursor, BC: Qt.SizeVerCursor,
        ML: Qt.SizeHorCursor, MR: Qt.SizeHorCursor,
    }

    def __init__(self, rect: QRectF, class_id: int, canvas: "Canvas",
                 proposal: bool = False, score: float = 0.0):
        super().__init__()
        self._canvas = canvas
        self.class_id = int(class_id)
        self.locked = False
        self.proposal = bool(proposal)
        self.score = float(score)
        self.setRect(QRectF(rect).normalized())
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(20 if proposal else 10)
        self.view_scale = canvas.current_scale()
        self._drag_mode: Optional[str] = None
        self._drag_handle: Optional[int] = None
        self._press_scene: Optional[QPointF] = None
        self._press_rect: Optional[QRectF] = None
        self._undo_pushed = False

    # ---------------------------------------------------------- görünüm
    def _margin(self) -> float:
        return (HANDLE_PX / 2.0) / max(self.view_scale, 1e-6)

    def set_view_scale(self, scale: float) -> None:
        if abs(scale - self.view_scale) < 1e-9:
            return
        self.prepareGeometryChange()
        self.view_scale = scale
        self.update()

    def boundingRect(self) -> QRectF:
        scale = max(self.view_scale, 1e-6)
        margin = self._margin() + 2.0 / scale
        label_h = (LABEL_FONT_PX + 10) / scale
        label_w = 300.0 / scale          # etiket metnine ayrılan pay
        return self.rect().adjusted(-margin, -margin - label_h, margin + label_w, margin)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        margin = self._margin()
        path.addRect(self.rect().adjusted(-margin, -margin, margin, margin))
        return path

    def color(self) -> str:
        return self._canvas.class_color(self.class_id)

    def label_text(self) -> str:
        name = self._canvas.class_name(self.class_id)
        if self.proposal:
            return f"? {name} {self.score:.2f}"
        if self.locked:
            return f"🔒 {name}"
        return name

    def handle_rects(self):
        rect = self.rect()
        margin = self._margin()
        cx = (rect.left() + rect.right()) / 2.0
        cy = (rect.top() + rect.bottom()) / 2.0
        points = {
            self.TL: (rect.left(), rect.top()),
            self.TC: (cx, rect.top()),
            self.TR: (rect.right(), rect.top()),
            self.ML: (rect.left(), cy),
            self.MR: (rect.right(), cy),
            self.BL: (rect.left(), rect.bottom()),
            self.BC: (cx, rect.bottom()),
            self.BR: (rect.right(), rect.bottom()),
        }
        return {key: QRectF(x - margin, y - margin, 2 * margin, 2 * margin)
                for key, (x, y) in points.items()}

    def handle_at(self, position: QPointF) -> Optional[int]:
        if not self.isSelected() or self.locked:
            return None
        for key, rect in self.handle_rects().items():
            if rect.contains(position):
                return key
        return None

    # ---------------------------------------------------------- çizim
    def paint(self, painter: QPainter, option, widget=None):
        color = QColor(self.color())
        rect = self.rect()
        selected = self.isSelected()

        pen = QPen(color, 3 if selected else 2)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.MiterJoin)
        if self.proposal:
            pen.setStyle(Qt.DashLine)
        elif self.locked:
            pen.setStyle(Qt.DotLine)
        painter.setPen(pen)

        fill = QColor(color)
        if selected:
            fill.setAlpha(70)
            painter.setBrush(QBrush(fill))
        elif self._canvas.show_fill:
            fill.setAlpha(28)
            painter.setBrush(QBrush(fill))
        else:
            painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        if selected and not self.locked:
            handle_pen = QPen(QColor("#202020"), 1)
            handle_pen.setCosmetic(True)
            painter.setPen(handle_pen)
            painter.setBrush(QBrush(QColor("#ffffff")))
            for handle_rect in self.handle_rects().values():
                painter.drawRect(handle_rect)

        self._paint_label(painter, color, rect)

    def _paint_label(self, painter: QPainter, color: QColor, rect: QRectF):
        text = self.label_text()
        if not text:
            return
        painter.save()
        # Etiketi cihaz koordinatında çiz: zoom ne olursa olsun okunur kalır
        top_left = painter.worldTransform().map(QPointF(rect.left(), rect.top()))
        painter.resetTransform()

        font = QFont()
        font.setPixelSize(LABEL_FONT_PX)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_w = metrics.boundingRect(text).width() + 10
        text_h = metrics.height() + 4

        x = top_left.x()
        y = top_left.y() - text_h
        if y < 0:
            y = top_left.y()

        background = QColor(color)
        if self.proposal:
            background.setAlpha(190)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(background))
        painter.drawRect(QRectF(x, y, text_w, text_h))
        painter.setPen(QColor("#000000") if is_light_color(self.color())
                       else QColor("#ffffff"))
        painter.drawText(QRectF(x, y, text_w, text_h), Qt.AlignCenter, text)
        painter.restore()

    # ---------------------------------------------------------- etkileşim
    def hoverMoveEvent(self, event):
        if self._canvas.mode == Mode.DRAW:
            self.unsetCursor()
        elif self.locked:
            self.setCursor(Qt.ForbiddenCursor)
        else:
            handle = self.handle_at(event.pos())
            self.setCursor(self.CURSORS[handle] if handle is not None
                           else Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._canvas.mode == Mode.DRAW:
            event.ignore()
            return
        self._select_only()
        self._canvas.boxDoubleClicked.emit(self)
        event.accept()

    def _select_only(self):
        if self.scene():
            self.scene().clearSelection()
        self.setSelected(True)
        self._canvas.notify_selection()

    def mousePressEvent(self, event):
        # Çizim modunda kutular "şeffaftır": olay tuvale geçsin
        if self._canvas.mode == Mode.DRAW or event.button() != Qt.LeftButton:
            event.ignore()
            return
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl+tık: çoklu seçime ekle/çıkar
            self.setSelected(not self.isSelected())
            self._canvas.notify_selection()
            event.accept()
            return

        handle = self.handle_at(event.pos())          # seçmeden ÖNCE bakılmalı
        if not self.isSelected():
            self._select_only()
        else:
            self._canvas.notify_selection()

        if self.locked:
            event.accept()
            return

        self._drag_handle = handle
        self._drag_mode = "resize" if handle is not None else "move"
        self._press_scene = event.scenePos()
        self._press_rect = QRectF(self.rect())
        self._undo_pushed = False
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_mode is None or self.locked:
            event.ignore()
            return
        if not self._undo_pushed:
            self._canvas.push_undo()
            self._undo_pushed = True

        delta = event.scenePos() - self._press_scene
        rect = QRectF(self._press_rect)
        if self._drag_mode == "move":
            rect.translate(delta)
            rect = self._clamp_moved(rect)
        else:
            rect = self._resized(rect, delta, self._drag_handle)

        self.prepareGeometryChange()
        self.setRect(rect)
        self._canvas.notify_boxes_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None:
            self.prepareGeometryChange()
            self.setRect(self.rect().normalized())
            self._canvas.mark_dirty()
            self._canvas.notify_boxes_changed()
        self._drag_mode = None
        self._drag_handle = None
        self._undo_pushed = False
        event.accept()

    # ---------------------------------------------------------- geometri
    def _clamp_moved(self, rect: QRectF) -> QRectF:
        image = self._canvas.image_rect()
        if image.isEmpty():
            return rect
        dx = dy = 0.0
        if rect.left() < image.left():
            dx = image.left() - rect.left()
        elif rect.right() > image.right():
            dx = image.right() - rect.right()
        if rect.top() < image.top():
            dy = image.top() - rect.top()
        elif rect.bottom() > image.bottom():
            dy = image.bottom() - rect.bottom()
        rect.translate(dx, dy)
        return rect

    def _resized(self, rect: QRectF, delta: QPointF, handle: int) -> QRectF:
        image = self._canvas.image_rect()
        left, top = rect.left(), rect.top()
        right, bottom = rect.right(), rect.bottom()

        if handle in (self.TL, self.ML, self.BL):
            left += delta.x()
        if handle in (self.TR, self.MR, self.BR):
            right += delta.x()
        if handle in (self.TL, self.TC, self.TR):
            top += delta.y()
        if handle in (self.BL, self.BC, self.BR):
            bottom += delta.y()

        if not image.isEmpty():
            left = max(image.left(), min(image.right(), left))
            right = max(image.left(), min(image.right(), right))
            top = max(image.top(), min(image.bottom(), top))
            bottom = max(image.top(), min(image.bottom(), bottom))

        if right - left < MIN_BOX_PX:
            if handle in (self.TL, self.ML, self.BL):
                left = right - MIN_BOX_PX
            else:
                right = left + MIN_BOX_PX
        if bottom - top < MIN_BOX_PX:
            if handle in (self.TL, self.TC, self.TR):
                top = bottom - MIN_BOX_PX
            else:
                bottom = top + MIN_BOX_PX
        return QRectF(QPointF(left, top), QPointF(right, bottom))

    def nudge(self, dx: float, dy: float, resize: bool = False) -> None:
        """Ok tuşlarıyla piksel hassasiyetinde taşıma / boyutlandırma."""
        if self.locked:
            return
        rect = QRectF(self.rect())
        if resize:
            rect.setRight(rect.right() + dx)
            rect.setBottom(rect.bottom() + dy)
            image = self._canvas.image_rect()
            if not image.isEmpty():
                rect.setRight(max(rect.left() + MIN_BOX_PX,
                                  min(image.right(), rect.right())))
                rect.setBottom(max(rect.top() + MIN_BOX_PX,
                                   min(image.bottom(), rect.bottom())))
        else:
            rect.translate(dx, dy)
            rect = self._clamp_moved(rect)
        self.prepareGeometryChange()
        self.setRect(rect.normalized())


# ============================================================ tuval
class Canvas(QGraphicsView):
    """Resmi ve kutuları gösteren; zoom, pan, çizim ve görüntü iyileştirme yapan görünüm."""

    boxesChanged = pyqtSignal()
    selectionChanged = pyqtSignal()
    boxCreated = pyqtSignal(object)
    boxDoubleClicked = pyqtSignal(object)
    contextMenuRequested = pyqtSignal(object, object)
    cursorMoved = pyqtSignal(float, float)
    zoomChanged = pyqtSignal(float)
    dirtyChanged = pyqtSignal(bool)
    modeChanged = pyqtSignal(int)

    MIN_SCALE = 0.02
    MAX_SCALE = 40.0
    UNDO_LIMIT = 80

    def __init__(self, class_manager, parent=None):
        super().__init__(parent)
        self.classes = class_manager

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QColor("#1e1f22"))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # ---- durum
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._boxes: List[BoxItem] = []
        self._img_w = 0
        self._img_h = 0
        self._base_pixmap: Optional[QPixmap] = None
        self._source_array = None            # numpy RGB (varsa)
        self._display_pixmap: Optional[QPixmap] = None

        self.mode = Mode.EDIT
        self.current_class = 0
        self.show_fill = False
        self.magnetic = False
        self.loupe_enabled = False
        self._boxes_visible = True
        self._hidden_classes: Set[int] = set()

        self._brightness = 0
        self._contrast = 0
        self._gamma = 1.0
        self._clahe = False
        self._edges = False

        self._fit = True
        self._dirty = False
        self._panning = False
        self._pan_last: Optional[QPoint] = None
        self._drawing = False
        self._draw_origin: Optional[QPointF] = None
        self._preview: Optional[QGraphicsRectItem] = None
        self._rubber = False
        self._rubber_origin: Optional[QPointF] = None
        self._rubber_item: Optional[QGraphicsRectItem] = None
        self._cursor_scene: Optional[QPointF] = None
        self._undo_stack: List[list] = []

        self._hint = QLabel(
            "Başlamak için  Dosya → Klasör Aç  (Ctrl+O)\n\n"
            "Sonra:  E  ile sınıf ekleyin  ·  W  ile çizim moduna geçin\n"
            "D / A  ile resimler arasında gezinin  ·  F1  tüm kısayollar", self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #9aa0a6; font-size: 14px; background: transparent;")
        self._hint.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    # ---------------------------------------------------------- resim
    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def image_size(self):
        return self._img_w, self._img_h

    def image_rect(self) -> QRectF:
        if self._pixmap_item is None:
            return QRectF()
        return QRectF(0, 0, self._img_w, self._img_h)

    def source_array(self):
        """Görüntü işleme için orijinal RGB dizisi (numpy yoksa None)."""
        return self._source_array

    def load_image(self, path: str) -> bool:
        pixmap = QPixmap(path)
        if pixmap.isNull() or pixmap.width() == 0 or pixmap.height() == 0:
            return False
        self.clear_all()
        self._img_w, self._img_h = pixmap.width(), pixmap.height()
        self._base_pixmap = pixmap
        self._source_array = (qimage_to_array(pixmap.toImage())
                              if imaging.HAS_NUMPY else None)

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(0, 0, self._img_w, self._img_h))

        self._apply_display_pipeline()
        self._fit = True
        self.fit_to_window()
        self.set_dirty(False)
        self._layout_hint()
        return True

    def clear_all(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._preview = None
        self._rubber_item = None
        self._boxes = []
        self._undo_stack = []
        self._img_w = self._img_h = 0
        self._base_pixmap = None
        self._source_array = None
        self._display_pixmap = None
        self._scene.setSceneRect(QRectF())
        self._layout_hint()

    # ---------------------------------------------------------- görüntü işleme
    def enhancement(self) -> dict:
        return {"brightness": self._brightness, "contrast": self._contrast,
                "gamma": self._gamma, "clahe": self._clahe, "edges": self._edges}

    def set_enhancement(self, brightness: int = 0, contrast: int = 0,
                        gamma: float = 1.0, clahe: bool = False) -> None:
        self._brightness = int(brightness)
        self._contrast = int(contrast)
        self._gamma = float(gamma)
        self._clahe = bool(clahe)
        self._apply_display_pipeline()

    def set_edge_overlay(self, enabled: bool) -> None:
        self._edges = bool(enabled)
        self._apply_display_pipeline()

    def has_enhancement(self) -> bool:
        return not imaging.is_identity(self._brightness, self._contrast,
                                       self._gamma, self._clahe) or self._edges

    def _apply_display_pipeline(self) -> None:
        """
        Görüntü iyileştirme yalnızca EKRANI etkiler; dosya ve etiket
        koordinatları hiçbir koşulda değişmez.
        """
        if self._pixmap_item is None or self._base_pixmap is None:
            return
        if not imaging.HAS_NUMPY or self._source_array is None or not self.has_enhancement():
            self._display_pixmap = self._base_pixmap
            self._pixmap_item.setPixmap(self._base_pixmap)
            return

        array = imaging.enhance(self._source_array, self._brightness,
                                self._contrast, self._gamma, self._clahe)
        if self._edges:
            edges = imaging.edge_map(self._source_array)
            mask = edges > 0
            array = array.copy()
            array[mask] = [0, 255, 120]
        pixmap = QPixmap.fromImage(array_to_qimage(array))
        self._display_pixmap = pixmap
        self._pixmap_item.setPixmap(pixmap)

    # ---------------------------------------------------------- kutular
    def boxes(self) -> List[BoxItem]:
        return list(self._boxes)

    def real_boxes(self) -> List[BoxItem]:
        """Öneri olmayan, kaydedilecek kutular."""
        return [b for b in self._boxes if not b.proposal]

    def proposals(self) -> List[BoxItem]:
        return [b for b in self._boxes if b.proposal]

    def box_count(self) -> int:
        return len(self._boxes)

    def add_box(self, rect: QRectF, class_id: int, select: bool = False,
                proposal: bool = False, score: float = 0.0) -> BoxItem:
        item = BoxItem(rect, class_id, self, proposal=proposal, score=score)
        self._scene.addItem(item)
        self._boxes.append(item)
        item.set_view_scale(self.current_scale())
        self._apply_visibility(item)
        if select:
            self._scene.clearSelection()
            item.setSelected(True)
            self.notify_selection()
        self.notify_boxes_changed()
        return item

    def remove_box(self, item: BoxItem) -> None:
        if item in self._boxes:
            self._boxes.remove(item)
        self._scene.removeItem(item)
        self.notify_boxes_changed()
        self.notify_selection()

    def delete_selected(self) -> int:
        selected = [b for b in self.selected_boxes() if not b.locked]
        if not selected:
            return 0
        self.push_undo()
        for item in selected:
            self.remove_box(item)
        self.mark_dirty()
        return len(selected)

    def clear_boxes(self) -> bool:
        if not self._boxes:
            return False
        self.push_undo()
        for item in list(self._boxes):
            self.remove_box(item)
        self.mark_dirty()
        return True

    def selected_boxes(self) -> List[BoxItem]:
        return [b for b in self._boxes if b.isSelected()]

    def selected_box(self) -> Optional[BoxItem]:
        selection = self.selected_boxes()
        return selection[0] if selection else None

    def is_box_hidden(self, item: BoxItem) -> bool:
        """Görünürlük kararını pencere durumundan bağımsız verir."""
        return (not self._boxes_visible) or item.class_id in self._hidden_classes

    def select_all(self) -> int:
        visible = [b for b in self._boxes if not self.is_box_hidden(b)]
        for item in visible:
            item.setSelected(True)
        self.notify_selection()
        return len(visible)

    def clear_selection(self) -> None:
        self._scene.clearSelection()
        self.notify_selection()

    def set_boxes(self, pixel_boxes: Sequence) -> None:
        """(class_id, x1, y1, x2, y2) demetlerinden kutuları kurar."""
        for item in list(self._boxes):
            self._scene.removeItem(item)
        self._boxes = []
        for row in pixel_boxes:
            class_id, x1, y1, x2, y2 = row[:5]
            self.add_box(QRectF(QPointF(x1, y1), QPointF(x2, y2)), class_id)
        self._scene.clearSelection()
        self.notify_boxes_changed()
        self.notify_selection()

    def assign_class_to_selection(self, class_id: int) -> int:
        selected = [b for b in self.selected_boxes() if not b.locked]
        if not selected:
            return 0
        self.push_undo()
        for item in selected:
            item.class_id = int(class_id)
            self._apply_visibility(item)
            item.update()
        self.mark_dirty()
        self.notify_boxes_changed()
        return len(selected)

    def reassign_class(self, old_id: int, new_id: int) -> int:
        """Bir sınıfın tüm kutularını başka sınıfa taşır."""
        moved = 0
        for item in self._boxes:
            if item.class_id == old_id:
                item.class_id = new_id
                item.update()
                moved += 1
        if moved:
            self.notify_boxes_changed()
        return moved

    def apply_class_mapping(self, mapping) -> int:
        """Sınıf silindiğinde açık resimdeki kutuları günceller."""
        removed = 0
        for item in list(self._boxes):
            new_id = mapping.get(item.class_id, item.class_id)
            if new_id is None:
                self.remove_box(item)
                removed += 1
            elif new_id != item.class_id:
                item.class_id = new_id
                item.update()
        self.notify_boxes_changed()
        return removed

    def orphan_boxes(self) -> List[BoxItem]:
        """Sınıfı sınıf listesinde bulunmayan kutular (olmaması gereken durum)."""
        limit = len(self.classes)
        return [b for b in self._boxes if not (0 <= b.class_id < limit)]

    def toggle_lock_selection(self) -> int:
        selected = self.selected_boxes()
        if not selected:
            return 0
        target = not all(b.locked for b in selected)
        for item in selected:
            item.locked = target
            item.update()
        self.notify_boxes_changed()
        return len(selected)

    def nudge_selection(self, dx: float, dy: float, resize: bool = False) -> int:
        movable = [b for b in self.selected_boxes() if not b.locked]
        if not movable:
            return 0
        self.push_undo()
        for item in movable:
            item.nudge(dx, dy, resize)
        self.mark_dirty()
        self.notify_boxes_changed()
        return len(movable)

    # ---------------------------------------------------------- öneriler
    def add_proposals(self, detections, class_mapping=None) -> int:
        """(class_id, x1, y1, x2, y2, score) listesini öneri kutusu olarak ekler."""
        added = 0
        for class_id, x1, y1, x2, y2, score in detections:
            target = class_id if class_mapping is None else class_mapping.get(class_id)
            if target is None:
                continue
            self.add_box(QRectF(QPointF(x1, y1), QPointF(x2, y2)), target,
                         proposal=True, score=score)
            added += 1
        return added

    def accept_proposals(self) -> int:
        pending = self.proposals()
        if not pending:
            return 0
        self.push_undo()
        for item in pending:
            item.proposal = False
            item.setZValue(10)
            item.update()
        self.mark_dirty()
        self.notify_boxes_changed()
        return len(pending)

    def discard_proposals(self) -> int:
        pending = self.proposals()
        for item in pending:
            self.remove_box(item)
        return len(pending)

    # ---------------------------------------------------------- görünürlük
    def set_boxes_visible(self, visible: bool) -> None:
        self._boxes_visible = bool(visible)
        for item in self._boxes:
            self._apply_visibility(item)

    def boxes_visible(self) -> bool:
        return self._boxes_visible

    def set_hidden_classes(self, hidden: Set[int]) -> None:
        self._hidden_classes = set(hidden or ())
        for item in self._boxes:
            self._apply_visibility(item)

    def hidden_classes(self) -> Set[int]:
        return set(self._hidden_classes)

    def _apply_visibility(self, item: BoxItem) -> None:
        visible = self._boxes_visible and item.class_id not in self._hidden_classes
        item.setVisible(visible)
        if not visible and item.isSelected():
            item.setSelected(False)

    # ---------------------------------------------------------- geri alma
    def snapshot(self) -> list:
        return [(b.class_id, b.rect().left(), b.rect().top(),
                 b.rect().right(), b.rect().bottom(), b.locked, b.proposal, b.score)
                for b in self._boxes]

    def restore(self, state: Sequence) -> None:
        for item in list(self._boxes):
            self._scene.removeItem(item)
        self._boxes = []
        for row in state:
            class_id, x1, y1, x2, y2 = row[:5]
            locked = row[5] if len(row) > 5 else False
            proposal = row[6] if len(row) > 6 else False
            score = row[7] if len(row) > 7 else 0.0
            item = self.add_box(QRectF(QPointF(x1, y1), QPointF(x2, y2)),
                                class_id, proposal=proposal, score=score)
            item.locked = bool(locked)
        self._scene.clearSelection()
        self.notify_boxes_changed()
        self.notify_selection()

    def push_undo(self) -> None:
        if self._pixmap_item is None:
            return
        self._undo_stack.append(self.snapshot())
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)

    def pop_undo(self) -> None:
        if self._undo_stack:
            self._undo_stack.pop()

    def clear_undo(self) -> None:
        """
        Sınıf kimlikleri değiştikten sonra eski anlık görüntüler geçersizdir:
        geri alınırlarsa sahipsiz kutu üretirler. Bu yüzden yığın boşaltılır.
        """
        self._undo_stack = []

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self.restore(self._undo_stack.pop())
        self.mark_dirty()
        return True

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    # ---------------------------------------------------------- durum
    def is_dirty(self) -> bool:
        return self._dirty

    def set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.dirtyChanged.emit(value)

    def mark_dirty(self) -> None:
        self.set_dirty(True)

    def notify_boxes_changed(self) -> None:
        self.boxesChanged.emit()

    def notify_selection(self) -> None:
        self.selectionChanged.emit()

    def set_mode(self, mode: int) -> None:
        if self.mode == mode:
            return
        self.mode = mode
        if mode == Mode.DRAW:
            self.viewport().setCursor(Qt.CrossCursor)
            self._scene.clearSelection()
            self.notify_selection()
        else:
            self.viewport().setCursor(Qt.ArrowCursor)
        self.viewport().update()
        self.modeChanged.emit(mode)

    def toggle_mode(self) -> None:
        self.set_mode(Mode.EDIT if self.mode == Mode.DRAW else Mode.DRAW)

    # ---------------------------------------------------------- sınıf bilgisi
    def class_color(self, class_id: int) -> str:
        return self.classes.color_of(class_id)

    def class_name(self, class_id: int) -> str:
        return self.classes.name_of(class_id)

    def refresh_boxes(self) -> None:
        for item in self._boxes:
            self._apply_visibility(item)
            item.update()

    def set_show_fill(self, value: bool) -> None:
        self.show_fill = bool(value)
        self.refresh_boxes()

    # ---------------------------------------------------------- zoom / pan
    def current_scale(self) -> float:
        return float(self.transform().m11())

    def _sync_item_scales(self) -> None:
        scale = self.current_scale()
        for item in self._boxes:
            item.set_view_scale(scale)

    def zoom_by(self, factor: float, under_mouse: bool = False) -> None:
        if self._pixmap_item is None:
            return
        current = self.current_scale()
        target = max(self.MIN_SCALE, min(self.MAX_SCALE, current * factor))
        factor = target / current
        if abs(factor - 1.0) < 1e-9:
            return
        anchor = (QGraphicsView.AnchorUnderMouse if under_mouse
                  else QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(anchor)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._fit = False
        self._sync_item_scales()
        self.zoomChanged.emit(self.current_scale())

    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._fit = True
        self._sync_item_scales()
        self.zoomChanged.emit(self.current_scale())

    def zoom_reset_100(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.centerOn(self._pixmap_item)
        self._fit = False
        self._sync_item_scales()
        self.zoomChanged.emit(self.current_scale())

    def zoom_to_box(self, item: BoxItem, margin: float = 40.0) -> None:
        if self._pixmap_item is None or item is None:
            return
        self.fitInView(item.rect().adjusted(-margin, -margin, margin, margin),
                       Qt.KeepAspectRatio)
        self._fit = False
        self._sync_item_scales()
        self.zoomChanged.emit(self.current_scale())

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(1.25 if delta > 0 else 1 / 1.25, under_mouse=True)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_hint()
        if self._fit:
            self.fit_to_window()

    def _layout_hint(self):
        self._hint.setGeometry(0, 0, self.width(), self.height())
        self._hint.setVisible(self._pixmap_item is None)

    # ---------------------------------------------------------- fare
    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self.image_rect()
        if rect.isEmpty():
            return point
        return QPointF(max(rect.left(), min(rect.right(), point.x())),
                       max(rect.top(), min(rect.bottom(), point.y())))

    def _start_pan(self, event) -> None:
        self._panning = True
        self._pan_last = event.pos()
        self.viewport().setCursor(Qt.ClosedHandCursor)

    def _end_pan(self) -> None:
        self._panning = False
        self._pan_last = None
        self.viewport().setCursor(
            Qt.CrossCursor if self.mode == Mode.DRAW else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            return

        if event.button() == Qt.MiddleButton or \
                (event.button() == Qt.LeftButton
                 and event.modifiers() & Qt.AltModifier):
            self._start_pan(event)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.mode == Mode.DRAW:
            self._drawing = True
            self._draw_origin = self._clamp_to_image(self.mapToScene(event.pos()))
            self._preview = QGraphicsRectItem(QRectF(self._draw_origin, self._draw_origin))
            pen = QPen(QColor(self.class_color(self.current_class)), 2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            self._preview.setPen(pen)
            self._preview.setZValue(50)
            self._scene.addItem(self._preview)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.mode == Mode.EDIT:
            item = self.itemAt(event.pos())
            if not isinstance(item, BoxItem):
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift + boşlukta sürükle: lastik kutuyla çoklu seçim
                    self._rubber = True
                    self._rubber_origin = self.mapToScene(event.pos())
                    self._rubber_item = QGraphicsRectItem(
                        QRectF(self._rubber_origin, self._rubber_origin))
                    pen = QPen(QColor("#3b82f6"), 1)
                    pen.setCosmetic(True)
                    pen.setStyle(Qt.DashLine)
                    self._rubber_item.setPen(pen)
                    self._rubber_item.setZValue(60)
                    self._scene.addItem(self._rubber_item)
                    event.accept()
                    return
                self._scene.clearSelection()
                self.notify_selection()
                self._start_pan(event)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pixmap_item is not None:
            self._cursor_scene = self.mapToScene(event.pos())
            self.cursorMoved.emit(self._cursor_scene.x(), self._cursor_scene.y())

        if self._panning and self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            event.accept()
            return

        if self._rubber and self._rubber_item is not None:
            self._rubber_item.setRect(
                QRectF(self._rubber_origin, self.mapToScene(event.pos())).normalized())
            event.accept()
            return

        if self._drawing and self._preview is not None:
            point = self._clamp_to_image(self.mapToScene(event.pos()))
            self._preview.setRect(QRectF(self._draw_origin, point).normalized())
            self.viewport().update()
            event.accept()
            return

        super().mouseMoveEvent(event)
        if self.mode == Mode.DRAW or self.loupe_enabled:
            self.viewport().update()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._end_pan()
            event.accept()
            return

        if self._rubber:
            self._rubber = False
            rect = QRectF()
            if self._rubber_item is not None:
                rect = self._rubber_item.rect().normalized()
                self._scene.removeItem(self._rubber_item)
                self._rubber_item = None
            if rect.width() > 2 and rect.height() > 2:
                count = 0
                for item in self._boxes:
                    if not self.is_box_hidden(item) and rect.intersects(item.rect()):
                        item.setSelected(True)
                        count += 1
                if count:
                    self.notify_selection()
            event.accept()
            return

        if self._drawing:
            self._drawing = False
            rect = QRectF()
            if self._preview is not None:
                rect = self._preview.rect().normalized()
                self._scene.removeItem(self._preview)
                self._preview = None
            if rect.width() >= MIN_BOX_PX and rect.height() >= MIN_BOX_PX:
                if self.magnetic:
                    rect = self._snap_rect(rect)
                self.push_undo()
                item = self.add_box(rect, self.current_class, select=True)
                self.mark_dirty()
                self.boxCreated.emit(item)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _snap_rect(self, rect: QRectF) -> QRectF:
        """Manyetik kutu: kaba dikdörtgeni en güçlü kenarlara oturt."""
        if self._source_array is None or not imaging.HAS_NUMPY:
            return rect
        snapped = imaging.snap_box(self._source_array, rect.left(), rect.top(),
                                   rect.right(), rect.bottom())
        if snapped is None:
            return rect
        x1, y1, x2, y2 = snapped
        return QRectF(QPointF(x1, y1), QPointF(x2, y2))

    def snap_selection(self) -> int:
        """Seçili kutuları kenarlara oturtur (S tuşu)."""
        targets = [b for b in self.selected_boxes() if not b.locked]
        if not targets or self._source_array is None:
            return 0
        self.push_undo()
        changed = 0
        for item in targets:
            new_rect = self._snap_rect(item.rect())
            if new_rect != item.rect():
                item.prepareGeometryChange()
                item.setRect(new_rect)
                changed += 1
        if changed:
            self.mark_dirty()
            self.notify_boxes_changed()
        else:
            self.pop_undo()
        return changed

    def cancel_drawing(self) -> bool:
        if not self._drawing:
            return False
        self._drawing = False
        if self._preview is not None:
            self._scene.removeItem(self._preview)
            self._preview = None
        self.viewport().update()
        return True

    def contextMenuEvent(self, event):
        if self._pixmap_item is None:
            return
        item = self.itemAt(event.pos())
        box = item if isinstance(item, BoxItem) else None
        if box is not None and not box.isSelected():
            self._scene.clearSelection()
            box.setSelected(True)
            self.notify_selection()
        self.contextMenuRequested.emit(event.globalPos(), box)
        event.accept()

    def leaveEvent(self, event):
        self._cursor_scene = None
        self.viewport().update()
        super().leaveEvent(event)

    # ---------------------------------------------------------- klavye
    def keyPressEvent(self, event):
        step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
        resize = bool(event.modifiers() & Qt.AltModifier)
        deltas = {
            Qt.Key_Left: (-step, 0.0),
            Qt.Key_Right: (step, 0.0),
            Qt.Key_Up: (0.0, -step),
            Qt.Key_Down: (0.0, step),
        }
        if event.key() in deltas and self.selected_boxes():
            dx, dy = deltas[event.key()]
            self.nudge_selection(dx, dy, resize)
            event.accept()
            return
        super().keyPressEvent(event)

    # ---------------------------------------------------------- üst katman
    def drawForeground(self, painter: QPainter, rect: QRectF):
        if self._pixmap_item is None:
            return
        if self.mode == Mode.DRAW and self._cursor_scene is not None:
            pen = QPen(QColor(0, 220, 120, 170), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            x, y = self._cursor_scene.x(), self._cursor_scene.y()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        if self.loupe_enabled and self._cursor_scene is not None:
            self._draw_loupe(painter)

    def _draw_loupe(self, painter: QPainter) -> None:
        """
        Sağ üst köşede, imlecin altındaki bölgeyi 4× gösteren büyüteç.
        Kutu kenarını piksel hassasiyetinde oturtmanın en hızlı yolu.
        """
        source = self._display_pixmap or self._base_pixmap
        if source is None:
            return
        span = max(8.0, LOUPE_SIZE / LOUPE_ZOOM)
        cx, cy = self._cursor_scene.x(), self._cursor_scene.y()
        source_rect = QRectF(cx - span / 2.0, cy - span / 2.0, span, span)

        painter.save()
        painter.resetTransform()
        margin = 12
        target = QRectF(self.viewport().width() - LOUPE_SIZE - margin, margin,
                        LOUPE_SIZE, LOUPE_SIZE)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.fillRect(target, QColor(20, 22, 28))
        painter.drawPixmap(target, source, source_rect)

        pen = QPen(QColor("#3b82f6"), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target)

        center_pen = QPen(QColor(255, 60, 60, 200), 1)
        painter.setPen(center_pen)
        center_x = target.left() + target.width() / 2.0
        center_y = target.top() + target.height() / 2.0
        painter.drawLine(QPointF(target.left(), center_y), QPointF(target.right(), center_y))
        painter.drawLine(QPointF(center_x, target.top()), QPointF(center_x, target.bottom()))
        painter.restore()
