"""Clickable QLabel for video preview with ROI point selection overlay."""

from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap, QColor, QPolygonF
from PySide6.QtWidgets import QLabel
import numpy as np


class ClickableLabel(QLabel):
    """QLabel that accepts mouse clicks and draws ROI polygon overlay.

    Signals:
        roi_changed: emitted when ROI points are updated, carries list of (x, y) tuples
            in original image coordinates.
    """

    roi_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(400)
        self._scaled_size: tuple[int, int] = (1, 1)
        self.setStyleSheet(
            "border: 1px solid #3a3a3a; background: #1a1a1a; color: #888;"
        )
        self.setText("暂无图像")
        self._roi_points: list[tuple[float, float]] = []
        self._selection_active = False
        self._roi_method: str = "four_point"
        self._original_size: tuple[int, int] = (1, 1)
        self._numpy_image: np.ndarray | None = None

    def set_numpy_image(self, img: np.ndarray | None):
        """Display a numpy RGB image, scaled to 400px height."""
        self._numpy_image = img
        if img is None:
            self.clear()
            self.setText("暂无图像")
            return
        self.setText("")
        h, w, _ = img.shape
        self._original_size = (w, h)
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaledToHeight(self.minimumHeight(), Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self._scaled_size = (scaled.width(), scaled.height())

    def roi_points(self) -> list[tuple[float, float]]:
        return list(self._roi_points)

    def set_roi_points(self, points: list[tuple[float, float]]):
        self._roi_points = list(points)
        self.update()

    def set_selection_active(self, active: bool):
        self._selection_active = active

    def selection_active(self) -> bool:
        return self._selection_active

    def set_roi_method(self, method: str):
        self._roi_method = method

    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        pixmap = self.pixmap()
        if pixmap is None:
            super().mousePressEvent(event)
            return

        # Map label coords → display coords → original image coords
        label_w, label_h = self.width(), self.height()
        disp_w, disp_h = self._scaled_size
        img_w, img_h = self._original_size

        # Letterbox offset from centered display
        offset_x = (label_w - disp_w) / 2
        offset_y = (label_h - disp_h) / 2

        click_x = event.position().x() - offset_x
        click_y = event.position().y() - offset_y

        if click_x < 0 or click_y < 0 or click_x > disp_w or click_y > disp_h:
            super().mousePressEvent(event)
            return

        img_x = click_x / disp_w * img_w
        img_y = click_y / disp_h * img_h

        from handlers.preview import handle_image_click

        frame_rgb = self._numpy_image
        if frame_rgb is None:
            frame_rgb = self._numpy_image

        result = handle_image_click(
            frame_rgb, self._roi_points, self._selection_active,
            float(img_x), float(img_y), method=self._roi_method,
        )
        out_frame, coord_text, new_points, new_active = result

        self._roi_points = new_points
        self._selection_active = new_active
        self.set_numpy_image(out_frame)
        self.roi_changed.emit(list(new_points))
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pixmap = self.pixmap()
        if pixmap is None or not self._roi_points:
            painter.end()
            return

        # Compute letterbox offset + scale from display pixmap
        label_w, label_h = self.width(), self.height()
        disp_w, disp_h = self._scaled_size
        img_w, img_h = self._original_size
        offset_x = (label_w - disp_w) / 2
        offset_y = (label_h - disp_h) / 2

        label_points = []
        for px, py in self._roi_points:
            lx = offset_x + (px / img_w) * disp_w
            ly = offset_y + (py / img_h) * disp_h
            label_points.append(QPointF(lx, ly))

        # Draw polygon edges
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for i in range(len(label_points)):
            p1 = label_points[i]
            p2 = label_points[(i + 1) % len(label_points)]
            painter.drawLine(p1, p2)

        # Draw vertex points
        for pt in label_points:
            painter.setBrush(QColor(255, 0, 0))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(pt, 4, 4)

        painter.end()
