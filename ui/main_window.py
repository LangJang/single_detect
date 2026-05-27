"""Qt main window for the YOLO detection system."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import shared_state
from scene_manager import (
    delete_scene,
    load_scene,
    on_model_change,
    refresh_scene_list,
    save_scene,
)
from handlers.batch import scan_directory, stop_batch
from handlers.monitor import _monitor_log_text, stop_monitor
from handlers.preview import (
    autofill_calib_from_roi,
    clear_roi_selection,
    load_preview_video,
    seek_preview_frame,
    start_roi_selection,
    sync_manual_roi,
)
from handlers.single import scan_video_dir
from ui.image_label import ClickableLabel
from ui.workers import BatchWorker, MonitorWorker, SingleVideoWorker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def numpy_to_pixmap(img: np.ndarray | None, target_w: int = 0, target_h: int = 0) -> QPixmap | None:
    """Convert numpy RGB image to QPixmap, optionally scaled to fit target size."""
    if img is None:
        return None
    h, w, _ = img.shape
    qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_w > 0 and target_h > 0:
        pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def _slider_spin_row(label, min_v, max_v, default, step, parent, scale_slider=3, scale_spin=1):
    """Return (slider, spinbox, row_widget)."""
    row = QWidget(parent)
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label, row)
    sld = QSlider(Qt.Horizontal, row)
    sld.setRange(0, int((max_v - min_v) / step))
    sld.setValue(int((default - min_v) / step))
    spn = QDoubleSpinBox(row)
    spn.setRange(min_v, max_v)
    spn.setSingleStep(step)
    spn.setDecimals(len(str(step).split(".")[-1]) if "." in str(step) else 0)
    spn.setValue(default)
    def _sync_slider_to_spin(v):
        spn.blockSignals(True)
        spn.setValue(min_v + v * step)
        spn.blockSignals(False)

    def _sync_spin_to_slider(v):
        sld.blockSignals(True)
        sld.setValue(int((v - min_v) / step))
        sld.blockSignals(False)

    sld.valueChanged.connect(_sync_slider_to_spin)
    spn.valueChanged.connect(_sync_spin_to_slider)
    lay.addWidget(lbl, 1)
    lay.addWidget(sld, scale_slider)
    lay.addWidget(spn, scale_spin)
    return sld, spn, row


class CollapsibleSection(QWidget):
    """A section with a clickable header that toggles content visibility.

    Content is hidden when collapsed, preventing accidental parameter changes.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = False

        self._header = QPushButton(f"▼  {title}")
        self._header.setFlat(True)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  padding: 5px 12px;"
            "  font-weight: bold;"
            "  font-size: 13px;"
            "  border: 1px solid #2a2a2a;"
            "  border-bottom: 2px solid #1a1a1a;"
            "  border-radius: 6px;"
            "  background: qlineargradient(x1:0,y1:0, x2:0,y2:1,"
            "    stop:0 #3d3d3d, stop:1 #282828);"
            "  color: #ddd;"
            "}"
            "QPushButton:hover {"
            "  border-color: #4a4a4a;"
            "  border-bottom-color: #3a3a3a;"
            "  background: qlineargradient(x1:0,y1:0, x2:0,y2:1,"
            "    stop:0 #4a4a4a, stop:1 #333333);"
            "}"
            "QPushButton:pressed {"
            "  border-bottom: 1px solid #2a2a2a;"
            "  background: qlineargradient(x1:0,y1:0, x2:0,y2:1,"
            "    stop:0 #282828, stop:1 #3a3a3a);"
            "}"
        )
        self._header.clicked.connect(self._toggle)

        self._content = QWidget()

        self._group = QGroupBox()
        group_lay = QVBoxLayout(self._group)
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.addWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._group)

    def content(self) -> QWidget:
        return self._content

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        arrow = "▶" if self._collapsed else "▼"
        self._header.setText(f"{arrow}  {self._title}")


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 目标检测系统")
        self.resize(1200, 900)

        # --- internal state (was gr.State) ---
        self.roi_points_state: list = []
        self.roi_selection_active: bool = False
        self.preview_video_meta: dict = {}
        self._monitor_worker: MonitorWorker | None = None
        self._single_worker: SingleVideoWorker | None = None
        self._batch_worker: BatchWorker | None = None

        # --- central widget ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        central = QWidget()
        scroll.setWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(12, 12, 12, 12)
        self._main_layout.setSpacing(8)

        # --- title ---
        title_lbl = QLabel('<h1>YOLO 目标检测系统</h1>')
        title_lbl.setAlignment(Qt.AlignCenter)
        self._main_layout.addWidget(title_lbl)

        # --- build sections ---
        self._build_scene_section()
        self._build_detection_section()
        self._build_roi_section()
        self._build_alert_section()
        self._build_tabs()

        # --- status bar with progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximum(100)
        self._status_label = QLabel("就绪")
        status = QStatusBar()
        status.addWidget(self._status_label, 1)
        status.addPermanentWidget(self._progress_bar)
        self.setStatusBar(status)

        # --- wire signals ---
        self._connect_signals()

        # --- initial data load ---
        self._refresh_scenes()

    # ==================================================================
    # Scene Management
    # ==================================================================

    def _build_scene_section(self):
        grp = QGroupBox("场景 (Tag) 管理")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.setEditable(False)
        self.scene_name_edit = QLineEdit()
        self.scene_name_edit.setPlaceholderText("输入名称以保存...")
        self.scene_load_btn = QPushButton("加载")
        self.scene_save_btn = QPushButton("保存")
        row1.addWidget(self.scene_combo, 3)
        row1.addWidget(self.scene_name_edit, 2)
        row1.addWidget(self.scene_load_btn, 1)
        row1.addWidget(self.scene_save_btn, 1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.scene_status_lbl = QLabel("")
        self.scene_refresh_btn = QPushButton("刷新列表")
        self.scene_delete_btn = QPushButton("删除")
        row2.addWidget(self.scene_status_lbl, 3)
        row2.addWidget(self.scene_refresh_btn, 1)
        row2.addWidget(self.scene_delete_btn, 1)
        lay.addLayout(row2)

        self._main_layout.addWidget(grp)

    # ==================================================================
    # Detection Parameters
    # ==================================================================

    def _build_detection_section(self):
        section = CollapsibleSection("检测参数")
        lay = QVBoxLayout(section.content())

        row1 = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(shared_state.MODEL_CHOICES)
        self.model_combo.setCurrentText(shared_state.DEFAULT_MODEL)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.model_status_lbl = QLabel(f"{shared_state.DEFAULT_MODEL} 已就绪")
        row1.addWidget(QLabel("检测模型"), 1)
        row1.addWidget(self.model_combo, 3)
        row1.addWidget(QLabel("计算设备"), 1)
        row1.addWidget(self.device_combo, 1)
        row1.addWidget(self.model_status_lbl, 2)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        _, _, self.conf_row = _slider_spin_row("置信度阈值", 0.1, 0.9, 0.25, 0.05, section.content())
        self.conf_slider = self.conf_row.layout().itemAt(1).widget()
        self.conf_spin = self.conf_row.layout().itemAt(2).widget()
        _, _, self.nms_row = _slider_spin_row("NMS IoU", 0.1, 0.9, 0.3, 0.05, section.content())
        self.nms_slider = self.nms_row.layout().itemAt(1).widget()
        self.nms_spin = self.nms_row.layout().itemAt(2).widget()
        # frame_skip uses QSpinBox (int) because range() requires int
        self.fskip_row = QWidget(section.content())
        fskip_lay = QHBoxLayout(self.fskip_row)
        fskip_lay.setContentsMargins(0, 0, 0, 0)
        fskip_lay.addWidget(QLabel("帧采样间隔"), 1)
        self.fskip_slider = QSlider(Qt.Horizontal)
        self.fskip_slider.setRange(1, 60)
        self.fskip_slider.setValue(10)
        self.fskip_spin = QSpinBox()
        self.fskip_spin.setRange(1, 60)
        self.fskip_spin.setValue(10)
        self.fskip_slider.valueChanged.connect(self.fskip_spin.setValue)
        self.fskip_spin.valueChanged.connect(self.fskip_slider.setValue)
        fskip_lay.addWidget(self.fskip_slider, 3)
        fskip_lay.addWidget(self.fskip_spin, 1)
        row2.addWidget(self.conf_row)
        row2.addWidget(self.nms_row)
        row2.addWidget(self.fskip_row)
        lay.addLayout(row2)

        self._main_layout.addWidget(section)

    # ==================================================================
    # ROI & Calibration
    # ==================================================================

    def _build_roi_section(self):
        section = CollapsibleSection("ROI 区域 & 相机标定")
        lay = QVBoxLayout(section.content())

        # Video path row
        row_path = QHBoxLayout()
        self.preview_path_edit = QLineEdit()
        self.preview_path_edit.setPlaceholderText("输入视频文件路径...")
        self.load_preview_btn = QPushButton("加载预览")
        row_path.addWidget(QLabel("预览视频"), 1)
        row_path.addWidget(self.preview_path_edit, 4)
        row_path.addWidget(self.load_preview_btn, 1)
        lay.addLayout(row_path)

        # Frame slider row
        row_slider = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 100)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_info_lbl = QLabel("请先加载预览视频")
        row_slider.addWidget(QLabel("跳转到帧"), 1)
        row_slider.addWidget(self.frame_slider, 8)
        row_slider.addWidget(self.frame_info_lbl, 3)
        lay.addLayout(row_slider)

        # Image + ROI controls
        row2 = QHBoxLayout()
        self.preview_label = ClickableLabel()
        self.preview_label.setMinimumHeight(400)
        row2.addWidget(self.preview_label, 3)

        # ROI controls column
        roi_col = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.roi_start_btn = QPushButton("开始选择区域")
        self.roi_clear_btn = QPushButton("清除区域")
        btn_row.addWidget(self.roi_start_btn)
        btn_row.addWidget(self.roi_clear_btn)
        roi_col.addLayout(btn_row)

        self.roi_coords_edit = QPlainTextEdit()
        self.roi_coords_edit.setReadOnly(True)
        self.roi_coords_edit.setPlaceholderText("点击「开始选择区域」后在图像上依次点击4个点")
        self.roi_coords_edit.setMaximumHeight(70)
        roi_col.addWidget(QLabel("ROI 顶点坐标"))
        roi_col.addWidget(self.roi_coords_edit)

        self.roi_strategy_combo = QComboBox()
        self.roi_strategy_combo.addItems(["centroid", "overlap", "full"])
        roi_col.addWidget(QLabel("过滤策略"))
        roi_col.addWidget(self.roi_strategy_combo)

        self.roi_manual_edit = QPlainTextEdit()
        self.roi_manual_edit.setPlaceholderText("x1,y1  x2,y2  x3,y3  x4,y4")
        self.roi_manual_edit.setMaximumHeight(60)
        roi_col.addWidget(QLabel("或手动输入坐标"))
        roi_col.addWidget(self.roi_manual_edit)

        # Calibration
        calib_grp = QGroupBox("相机标定 · Homography 4点法")
        calib_lay = QVBoxLayout(calib_grp)
        self.calib_src_edit = QPlainTextEdit()
        self.calib_src_edit.setPlaceholderText("例如: 200,500  800,500  800,100  200,100")
        self.calib_src_edit.setMaximumHeight(60)
        calib_lay.addWidget(QLabel("像素坐标（4个地面点）"))
        calib_lay.addWidget(self.calib_src_edit)
        self.calib_dst_edit = QPlainTextEdit()
        self.calib_dst_edit.setPlaceholderText("例如: 0,0  6,0  6,4  0,4")
        self.calib_dst_edit.setMaximumHeight(60)
        calib_lay.addWidget(QLabel("世界坐标（4点，单位：米）"))
        calib_lay.addWidget(self.calib_dst_edit)
        self.calib_autofill_btn = QPushButton("从 ROI 自动填充像素坐标")
        calib_lay.addWidget(self.calib_autofill_btn)
        roi_col.addWidget(calib_grp)

        row2.addLayout(roi_col, 2)
        lay.addLayout(row2)

        self._main_layout.addWidget(section)

    # ==================================================================
    # Alert Settings
    # ==================================================================

    def _build_alert_section(self):
        section = CollapsibleSection("告警设置")
        lay = QVBoxLayout(section.content())

        row1 = QHBoxLayout()
        _, _, self.athresh_row = _slider_spin_row("告警阈值", 0.1, 1.0, 0.5, 0.05, section.content())
        self.athresh_slider = self.athresh_row.layout().itemAt(1).widget()
        self.athresh_spin = self.athresh_row.layout().itemAt(2).widget()
        row1.addWidget(self.athresh_row)
        row1.addWidget(QLabel("期望最大目标数"))
        self.alert_max_spin = QSpinBox()
        self.alert_max_spin.setRange(1, 1000)
        self.alert_max_spin.setValue(10)
        row1.addWidget(self.alert_max_spin)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        _, _, self.wcount_row = _slider_spin_row("数量权重", 0.0, 1.0, 0.4, 0.1, section.content())
        self.wcount_slider = self.wcount_row.layout().itemAt(1).widget()
        self.wcount_spin = self.wcount_row.layout().itemAt(2).widget()
        _, _, self.warea_row = _slider_spin_row("面积权重", 0.0, 1.0, 0.6, 0.1, section.content())
        self.warea_slider = self.warea_row.layout().itemAt(1).widget()
        self.warea_spin = self.warea_row.layout().itemAt(2).widget()
        row2.addWidget(self.wcount_row)
        row2.addWidget(self.warea_row)
        lay.addLayout(row2)

        # Email sub-group
        email_grp = QGroupBox("邮件告警 (SMTP)")
        email_lay = QVBoxLayout(email_grp)
        self.email_enabled_cb = QCheckBox("启用邮件告警")
        email_lay.addWidget(self.email_enabled_cb)
        erow1 = QHBoxLayout()
        self.email_server_edit = QLineEdit("smtp.163.com")
        erow1.addWidget(QLabel("SMTP 服务器"))
        erow1.addWidget(self.email_server_edit, 2)
        self.email_port_spin = QSpinBox()
        self.email_port_spin.setRange(1, 65535)
        self.email_port_spin.setValue(465)
        erow1.addWidget(QLabel("端口"))
        erow1.addWidget(self.email_port_spin, 1)
        email_lay.addLayout(erow1)
        erow2 = QHBoxLayout()
        self.email_sender_edit = QLineEdit()
        self.email_sender_edit.setPlaceholderText("your@163.com")
        erow2.addWidget(QLabel("发件邮箱"))
        erow2.addWidget(self.email_sender_edit, 2)
        self.email_pass_edit = QLineEdit()
        self.email_pass_edit.setEchoMode(QLineEdit.Password)
        self.email_pass_edit.setPlaceholderText("SMTP 授权码（非登录密码）")
        erow2.addWidget(QLabel("授权码"))
        erow2.addWidget(self.email_pass_edit, 2)
        email_lay.addLayout(erow2)
        self.email_recv_edit = QLineEdit()
        self.email_recv_edit.setPlaceholderText("receiver@qq.com (多个用逗号或分号分隔)")
        email_lay.addWidget(QLabel("收件邮箱"))
        email_lay.addWidget(self.email_recv_edit)
        lay.addWidget(email_grp)

        self._main_layout.addWidget(section)

    # ==================================================================
    # Task Tabs
    # ==================================================================

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self._build_monitor_tab()
        self._build_single_tab()
        self._build_batch_tab()
        self._main_layout.addWidget(self.tabs)

    # --- Tab 1: Monitor ---
    def _build_monitor_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        row1 = QHBoxLayout()
        self.monitor_dir_edit = QLineEdit(str(Path(__file__).parent.parent))
        row1.addWidget(QLabel("监控目录"), 1)
        row1.addWidget(self.monitor_dir_edit, 4)
        self.poll_interval_spin = QSpinBox()
        self.poll_interval_spin.setRange(1, 300)
        self.poll_interval_spin.setValue(10)
        row1.addWidget(QLabel("轮询间隔 (秒)"))
        row1.addWidget(self.poll_interval_spin, 1)
        self.stable_time_spin = QSpinBox()
        self.stable_time_spin.setRange(1, 60)
        self.stable_time_spin.setValue(5)
        row1.addWidget(QLabel("文件稳定等待 (秒)"))
        row1.addWidget(self.stable_time_spin, 1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.monitor_start_btn = QPushButton("启动监控")
        self.monitor_stop_btn = QPushButton("停止监控")
        self.monitor_refresh_btn = QPushButton("刷新日志")
        row2.addWidget(self.monitor_start_btn, 2)
        row2.addWidget(self.monitor_stop_btn, 1)
        row2.addWidget(self.monitor_refresh_btn, 1)
        lay.addLayout(row2)

        self.monitor_status_lbl = QLabel("就绪")
        lay.addWidget(QLabel("状态"))
        lay.addWidget(self.monitor_status_lbl)

        self.monitor_log_edit = QPlainTextEdit()
        self.monitor_log_edit.setReadOnly(True)
        self.monitor_log_edit.setMaximumBlockCount(500)
        lay.addWidget(QLabel("日志"))
        lay.addWidget(self.monitor_log_edit)

        self.tabs.addTab(tab, "实时监控")

    # --- Tab 2: Single Video ---
    def _build_single_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        row1 = QHBoxLayout()
        self.video_dir_edit = QLineEdit(str(Path(__file__).parent.parent))
        row1.addWidget(QLabel("视频所在目录"), 1)
        row1.addWidget(self.video_dir_edit, 3)
        self.scan_btn = QPushButton("扫描")
        row1.addWidget(self.scan_btn, 1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.video_combo = QComboBox()
        self.video_combo.setEditable(True)
        self.scan_info_lbl = QLabel("")
        row2.addWidget(QLabel("选择视频"), 1)
        row2.addWidget(self.video_combo, 4)
        row2.addWidget(self.scan_info_lbl, 2)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.start_sec_spin = QDoubleSpinBox()
        self.start_sec_spin.setRange(0, 999999)
        self.start_sec_spin.setDecimals(1)
        self.start_sec_spin.setValue(0)
        self.end_sec_spin = QDoubleSpinBox()
        self.end_sec_spin.setRange(0, 999999)
        self.end_sec_spin.setDecimals(1)
        self.end_sec_spin.setValue(0)
        self.end_sec_spin.setSpecialValueText("末尾")
        self.output_annotated_cb = QCheckBox("输出标注视频")
        self.output_annotated_cb.setChecked(True)
        self.process_btn = QPushButton("开始处理")
        row3.addWidget(QLabel("起始 (秒)"))
        row3.addWidget(self.start_sec_spin, 1)
        row3.addWidget(QLabel("结束 (秒)"))
        row3.addWidget(self.end_sec_spin, 1)
        row3.addWidget(self.output_annotated_cb)
        row3.addWidget(self.process_btn)
        lay.addLayout(row3)

        row4 = QHBoxLayout()
        self.result_preview_label = QLabel("暂无检测结果\n\n请选择视频并点击「开始处理」")
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setMinimumHeight(300)
        self.result_preview_label.setStyleSheet(
            "border: 1px solid #3a3a3a; background: #1a1a1a; color: #666;"
            " font-size: 14px;"
        )
        self.result_summary_edit = QPlainTextEdit()
        self.result_summary_edit.setReadOnly(True)
        row4.addWidget(self.result_preview_label, 3)
        row4.addWidget(self.result_summary_edit, 2)
        lay.addLayout(row4)

        self.result_csv_lbl = QLabel("")
        self.result_csv_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(QLabel("输出文件"))
        lay.addWidget(self.result_csv_lbl)

        self.tabs.addTab(tab, "单视频检测")

    # --- Tab 3: Batch ---
    def _build_batch_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        row1 = QHBoxLayout()
        self.batch_dir_edit = QLineEdit(str(Path(__file__).parent.parent))
        row1.addWidget(QLabel("视频根目录"), 1)
        row1.addWidget(self.batch_dir_edit, 3)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.batch_start_date_edit = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self.batch_start_date_edit.setPlaceholderText("YYYY-MM-DD")
        self.batch_end_date_edit = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        self.batch_end_date_edit.setPlaceholderText("YYYY-MM-DD")
        self.batch_start_time_edit = QLineEdit("00:00:00")
        self.batch_end_time_edit = QLineEdit("23:59:59")
        self.batch_scan_btn = QPushButton("扫描")
        row2.addWidget(QLabel("开始日期"))
        row2.addWidget(self.batch_start_date_edit)
        row2.addWidget(QLabel("结束日期"))
        row2.addWidget(self.batch_end_date_edit)
        row2.addWidget(QLabel("开始时间"))
        row2.addWidget(self.batch_start_time_edit)
        row2.addWidget(QLabel("结束时间"))
        row2.addWidget(self.batch_end_time_edit)
        row2.addWidget(self.batch_scan_btn)
        lay.addLayout(row2)

        self.batch_scan_info_lbl = QLabel("")
        lay.addWidget(self.batch_scan_info_lbl)

        self.batch_file_list = QListWidget()
        self.batch_file_list.setSelectionMode(QListWidget.MultiSelection)
        lay.addWidget(QLabel("待处理文件"))
        lay.addWidget(self.batch_file_list)

        row3 = QHBoxLayout()
        self.batch_resume_cb = QCheckBox("断点续跑")
        self.batch_resume_cb.setChecked(True)
        self.batch_process_btn = QPushButton("开始批量处理")
        self.batch_stop_btn = QPushButton("停止")
        row3.addWidget(self.batch_resume_cb)
        row3.addStretch()
        row3.addWidget(self.batch_process_btn)
        row3.addWidget(self.batch_stop_btn)
        lay.addLayout(row3)

        row4 = QHBoxLayout()
        self.batch_summary_edit = QPlainTextEdit()
        self.batch_summary_edit.setReadOnly(True)
        self.batch_log_edit = QPlainTextEdit()
        self.batch_log_edit.setReadOnly(True)
        self.batch_log_edit.setMaximumBlockCount(500)
        row4.addWidget(self.batch_summary_edit, 1)
        row4.addWidget(self.batch_log_edit, 2)
        lay.addLayout(row4)

        self.batch_output_dir_lbl = QLabel("")
        self.batch_output_dir_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(QLabel("结果目录"))
        lay.addWidget(self.batch_output_dir_lbl)

        self.tabs.addTab(tab, "批量处理")

    # ==================================================================
    # Signal / Slot connections
    # ==================================================================

    def _connect_signals(self):
        # --- Scene ---
        self.scene_save_btn.clicked.connect(self._on_save_scene)
        self.scene_load_btn.clicked.connect(self._on_load_scene)
        self.scene_refresh_btn.clicked.connect(self._refresh_scenes)
        self.scene_delete_btn.clicked.connect(self._on_delete_scene)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.device_combo.currentTextChanged.connect(self._on_model_changed)

        # --- ROI & Preview ---
        self.load_preview_btn.clicked.connect(self._on_load_preview)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        self.roi_start_btn.clicked.connect(self._on_roi_start)
        self.roi_clear_btn.clicked.connect(self._on_roi_clear)
        self.roi_manual_edit.textChanged.connect(self._on_roi_manual)
        self.calib_autofill_btn.clicked.connect(self._on_calib_autofill)
        self.preview_label.roi_changed.connect(self._on_roi_points_changed)

        # --- Monitor ---
        self.monitor_start_btn.clicked.connect(self._on_monitor_start)
        self.monitor_stop_btn.clicked.connect(self._on_monitor_stop)
        self.monitor_refresh_btn.clicked.connect(self._on_monitor_refresh)

        # --- Single ---
        self.scan_btn.clicked.connect(self._on_scan_videos)
        self.process_btn.clicked.connect(self._on_process_single)

        # --- Batch ---
        self.batch_scan_btn.clicked.connect(self._on_batch_scan)
        self.batch_process_btn.clicked.connect(self._on_batch_process)
        self.batch_stop_btn.clicked.connect(self._on_batch_stop)

    # ==================================================================
    # Scene slots
    # ==================================================================

    def _refresh_scenes(self):
        names = refresh_scene_list()
        self.scene_combo.clear()
        self.scene_combo.addItems(names)

    def _on_save_scene(self):
        status, choices, value = save_scene(
            self.scene_name_edit.text(),
            self.model_combo.currentText(),
            self.conf_spin.value(),
            self.nms_spin.value(),
            self.fskip_spin.value(),
            self.device_combo.currentText(),
            self.roi_coords_edit.toPlainText().strip(),
            self.roi_strategy_combo.currentText(),
            self.calib_src_edit.toPlainText().strip(),
            self.calib_dst_edit.toPlainText().strip(),
            self.athresh_spin.value(),
            self.wcount_spin.value(),
            self.warea_spin.value(),
            self.alert_max_spin.value(),
        )
        self.scene_status_lbl.setText(status)
        if choices:
            self.scene_combo.clear()
            self.scene_combo.addItems(choices)
            if value:
                self.scene_combo.setCurrentText(value)

    def _on_load_scene(self):
        d = load_scene(self.scene_combo.currentText())
        self.model_combo.setCurrentText(d["model_path"])
        self.conf_spin.setValue(d["confidence"])
        self.nms_spin.setValue(d["nms_iou"])
        self.fskip_spin.setValue(d["frame_skip"])
        self.device_combo.setCurrentText(d["device"])
        self.roi_coords_edit.setPlainText(d["roi_text"])
        self.roi_strategy_combo.setCurrentText(d["roi_strategy"])
        self.calib_src_edit.setPlainText(d["calib_src"])
        self.calib_dst_edit.setPlainText(d["calib_dst"])
        self.athresh_spin.setValue(d["alert_threshold"])
        self.wcount_spin.setValue(d["alert_w_count"])
        self.warea_spin.setValue(d["alert_w_area"])
        self.alert_max_spin.setValue(d["alert_max_count"])
        self.roi_points_state = d["roi_points"]
        self.scene_status_lbl.setText(d["status"])
        self.model_status_lbl.setText(d["model_msg"])

    def _on_delete_scene(self):
        status, choices = delete_scene(self.scene_combo.currentText())
        self.scene_status_lbl.setText(status)
        if choices is not None:
            self.scene_combo.clear()
            self.scene_combo.addItems(choices)

    def _on_model_changed(self):
        msg = on_model_change(
            self.model_combo.currentText(),
            self.device_combo.currentText(),
        )
        self.model_status_lbl.setText(msg)

    # ==================================================================
    # ROI / Preview slots
    # ==================================================================

    def _on_load_preview(self):
        path = self.preview_path_edit.text().strip()
        frame, info_str, total_frames, meta = load_preview_video(path)
        if frame is None:
            self.frame_info_lbl.setText(info_str)
            return
        self.frame_info_lbl.setText(info_str)
        self.frame_slider.setMaximum(max(0, total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(True)
        self.preview_video_meta = meta
        self.preview_label.set_numpy_image(frame)
        self.roi_points_state = []

    def _on_frame_slider(self, idx):
        if not self.preview_video_meta:
            return
        frame, info_str = seek_preview_frame(
            self.preview_video_meta, idx, self.roi_points_state)
        if frame is not None:
            self.preview_label.set_numpy_image(frame)
        self.frame_info_lbl.setText(info_str)

    def _on_roi_start(self):
        self.roi_selection_active = True
        self.roi_coords_edit.setPlainText("请在预览图像上依次点击4个顶点（任意顺序均可）")
        self.roi_points_state = []
        self.preview_label.set_selection_active(True)

    def _on_roi_clear(self):
        frame, msg, active, pts = clear_roi_selection(
            self.preview_video_meta, self.frame_slider.value(), self.roi_points_state)
        self.roi_selection_active = active
        self.roi_points_state = pts
        self.roi_coords_edit.setPlainText(msg)
        self.preview_label.set_selection_active(False)
        self.preview_label.set_roi_points([])
        if frame is not None:
            self.preview_label.set_numpy_image(frame)

    def _on_roi_manual(self):
        text = self.roi_manual_edit.toPlainText()
        if not text.strip():
            return
        frame, coord_text, pts = sync_manual_roi(
            text, self.preview_video_meta, self.frame_slider.value())
        self.roi_points_state = pts
        self.roi_coords_edit.setPlainText(coord_text)
        self.preview_label.set_roi_points(pts)
        if frame is not None:
            self.preview_label.set_numpy_image(frame)
            self.preview_label.set_selection_active(False)
            self.roi_selection_active = False

    def _on_calib_autofill(self):
        src, dst = autofill_calib_from_roi(self.roi_points_state)
        if src:
            self.calib_src_edit.setPlainText(src)
            self.calib_dst_edit.setPlainText(dst)

    def _on_roi_points_changed(self, points):
        self.roi_points_state = points
        from geometry import format_point_list
        self.roi_coords_edit.setPlainText(format_point_list(points))

    # ==================================================================
    # Monitor slots
    # ==================================================================

    def _on_monitor_start(self):
        if self._monitor_worker and self._monitor_worker.isRunning():
            self._monitor_worker.stop()
            self._monitor_worker.wait(2000)

        self._monitor_worker = MonitorWorker(
            self.monitor_dir_edit.text(),
            self.conf_spin.value(), self.nms_spin.value(),
            self.fskip_spin.value(),
            self.roi_coords_edit.toPlainText().strip(),
            self.roi_strategy_combo.currentText(),
            self.poll_interval_spin.value(), self.stable_time_spin.value(),
            self.athresh_spin.value(), self.wcount_spin.value(),
            self.warea_spin.value(), self.alert_max_spin.value(),
            self.email_enabled_cb.isChecked(),
            self.email_server_edit.text(), self.email_port_spin.value(),
            self.email_sender_edit.text(), self.email_pass_edit.text(),
            self.email_recv_edit.text(),
        )
        self._monitor_worker.log_signal.connect(self._on_monitor_log)
        self._monitor_worker.status_signal.connect(self.monitor_status_lbl.setText)
        self._monitor_worker.start()
        self.monitor_status_lbl.setText("监控运行中...")

    def _on_monitor_stop(self):
        if self._monitor_worker:
            self._monitor_worker.stop()
        status, log_text = stop_monitor()
        self.monitor_status_lbl.setText(status)
        self.monitor_log_edit.setPlainText(log_text)

    def _on_monitor_refresh(self):
        self.monitor_log_edit.setPlainText(_monitor_log_text())

    def _on_monitor_log(self, msg: str):
        self.monitor_log_edit.appendPlainText(msg)

    # ==================================================================
    # Single Video slots
    # ==================================================================

    def _on_scan_videos(self):
        files, info = scan_video_dir(self.video_dir_edit.text())
        self.video_combo.clear()
        self.video_combo.addItems(files)
        self.scan_info_lbl.setText(info)

    def _on_process_single(self):
        video_path = self.video_combo.currentText()
        if not video_path:
            self.result_summary_edit.setPlainText("请先选择视频文件")
            return

        self.process_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        self._single_worker = SingleVideoWorker(
            video_path,
            self.conf_spin.value(), self.nms_spin.value(),
            self.fskip_spin.value(), self.start_sec_spin.value(),
            self.end_sec_spin.value(),
            self.roi_coords_edit.toPlainText().strip(),
            self.roi_strategy_combo.currentText(),
            self.calib_src_edit.toPlainText().strip(),
            self.calib_dst_edit.toPlainText().strip(),
            self.athresh_spin.value(), self.wcount_spin.value(),
            self.warea_spin.value(), self.alert_max_spin.value(),
            self.output_annotated_cb.isChecked(),
            self.email_enabled_cb.isChecked(),
            self.email_server_edit.text(), self.email_port_spin.value(),
            self.email_sender_edit.text(), self.email_pass_edit.text(),
            self.email_recv_edit.text(),
        )
        self._single_worker.progress_signal.connect(self._on_progress)
        self._single_worker.finished_signal.connect(self._on_single_finished)
        self._single_worker.error_signal.connect(self._on_single_error)
        self._single_worker.finished.connect(self._on_single_done)
        self._single_worker.start()

    def _on_single_finished(self, preview, summary, csv_path):
        if preview is not None:
            pix = numpy_to_pixmap(preview, self.result_preview_label.width(), 300)
            if pix:
                self.result_preview_label.setPixmap(pix)
            else:
                self.result_preview_label.setText("无检测帧预览")
        else:
            self.result_preview_label.setText("无检测帧")
        self.result_summary_edit.setPlainText(summary)
        self.result_csv_lbl.setText(csv_path)
        self._progress_bar.setVisible(False)

    def _on_single_error(self, msg):
        self.result_summary_edit.setPlainText(msg)
        self._progress_bar.setVisible(False)

    def _on_single_done(self):
        self.process_btn.setEnabled(True)

    def _on_progress(self, current, total, desc):
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        self._status_label.setText(desc)

    # ==================================================================
    # Batch slots
    # ==================================================================

    def _on_batch_scan(self):
        info, files = scan_directory(
            self.batch_dir_edit.text(),
            self.batch_start_date_edit.text(),
            self.batch_end_date_edit.text(),
            self.batch_start_time_edit.text(),
            self.batch_end_time_edit.text(),
        )
        self.batch_scan_info_lbl.setText(info)
        self.batch_file_list.clear()
        for f in files:
            item = QListWidgetItem(f)
            item.setCheckState(Qt.Checked)
            self.batch_file_list.addItem(item)

    def _on_batch_process(self):
        file_list = []
        for i in range(self.batch_file_list.count()):
            item = self.batch_file_list.item(i)
            if item.checkState() == Qt.Checked:
                file_list.append(item.text())
        if not file_list:
            self.batch_summary_edit.setPlainText("没有待处理的视频")
            return

        self.batch_process_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self.batch_log_edit.clear()

        self._batch_worker = BatchWorker(
            file_list,
            self.conf_spin.value(), self.nms_spin.value(),
            self.fskip_spin.value(),
            self.roi_coords_edit.toPlainText().strip(),
            self.roi_strategy_combo.currentText(),
            self.calib_src_edit.toPlainText().strip(),
            self.calib_dst_edit.toPlainText().strip(),
            self.athresh_spin.value(), self.wcount_spin.value(),
            self.warea_spin.value(), self.alert_max_spin.value(),
            self.batch_resume_cb.isChecked(),
            self.email_enabled_cb.isChecked(),
            self.email_server_edit.text(), self.email_port_spin.value(),
            self.email_sender_edit.text(), self.email_pass_edit.text(),
            self.email_recv_edit.text(),
        )
        self._batch_worker.progress_signal.connect(self._on_progress)
        self._batch_worker.log_signal.connect(lambda msg: self.batch_log_edit.appendPlainText(msg))
        self._batch_worker.finished_signal.connect(self._on_batch_finished)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.start()

    def _on_batch_finished(self, summary, log_text, output_dir):
        self.batch_summary_edit.setPlainText(summary)
        self.batch_output_dir_lbl.setText(output_dir)
        self._progress_bar.setVisible(False)

    def _on_batch_done(self):
        self.batch_process_btn.setEnabled(True)

    def _on_batch_stop(self):
        stop_batch()
        if self._batch_worker:
            self._batch_worker.stop()
        self.batch_summary_edit.setPlainText("已发送停止信号...")

    # ==================================================================
    # Cleanup
    # ==================================================================

    def closeEvent(self, event):
        if self._monitor_worker and self._monitor_worker.isRunning():
            self._monitor_worker.stop()
            self._monitor_worker.wait(3000)
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self._batch_worker.wait(3000)
        if self._single_worker and self._single_worker.isRunning():
            self._single_worker.wait(3000)
        super().closeEvent(event)
