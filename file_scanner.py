import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional


class ProcessedTracker:
    """追踪已处理的视频文件，避免重复处理"""

    def __init__(self, record_path: str = "processed.json"):
        self.record_path = record_path
        self._records: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.record_path):
            with open(self.record_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.record_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, ensure_ascii=False)

    def is_processed(self, filepath: str) -> bool:
        return filepath in self._records

    def mark_processed(self, filepath: str, result_summary: dict | None = None):
        self._records[filepath] = {
            "processed_at": datetime.now().isoformat(),
            "summary": result_summary or {},
        }
        self._save()

    def unmark(self, filepath: str):
        self._records.pop(filepath, None)
        self._save()

    def get_summary(self, filepath: str) -> dict | None:
        rec = self._records.get(filepath)
        return rec["summary"] if rec else None

    def clear(self):
        self._records = {}
        self._save()


class FileScanner:
    """扫描监控视频目录，支持时间范围过滤和新文件监控"""

    # 常见的日期目录结构模式
    DATE_DIR_PATTERNS = [
        re.compile(r"^(\d{4})/(\d{2})/(\d{2})$"),       # YYYY/MM/DD
        re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"),        # YYYY-MM-DD
    ]

    # 从文件名提取时间戳的模式
    FILENAME_TIME_PATTERNS = [
        # CAM20260518143000.mp4 → 日期+时间都在文件名中
        re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})"),
        # 2026-05-18_14-30-00.mp4
        re.compile(r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})[.-](\d{2})[.-](\d{2})"),
    ]

    def __init__(self, root_dir: str, tracker: ProcessedTracker | None = None):
        self.root_dir = root_dir
        self.tracker = tracker or ProcessedTracker()

    # ------------------------------------------------------------------
    def scan(self, start_dt: datetime | None = None,
             end_dt: datetime | None = None,
             extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
             skip_processed: bool = True) -> list[dict]:
        """
        扫描 root_dir 下所有视频文件，返回时间范围内的文件列表。

        返回: [{"path": ..., "datetime": datetime, "size": bytes}, ...]
        按时间升序排列。
        """
        results = []
        root = Path(self.root_dir)

        if not root.exists():
            return results

        for filepath in root.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in extensions:
                continue

            dt = self._extract_datetime(str(filepath))
            if dt is None:
                continue

            if start_dt and dt < start_dt:
                continue
            if end_dt and dt > end_dt:
                continue

            if skip_processed and self.tracker.is_processed(str(filepath)):
                continue

            results.append({
                "path": str(filepath),
                "datetime": dt,
                "size": filepath.stat().st_size,
            })

        results.sort(key=lambda x: x["datetime"])
        return results

    # ------------------------------------------------------------------
    def scan_by_date_range(self, start_date: str, end_date: str,
                           start_time: str = "00:00:00",
                           end_time: str = "23:59:59",
                           **kwargs) -> list[dict]:
        """
        便捷方法：用字符串指定日期范围。
        start_date / end_date: "2026-05-18"
        start_time / end_time: "14:30:00"
        """
        start_dt = datetime.fromisoformat(f"{start_date}T{start_time}")
        end_dt = datetime.fromisoformat(f"{end_date}T{end_time}")
        return self.scan(start_dt, end_dt, **kwargs)

    # ------------------------------------------------------------------
    def watch_for_new_files(
        self,
        callback: Callable[[str], None],
        poll_interval: float = 10.0,
        stable_time: float = 5.0,
        extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
        stop_event: Optional["threading.Event"] = None,
    ):
        """
        持续监控根目录下的新视频文件。

        callback: 发现新文件且文件稳定后的回调，参数为文件路径
        poll_interval: 轮询间隔（秒）
        stable_time: 文件修改后需稳定的秒数，确保写入完成
        stop_event: 外部停止信号（threading.Event）
        """
        import threading

        if stop_event is None:
            stop_event = threading.Event()

        known_files: set[str] = set()
        # 初始化已知文件列表
        for f in self.scan(skip_processed=False):
            known_files.add(f["path"])

        while not stop_event.is_set():
            try:
                current_files = set()
                root = Path(self.root_dir)
                if root.exists():
                    for filepath in root.rglob("*"):
                        if not filepath.is_file():
                            continue
                        if filepath.suffix.lower() not in extensions:
                            continue
                        current_files.add(str(filepath))

                new_files = current_files - known_files

                for fp in new_files:
                    # 检查文件是否还在写入（修改时间稳定检查）
                    if self._is_file_stable(fp, stable_time):
                        known_files.add(fp)
                        callback(fp)

            except Exception:
                pass

            stop_event.wait(poll_interval)

    # ------------------------------------------------------------------
    def _extract_datetime(self, filepath: str) -> datetime | None:
        """从文件路径提取日期时间，优先从目录结构推断，其次从文件名。"""
        rel = os.path.relpath(filepath, self.root_dir)
        parts = rel.replace("\\", "/").split("/")

        # 尝试从目录路径提取日期
        year = month = day = None
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else ""

        for pattern in self.DATE_DIR_PATTERNS:
            # 检查每一层子目录
            for i in range(len(parts) - 1):
                subpath = "/".join(parts[i:i + 3]) if i + 3 <= len(parts) else ""
                m = pattern.match(parts[i])
                if m:
                    # 单个目录匹配到完整日期 (YYYY-MM-DD)
                    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    break
            if year:
                break

        # 尝试从更深的目录结构 (YYYY/MM/DD) 中提取
        if year is None and len(parts) >= 4:
            for i in range(len(parts) - 2):
                try:
                    y, mo, d = int(parts[i]), int(parts[i+1]), int(parts[i+2])
                    if 2020 <= y <= 2040 and 1 <= mo <= 12 and 1 <= d <= 31:
                        year, month, day = y, mo, d
                        break
                except ValueError:
                    continue

        # 从文件名提取时间
        filename = parts[-1] if parts else ""
        hour = minute = sec = 0

        for pattern in self.FILENAME_TIME_PATTERNS:
            m = pattern.search(filename)
            if m:
                if year is None:
                    year = int(m.group(1))
                month = int(m.group(2)) if month is None else month
                day = int(m.group(3)) if day is None else day
                hour = int(m.group(4))
                minute = int(m.group(5))
                sec = int(m.group(6))
                break

        if year is None:
            return None

        try:
            return datetime(year, month or 1, day or 1, hour, minute, sec)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _is_file_stable(filepath: str, stable_sec: float) -> bool:
        """检查文件在 stable_sec 秒内未被修改"""
        try:
            mtime = os.path.getmtime(filepath)
            return (time.time() - mtime) >= stable_sec
        except OSError:
            return False


class SceneConfig:
    """场景配置：封装一个监控场景的全部检测参数（模型、ROI、标定、告警）"""

    def __init__(self, name: str):
        self.name = name
        # 检测参数
        self.model_path = "ep950-loss0.050-val_loss0.055.pth"
        self.confidence = 0.25
        self.nms_iou = 0.3
        self.frame_skip = 10
        self.device = "auto"
        # ROI
        self.roi_points: list[tuple] = []
        self.roi_strategy = "centroid"
        self.roi_resolution: tuple | None = None
        # 标定
        self.calib_src: list[tuple] = []
        self.calib_dst: list[tuple] = []
        self.calib_resolution: tuple | None = None
        # 告警
        self.alert_threshold = 0.5
        self.alert_weight_count = 0.4
        self.alert_weight_area = 0.6
        self.alert_max_count = 10

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": 2,
            "model_path": self.model_path,
            "confidence": self.confidence,
            "nms_iou": self.nms_iou,
            "frame_skip": self.frame_skip,
            "device": self.device,
            "roi_points": self.roi_points,
            "roi_strategy": self.roi_strategy,
            "roi_resolution": list(self.roi_resolution) if self.roi_resolution else None,
            "calib_src": self.calib_src,
            "calib_dst": self.calib_dst,
            "calib_resolution": list(self.calib_resolution) if self.calib_resolution else None,
            "alert_threshold": self.alert_threshold,
            "alert_weight_count": self.alert_weight_count,
            "alert_weight_area": self.alert_weight_area,
            "alert_max_count": self.alert_max_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConfig":
        s = cls(d["name"])
        # v1 backward-compat: source_dir is silently ignored
        # v2 fields with defaults
        s.model_path = d.get("model_path", "ep950-loss0.050-val_loss0.055.pth")
        s.confidence = d.get("confidence", 0.25)
        s.nms_iou = d.get("nms_iou", 0.3)
        s.frame_skip = d.get("frame_skip", 10)
        s.device = d.get("device", "auto")
        s.roi_points = d.get("roi_points", [])
        s.roi_strategy = d.get("roi_strategy", "centroid")
        s.roi_resolution = tuple(d["roi_resolution"]) if d.get("roi_resolution") else None
        s.calib_src = d.get("calib_src", [])
        s.calib_dst = d.get("calib_dst", [])
        s.calib_resolution = tuple(d["calib_resolution"]) if d.get("calib_resolution") else None
        s.alert_threshold = d.get("alert_threshold", 0.5)
        s.alert_weight_count = d.get("alert_weight_count", 0.4)
        s.alert_weight_area = d.get("alert_weight_area", 0.6)
        s.alert_max_count = d.get("alert_max_count", 10)
        return s

    # ------------------------------------------------------------------
    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "SceneConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TaskConfig:
    """轻量级任务定义：绑定场景 + 输入 + 处理模式（不持久化）"""
    mode: str                              # "monitor" | "single" | "batch"
    scene: "SceneConfig"
    input_dir: str = ""                    # monitor / batch 模式
    input_files: list[str] = field(default_factory=list)  # single / batch 模式
    start_sec: float = 0.0
    end_sec: float = 0.0
    start_date: str = ""
    end_date: str = ""
    start_time: str = "00:00:00"
    end_time: str = "23:59:59"
    poll_interval: int = 10
    stable_time: int = 5
    resume: bool = True
    output_annotated: bool = True
