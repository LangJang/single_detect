import os
import re
from datetime import datetime
from pathlib import Path


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

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    # ------------------------------------------------------------------
    def scan(self, start_dt: datetime | None = None,
             end_dt: datetime | None = None,
             extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> list[dict]:
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
                           end_time: str = "23:59:59") -> list[dict]:
        """
        便捷方法：用字符串指定日期范围。
        start_date / end_date: "2026-05-18"
        start_time / end_time: "14:30:00"
        """
        start_dt = datetime.fromisoformat(f"{start_date}T{start_time}")
        end_dt = datetime.fromisoformat(f"{end_date}T{end_time}")
        return self.scan(start_dt, end_dt)

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

