"""Pure geometry / image helpers — no state, no Gradio dependencies."""

import math
import os

import cv2
import numpy as np


def sort_points_convex(points: list) -> list:
    """Sort points in clockwise order around their centroid."""
    if len(points) < 3:
        return list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _polygon_area(points: list) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def is_quadrilateral_valid(points: list) -> bool:
    """Check that 4 points form a valid quadrilateral (area >= 100, edges >= 10)."""
    if len(points) != 4:
        return False
    if _polygon_area(points) < 100:
        return False
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        if math.hypot(x2 - x1, y2 - y1) < 10:
            return False
    return True


def draw_roi_on_frame(frame: np.ndarray, points: list) -> np.ndarray:
    """Draw ROI polygon and vertex markers on a copy of *frame* (RGB)."""
    out = frame.copy()
    if len(points) < 3:
        return out
    pts = np.array(points, dtype=np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 6, (255, 0, 0), -1)
        cv2.putText(out, str(i + 1), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return out


def parse_point_list(text: str) -> list:
    """Parse "x1,y1  x2,y2  ..."  or  "x1,y1; x2,y2; ..." into [(x,y), ...]."""
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    text = text.replace(";", " ").replace("\n", " ")
    points = []
    for part in text.split():
        part = part.strip().strip("()（）")
        if "," in part:
            try:
                x, y = part.split(",")
                points.append((float(x.strip()), float(y.strip())))
            except ValueError:
                continue
    return points


def format_point_list(points: list) -> str:
    """Format [(x,y), ...] → "x1,y1 x2,y2 ..."."""
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def load_video_frame(video_path: str, frame_idx: int = 0):
    """Read a single frame (RGB) from *video_path*. Returns (frame, info_dict)."""
    if not video_path or not os.path.isfile(video_path):
        return None, f"视频文件不存在: {video_path}"
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "无法打开视频"
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / fps if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, "无法读取指定帧"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        info = {"fps": fps, "total_frames": total, "width": w, "height": h, "duration_sec": dur}
        return frame_rgb, info
    except Exception as e:
        return None, f"加载帧失败: {e}"
