import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

from detector import Detector


@dataclass
class Detection:
    """单条检测结果"""
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) 像素坐标
    centroid: tuple  # (cx, cy) 像素坐标，bbox 中心点
    world_position: Optional[tuple] = None  # (x, y) 米，需标定
    world_size: Optional[tuple] = None       # (w, h) 米，需标定


@dataclass
class FrameResult:
    """单帧检测结果"""
    frame_idx: int
    timestamp_sec: float
    detections: list[Detection] = field(default_factory=list)


@dataclass
class VideoResult:
    """单个视频的完整检测结果"""
    video_path: str
    video_fps: float
    video_total_frames: int
    video_duration_sec: float
    frames_processed: int
    frame_results: list[FrameResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_detections(self) -> int:
        return sum(len(fr.detections) for fr in self.frame_results)

    @property
    def class_counts(self) -> dict:
        counts: dict[str, int] = {}
        for fr in self.frame_results:
            for d in fr.detections:
                counts[d.label] = counts.get(d.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


class RoiMask:
    """多边形 ROI 掩码，用于过滤检测结果"""

    def __init__(self, points: list[tuple], resolution: tuple, strategy: str = "centroid"):
        """
        points: 多边形顶点列表 [(x1,y1), (x2,y2), ...]（像素坐标）
        resolution: (width, height)，用于分辨率比例换算
        strategy: 'centroid' | 'overlap' | 'full'
        """
        self.points = points
        self.resolution = resolution
        self.strategy = strategy

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "resolution": self.resolution,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoiMask":
        return cls(d["points"], tuple(d["resolution"]), d.get("strategy", "centroid"))

    def contains(self, bbox: tuple) -> bool:
        """判断 bbox (x1,y1,x2,y2) 是否在 ROI 内"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pts = np.array(self.points, dtype=np.int32)

        if self.strategy == "centroid":
            return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
        elif self.strategy == "full":
            return all(
                cv2.pointPolygonTest(pts, (px, py), False) >= 0
                for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            )
        elif self.strategy == "overlap":
            return any(
                cv2.pointPolygonTest(pts, (px, py), False) >= 0
                for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (cx, cy)]
            )
        return True

    def filter(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if self.contains(d.bbox)]

    def draw(self, image: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        pts = np.array(self.points, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
        return image


class VideoProcessor:
    """视频检测处理器"""

    def __init__(self, detector: Detector):
        self.detector = detector

    def process_video(
        self,
        video_path: str,
        conf: float = 0.25,
        nms_iou: float = 0.3,
        frame_skip: int = 1,
        start_sec: float = 0,
        end_sec: Optional[float] = None,
        roi: Optional[RoiMask] = None,
        calibration: Optional["CameraCalibration"] = None,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> VideoResult:
        """
        处理单个视频，返回完整检测结果。

        frame_skip: 每 N 帧检测一次（1=全检, 10=每10帧检1帧）
        start_sec / end_sec: 裁剪时间范围（秒），None 表示到结尾
        output_video_path: 若给定，输出带标注的视频
        progress_callback: (current_frame, total_frames) -> None
        """
        frame_skip = int(frame_skip)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_sec is None:
            end_sec = duration

        result = VideoResult(
            video_path=video_path,
            video_fps=fps,
            video_total_frames=total_frames,
            video_duration_sec=duration,
            frames_processed=0,
        )

        # 计算要处理的帧范围
        start_frame = max(0, int(start_sec * fps))
        end_frame = total_frames if end_sec >= duration else int(end_sec * fps)
        total_to_process = (end_frame - start_frame + frame_skip - 1) // frame_skip

        writer = None
        if output_video_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_skip, (w, h))

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            processed = 0

            for fi in range(start_frame, end_frame, frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = fi / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                annotated, raw_detections = self.detector.detect(
                    frame_rgb, conf=conf, nms_iou=nms_iou,
                )

                # 构建 Detection 列表
                detections = []
                for d in raw_detections:
                    x1, y1, x2, y2 = d["bbox"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    det = Detection(
                        label=d["label"],
                        confidence=d["confidence"],
                        bbox=(x1, y1, x2, y2),
                        centroid=(cx, cy),
                    )
                    if calibration is not None:
                        try:
                            wx, wy = calibration.pixel_to_world(cx, y2)
                            det.world_position = (round(wx, 3), round(wy, 3))
                            bw, bh = calibration.estimate_size(d["bbox"])
                            det.world_size = (round(bw, 3), round(bh, 3))
                        except Exception:
                            pass
                    detections.append(det)

                # ROI 过滤
                if roi is not None:
                    detections = roi.filter(detections)

                result.frame_results.append(FrameResult(
                    frame_idx=fi,
                    timestamp_sec=round(timestamp, 3),
                    detections=detections,
                ))

                # 写标注帧到输出视频
                if writer is not None:
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    if roi is not None:
                        roi.draw(annotated_bgr)
                    writer.write(annotated_bgr)

                processed += 1
                if progress_callback:
                    progress_callback(processed, total_to_process)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        result.frames_processed = len(result.frame_results)
        return result

    def process_video_stream(
        self,
        video_path: str,
        conf: float = 0.25,
        nms_iou: float = 0.3,
        frame_skip: int = 1,
        roi: Optional[RoiMask] = None,
    ) -> Generator[tuple[np.ndarray, list[Detection]], None, None]:
        """
        流式处理视频，逐帧 yield (annotated_frame, detections)，用于实时预览。
        """
        frame_skip = int(frame_skip)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if fi % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated, raw = self.detector.detect(
                        frame_rgb, conf=conf, nms_iou=nms_iou,
                    )

                    detections = []
                    for d in raw:
                        x1, y1, x2, y2 = d["bbox"]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        detections.append(Detection(
                            label=d["label"],
                            confidence=d["confidence"],
                            bbox=(x1, y1, x2, y2),
                            centroid=(cx, cy),
                        ))

                    if roi is not None:
                        detections = roi.filter(detections)
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        roi.draw(annotated_bgr)
                        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    yield annotated, detections

                fi += 1

        finally:
            cap.release()

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """读取视频基本信息（不检测）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        info = {
            "path": video_path,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": 0,
        }
        if info["fps"] > 0:
            info["duration_sec"] = info["total_frames"] / info["fps"]
        cap.release()
        return info


class AlertEvaluator:
    """告警评估器 — 当目标数量 + 框面积占比的加权和超过阈值时触发"""

    def __init__(self, threshold: float = 0.5, weight_count: float = 0.4,
                 weight_area: float = 0.6, max_expected_count: int = 10):
        """
        threshold: 告警阈值 (0-1)，加权分 >= threshold 时触发
        weight_count: 目标数量权重
        weight_area: 框面积占比权重
        max_expected_count: 用于归一化目标数量的期望最大值
        """
        self.threshold = threshold
        self.weight_count = weight_count
        self.weight_area = weight_area
        self.max_expected_count = max_expected_count

    def evaluate(self, detections: list[Detection], frame_width: int,
                 frame_height: int) -> tuple[bool, float, dict]:
        """
        评估单帧是否触发告警。
        返回: (alert_triggered, score, detail)
        """
        frame_area = frame_width * frame_height
        count = len(detections)
        count_score = min(count / self.max_expected_count, 1.0) if self.max_expected_count > 0 else 0

        bbox_area_total = 0
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            bbox_area_total += (x2 - x1) * (y2 - y1)
        area_ratio = min(bbox_area_total / frame_area, 1.0) if frame_area > 0 else 0

        score = self.weight_count * count_score + self.weight_area * area_ratio
        triggered = score >= self.threshold

        return triggered, round(score, 4), {
            "count": count,
            "count_score": round(count_score, 3),
            "bbox_area_ratio": round(area_ratio, 3),
            "weighted_score": round(score, 4),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "weight_count": self.weight_count,
            "weight_area": self.weight_area,
            "max_expected_count": self.max_expected_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlertEvaluator":
        return cls(
            threshold=d.get("threshold", 0.5),
            weight_count=d.get("weight_count", 0.4),
            weight_area=d.get("weight_area", 0.6),
            max_expected_count=d.get("max_expected_count", 10),
        )
