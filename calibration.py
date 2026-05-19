import json
import cv2
import numpy as np


class CameraCalibration:
    """基于 Homography 的相机标定：像素坐标 ↔ 地面真实坐标（米）"""

    def __init__(self):
        self._H: np.ndarray | None = None        # 像素 → 世界 的单应性矩阵
        self._H_inv: np.ndarray | None = None     # 世界 → 像素
        self._src_points: list[tuple] | None = None  # 4个像素点
        self._dst_points: list[tuple] | None = None  # 4个世界坐标点
        self._origin: tuple | None = None            # 世界坐标原点描述
        self._resolution: tuple | None = None        # 标定时使用的视频分辨率

    # ------------------------------------------------------------------
    def set_homography(self, src_points: list[tuple], dst_points: list[tuple],
                       resolution: tuple, origin_desc: str = ""):
        """
        设置单应性矩阵。

        src_points: 图像上的 4 个点 [(px1,py1), ...]
        dst_points: 对应的真实世界坐标 [(mx1,my1), ...]（单位：米）
        resolution: (width, height) 标定时使用的视频分辨率
        origin_desc: 坐标原点描述（如 "画面左下角"）
        """
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("需要恰好 4 对点来计算单应性矩阵")

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self._H, mask = cv2.findHomography(src, dst)
        if self._H is None:
            raise RuntimeError("单应性矩阵计算失败，检查点是否共线或距离过近")
        self._H_inv = np.linalg.inv(self._H)
        self._src_points = [tuple(p) for p in src_points]
        self._dst_points = [tuple(p) for p in dst_points]
        self._resolution = tuple(resolution)
        self._origin = origin_desc

    # ------------------------------------------------------------------
    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """将像素坐标 (px, py) 映射到世界坐标 (wx, wy)，单位米。"""
        if self._H is None:
            raise RuntimeError("标定未设置，请先调用 set_homography()")
        pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pt, self._H)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    def world_to_pixel(self, wx: float, wy: float) -> tuple[float, float]:
        """将世界坐标 (wx, wy) 映射回像素坐标 (px, py)。"""
        if self._H_inv is None:
            raise RuntimeError("标定未设置，请先调用 set_homography()")
        pt = np.array([[wx, wy]], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pt, self._H_inv)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

    # ------------------------------------------------------------------
    def estimate_size(self, bbox: tuple) -> tuple[float, float]:
        """
        估算 bbox 对应目标的实际宽 × 高（米）。

        bbox: (x1, y1, x2, y2) 像素坐标
        假设目标站立在地面上：
        - 宽度从底边两端点映射到世界后计算距离
        - 高度的估算是像素高度 × 底边处的尺度因子（m/px）
        """
        if self._H is None:
            raise RuntimeError("标定未设置")

        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        if pixel_height <= 0:
            return 0.0, 0.0

        # 底边两点映射到世界坐标
        bl_wx, bl_wy = self.pixel_to_world(x1, y2)  # 底边左
        br_wx, br_wy = self.pixel_to_world(x2, y2)  # 底边右

        # 实际宽度 = 底边两端点在世界中的欧氏距离
        width_m = np.sqrt((br_wx - bl_wx) ** 2 + (br_wy - bl_wy) ** 2)

        # 尺度因子：底边中点处 1 像素对应多少米
        cx = (x1 + x2) / 2
        ref_wx1, ref_wy1 = self.pixel_to_world(cx, y2)
        ref_wx2, ref_wy2 = self.pixel_to_world(cx + 1, y2)
        scale = np.sqrt((ref_wx2 - ref_wx1) ** 2 + (ref_wy2 - ref_wy1) ** 2)

        # 实际高度 = 像素高度 × 尺度因子（近似）
        height_m = pixel_height * scale

        return width_m, height_m

    # ------------------------------------------------------------------
    def is_calibrated(self) -> bool:
        return self._H is not None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """导出标定参数为字典，用于 JSON 持久化。"""
        if not self.is_calibrated():
            return {}
        return {
            "src_points": list(self._src_points),
            "dst_points": list(self._dst_points),
            "resolution": list(self._resolution),
            "origin": self._origin or "",
        }

    def save(self, filepath: str):
        """保存标定参数到 JSON 文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """从 JSON 文件加载标定参数。"""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.set_homography(
            src_points=[tuple(p) for p in d["src_points"]],
            dst_points=[tuple(p) for p in d["dst_points"]],
            resolution=tuple(d["resolution"]),
            origin_desc=d.get("origin", ""),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def suggest_default_points(image_width: int, image_height: int) -> dict:
        """
        为首次标定提供建议的 4 个像素点位置（画面四个角内缩 15%）。
        返回格式适合 Gradio 前端使用。
        """
        m = 0.15
        return {
            "src_points": [
                (int(image_width * m), int(image_height * (1 - m))),
                (int(image_width * (1 - m)), int(image_height * (1 - m))),
                (int(image_width * (1 - m)), int(image_height * m)),
                (int(image_width * m), int(image_height * m)),
            ],
            "resolution": [image_width, image_height],
        }
