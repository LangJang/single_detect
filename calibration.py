import json
import cv2
import numpy as np


class CameraCalibration:
    """相机标定：像素坐标 ↔ 真实世界坐标（米）。

    支持两种模式：
    - scale:  线性比例尺，适用于正对垂直断面的水下拍摄（默认推荐）
    - homography: 单应性矩阵，适用于相机倾斜、有透视畸变的场景
    """

    def __init__(self):
        self._mode: str = ""                 # "scale" | "homography"
        # scale 模式
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._physical_size: tuple | None = None   # (width_m, height_m)
        # homography 模式
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._src_points: list[tuple] | None = None
        self._dst_points: list[tuple] | None = None
        self._origin: str = ""
        self._resolution: tuple | None = None

    # ------------------------------------------------------------------
    # ROI-based (推荐：ROI 4 像素点 + 物理尺寸 → 单应性)
    # ------------------------------------------------------------------

    def set_from_roi(self, roi_points: list[tuple], physical_width_m: float,
                     physical_height_m: float, resolution: tuple,
                     origin_x: float = 0.0, origin_y: float = 0.0):
        """从 ROI 4 个像素点 + 物理尺寸 + 原点偏移计算单应性矩阵（对角法）。

        roi_points:        4 个像素点（顺时针，从左上角起）
        physical_width_m:  ROI 区域实际物理宽度（米）
        physical_height_m: ROI 区域实际物理高度（米）
        resolution:        (width, height) 视频分辨率
        origin_x / origin_y: ROI 左上角在真实世界中的坐标（米）
        """
        if len(roi_points) != 4:
            raise ValueError("ROI 需要恰好 4 个顶点")
        x0, y0 = origin_x, origin_y
        w, h = physical_width_m, physical_height_m
        dst = [
            (x0, y0),
            (x0 + w, y0),
            (x0 + w, y0 + h),
            (x0, y0 + h),
        ]
        self.set_homography(roi_points, dst, resolution)

    # ------------------------------------------------------------------
    # Scale mode (简单比例尺，无 ROI 时使用)
    # ------------------------------------------------------------------

    def set_scale(self, physical_width_m: float, physical_height_m: float,
                  resolution: tuple, origin_desc: str = ""):
        """设置线性比例尺（水下垂直断面模式）。

        physical_width_m:  断面物理宽度（米）
        physical_height_m: 断面物理高度（米）
        resolution: (width, height) 视频分辨率
        """
        if physical_width_m <= 0 or physical_height_m <= 0:
            raise ValueError("物理尺寸必须大于 0")
        self._mode = "scale"
        self._scale_x = physical_width_m / resolution[0]
        self._scale_y = physical_height_m / resolution[1]
        self._physical_size = (physical_width_m, physical_height_m)
        self._resolution = tuple(resolution)
        self._origin = origin_desc

    # ------------------------------------------------------------------
    # Homography mode (透视校正)
    # ------------------------------------------------------------------

    def set_homography(self, src_points: list[tuple], dst_points: list[tuple],
                       resolution: tuple, origin_desc: str = ""):
        """设置单应性矩阵（透视校正模式）。

        src_points: 图像上的 4 个点 [(px1,py1), ...]
        dst_points: 对应的真实世界坐标 [(mx1,my1), ...]（单位：米）
        resolution: (width, height) 视频分辨率
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
        self._mode = "homography"

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """像素坐标 → 世界坐标（米）。"""
        if self._mode == "scale":
            return px * self._scale_x, py * self._scale_y
        if self._H is not None:
            pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(pt, self._H)
            return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        raise RuntimeError("标定未设置，请先调用 set_scale() 或 set_homography()")

    def world_to_pixel(self, wx: float, wy: float) -> tuple[float, float]:
        """世界坐标（米）→ 像素坐标。"""
        if self._mode == "scale":
            px = wx / self._scale_x if self._scale_x != 0 else 0
            py = wy / self._scale_y if self._scale_y != 0 else 0
            return px, py
        if self._H_inv is not None:
            pt = np.array([[wx, wy]], dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(pt, self._H_inv)
            return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        raise RuntimeError("标定未设置")

    # ------------------------------------------------------------------
    # 尺寸估算
    # ------------------------------------------------------------------

    def estimate_size(self, bbox: tuple) -> tuple[float, float]:
        """估算 bbox 对应目标的实际宽 × 高（米）。

        bbox: (x1, y1, x2, y2) 像素坐标

        scale 模式：像素宽高直接 × 比例尺
        homography 模式：四角全部映射到世界坐标，取对边平均距离
        """
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0

        if self._mode == "scale":
            return (x2 - x1) * self._scale_x, (y2 - y1) * self._scale_y

        if self._H is not None:
            # 映射四角到世界坐标
            tl = self.pixel_to_world(x1, y1)
            tr = self.pixel_to_world(x2, y1)
            bl = self.pixel_to_world(x1, y2)
            br = self.pixel_to_world(x2, y2)
            # 宽度 = 上下边世界距离的平均
            top_w = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
            bot_w = np.hypot(br[0] - bl[0], br[1] - bl[1])
            # 高度 = 左右边世界距离的平均
            left_h = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
            right_h = np.hypot(br[0] - tr[0], br[1] - tr[1])
            return round((top_w + bot_w) / 2, 3), round((left_h + right_h) / 2, 3)

        raise RuntimeError("标定未设置")

    # ------------------------------------------------------------------

    def is_calibrated(self) -> bool:
        return self._mode in ("scale", "homography")

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """导出标定参数为字典，用于 JSON 持久化。"""
        if not self.is_calibrated():
            return {}
        d = {
            "mode": self._mode,
            "resolution": list(self._resolution) if self._resolution else [],
            "origin": self._origin or "",
        }
        if self._mode == "scale":
            d["physical_width_m"] = self._physical_size[0]
            d["physical_height_m"] = self._physical_size[1]
        elif self._mode == "homography":
            d["src_points"] = list(self._src_points)
            d["dst_points"] = list(self._dst_points)
        return d

    def save(self, filepath: str):
        """保存标定参数到 JSON 文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        """从 JSON 文件加载标定参数。"""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        mode = d.get("mode", "homography")  # 兼容旧文件（无 mode 字段）
        resolution = tuple(d["resolution"]) if d.get("resolution") else None
        if mode == "scale":
            self.set_scale(
                physical_width_m=d["physical_width_m"],
                physical_height_m=d["physical_height_m"],
                resolution=resolution,
                origin_desc=d.get("origin", ""),
            )
        else:
            self.set_homography(
                src_points=[tuple(p) for p in d["src_points"]],
                dst_points=[tuple(p) for p in d["dst_points"]],
                resolution=resolution,
                origin_desc=d.get("origin", ""),
            )

    # ------------------------------------------------------------------

    @staticmethod
    def suggest_default_points(image_width: int, image_height: int) -> dict:
        """为首次标定提供建议的 4 个像素点位置（画面四个角内缩 15%）。"""
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
