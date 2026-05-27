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
        self.calib_method: str = "diagonal"        # "diagonal" | "four_point"
        self.calib_physical_width: float = 0.0     # 对角法：区域宽度(米)
        self.calib_physical_height: float = 0.0    # 对角法：区域高度(米)
        self.calib_origin_x: float = 0.0           # 对角法：左上角原点 X(米)
        self.calib_origin_y: float = 0.0           # 对角法：左上角原点 Y(米)
        self.calib_dst_points: list[tuple] = []    # 四点法：4个世界坐标
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
            "calib_method": self.calib_method,
            "calib_physical_width": self.calib_physical_width,
            "calib_physical_height": self.calib_physical_height,
            "calib_origin_x": self.calib_origin_x,
            "calib_origin_y": self.calib_origin_y,
            "calib_dst_points": self.calib_dst_points,
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
        s.calib_method = d.get("calib_method", "diagonal")
        s.calib_physical_width = d.get("calib_physical_width", 0.0)
        s.calib_physical_height = d.get("calib_physical_height", 0.0)
        s.calib_origin_x = d.get("calib_origin_x", 0.0)
        s.calib_origin_y = d.get("calib_origin_y", 0.0)
        s.calib_dst_points = d.get("calib_dst_points", [])
        s.alert_threshold = d.get("alert_threshold", 0.5)
        s.alert_weight_count = d.get("alert_weight_count", 0.4)
        s.alert_weight_area = d.get("alert_weight_area", 0.6)
        s.alert_max_count = d.get("alert_max_count", 10)
        return s
