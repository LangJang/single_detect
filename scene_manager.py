"""Scene (Tag) persistence via SQLite: save / load / delete / refresh.

Also holds model-switching logic because scene-load triggers a model change.
"""

import shared_state
from database import SceneDB
from scene_config import SceneConfig
from geometry import format_point_list, parse_point_list


def refresh_scene_list():
    """Return list of scene names."""
    for name in SceneDB.list_all():
        if name not in shared_state.session_state["scenes"]:
            try:
                row = SceneDB.get(name)
                if row:
                    shared_state.session_state["scenes"][name] = SceneConfig.from_dict(row)
            except Exception:
                pass
    return list(shared_state.session_state["scenes"].keys())


def save_scene(name, model, conf, nms_iou, frame_skip, device,
               roi_text, roi_strategy,
               calib_method, calib_physical_width, calib_physical_height,
               calib_origin_x, calib_origin_y, calib_dst_text,
               alert_threshold, alert_w_count, alert_w_area, alert_max_count):
    """Save a scene. Returns (status_str, choices_list, value)."""
    if not name or not name.strip():
        return "请输入场景名称", [], None
    name = name.strip()
    existing = name in shared_state.session_state["scenes"]
    cfg = SceneConfig(name)
    cfg.model_path = model
    cfg.confidence = conf
    cfg.nms_iou = nms_iou
    cfg.frame_skip = frame_skip
    cfg.device = device
    cfg.roi_points = parse_point_list(roi_text)
    cfg.roi_strategy = roi_strategy
    cfg.calib_method = calib_method
    cfg.calib_physical_width = calib_physical_width
    cfg.calib_physical_height = calib_physical_height
    cfg.calib_origin_x = calib_origin_x
    cfg.calib_origin_y = calib_origin_y
    cfg.calib_dst_points = parse_point_list(calib_dst_text)
    cfg.alert_threshold = alert_threshold
    cfg.alert_weight_count = alert_w_count
    cfg.alert_weight_area = alert_w_area
    cfg.alert_max_count = alert_max_count
    SceneDB.save(name, cfg.to_dict())
    shared_state.session_state["scenes"][name] = cfg
    choices = list(shared_state.session_state["scenes"].keys())
    suffix = " (已覆盖)" if existing else ""
    return f"场景 '{name}' 已保存{suffix}", choices, name


def load_scene(name):
    """Load a scene. Returns a dict with all scene fields + model_msg."""
    empty = SceneConfig("_empty_")
    if not name:
        return {**_scene_to_dict(empty), "model_msg": ""}

    cfg = shared_state.session_state["scenes"].get(name)
    if cfg is None:
        row = SceneDB.get(name)
        if row:
            cfg = SceneConfig.from_dict(row)
            shared_state.session_state["scenes"][name] = cfg
        else:
            return {**_scene_to_dict(empty), "model_msg": ""}

    model_msg = _switch_model(cfg.model_path, cfg.device)
    return {**_scene_to_dict(cfg), "model_msg": model_msg}


def delete_scene(name):
    """Delete a scene. Returns (status_str, choices_list)."""
    if not name:
        return "请先选择要删除的场景", None
    SceneDB.delete(name)
    shared_state.session_state["scenes"].pop(name, None)
    choices = list(shared_state.session_state["scenes"].keys())
    return f"场景 '{name}' 已删除", choices


def _scene_to_dict(cfg: SceneConfig) -> dict:
    """Convert a SceneConfig to a plain dict of UI values."""
    return {
        "model_path": cfg.model_path,
        "confidence": cfg.confidence,
        "nms_iou": cfg.nms_iou,
        "frame_skip": cfg.frame_skip,
        "device": cfg.device,
        "roi_text": format_point_list(cfg.roi_points),
        "roi_strategy": cfg.roi_strategy,
        "calib_method": cfg.calib_method,
        "calib_physical_width": cfg.calib_physical_width,
        "calib_physical_height": cfg.calib_physical_height,
        "calib_origin_x": cfg.calib_origin_x,
        "calib_origin_y": cfg.calib_origin_y,
        "calib_dst_points": cfg.calib_dst_points,
        "alert_threshold": cfg.alert_threshold,
        "alert_w_count": cfg.alert_weight_count,
        "alert_w_area": cfg.alert_weight_area,
        "alert_max_count": cfg.alert_max_count,
        "roi_points": list(cfg.roi_points) if cfg.roi_points else [],
        "status": f"场景 '{cfg.name}' 已加载",
    }


# ---------------------------------------------------------------------------
# Model switching (triggers on scene load or manual dropdown change)
# ---------------------------------------------------------------------------
def _switch_model(model_filename: str, device: str = "auto"):
    from detector import Detector
    from video_processor import VideoProcessor

    try:
        shared_state.detector = Detector(model_filename, device=device)
        shared_state.video_processor = VideoProcessor(shared_state.detector)
        return f"{model_filename} 已就绪 (设备: {shared_state.detector.device_str})"
    except Exception as e:
        return f"模型加载失败: {e}"


def on_model_change(model_filename, device):
    """Switch model. Returns status string."""
    return _switch_model(model_filename, device)
