"""Scene (Tag) persistence via SQLite: save / load / delete / refresh.

Also holds model-switching logic because scene-load triggers a model change.
"""

import gradio as gr

import shared_state
from database import SceneDB
from file_scanner import SceneConfig
from geometry import format_point_list, parse_point_list


def refresh_scene_list():
    for name in SceneDB.list_all():
        if name not in shared_state.session_state["scenes"]:
            try:
                row = SceneDB.get(name)
                if row:
                    shared_state.session_state["scenes"][name] = SceneConfig.from_dict(row)
            except Exception:
                pass
    return gr.update(choices=list(shared_state.session_state["scenes"].keys()))


def save_scene(name, model, conf, nms_iou, frame_skip, device,
               roi_text, roi_strategy, calib_src_text, calib_dst_text,
               alert_threshold, alert_w_count, alert_w_area, alert_max_count):
    if not name or not name.strip():
        return "请输入场景名称", gr.update(choices=[])
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
    cfg.calib_src = parse_point_list(calib_src_text)
    cfg.calib_dst = parse_point_list(calib_dst_text)
    cfg.alert_threshold = alert_threshold
    cfg.alert_weight_count = alert_w_count
    cfg.alert_weight_area = alert_w_area
    cfg.alert_max_count = alert_max_count
    SceneDB.save(name, cfg.to_dict())
    shared_state.session_state["scenes"][name] = cfg
    choices = list(shared_state.session_state["scenes"].keys())
    suffix = " (已覆盖)" if existing else ""
    return f"场景 '{name}' 已保存{suffix}", gr.update(choices=choices, value=name)


def load_scene(name):
    if not name:
        empty = SceneConfig("_empty_")
        return _scene_to_ui(empty) + ("",)
    cfg = shared_state.session_state["scenes"].get(name)
    if cfg is None:
        row = SceneDB.get(name)
        if row:
            cfg = SceneConfig.from_dict(row)
            shared_state.session_state["scenes"][name] = cfg
        else:
            empty = SceneConfig("_empty_")
            return _scene_to_ui(empty) + ("",)
    model_msg = _switch_model(cfg.model_path, cfg.device)
    return _scene_to_ui(cfg) + (model_msg,)


def delete_scene(name):
    if not name:
        return "请先选择要删除的场景", gr.update()
    SceneDB.delete(name)
    shared_state.session_state["scenes"].pop(name, None)
    choices = list(shared_state.session_state["scenes"].keys())
    return f"场景 '{name}' 已删除", gr.update(choices=choices, value=None)


def _scene_to_ui(cfg: SceneConfig):
    """Expand a SceneConfig into the tuple of UI-component updates."""
    return (
        cfg.model_path,
        cfg.confidence,
        cfg.nms_iou,
        cfg.frame_skip,
        cfg.device,
        format_point_list(cfg.roi_points),
        cfg.roi_strategy,
        format_point_list(cfg.calib_src),
        format_point_list(cfg.calib_dst),
        cfg.alert_threshold,
        cfg.alert_weight_count,
        cfg.alert_weight_area,
        cfg.alert_max_count,
        list(cfg.roi_points) if cfg.roi_points else [],
        f"场景 '{cfg.name}' 已加载",
    )


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
    return _switch_model(model_filename, device)
