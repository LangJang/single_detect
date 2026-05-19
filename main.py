import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from detector import Detector
from video_processor import VideoProcessor, RoiMask, AlertEvaluator
from calibration import CameraCalibration
from file_scanner import FileScanner, ProcessedTracker, SceneConfig

# ==============================================================================
# 1. Constants & Globals
# ==============================================================================
DEFAULT_MODEL = "ep950-loss0.050-val_loss0.055.pth"
MODEL_CHOICES = [
    "ep950-loss0.050-val_loss0.055.pth",
    "ep400-loss0.049-val_loss0.034.pth",
]

detector = Detector(DEFAULT_MODEL)
video_processor = VideoProcessor(detector)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

session_state: dict = {
    "scenes": {},                # {name: SceneConfig}  所有 Tag/场景
    "monitor_stop_event": None,
    "monitor_thread": None,
    "batch_stop_flag": False,
    "monitor_logs": [],
}


# ==============================================================================
# 2. Geometry helpers
# ==============================================================================
def sort_points_convex(points: list) -> list:
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
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def load_video_frame(video_path: str, frame_idx: int = 0):
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


# ==============================================================================
# 3. Scene (Tag) management
# ==============================================================================
def _scene_json_path(name: str) -> Path:
    return OUTPUT_DIR / f"scene_{name}.json"


def refresh_scene_list():
    for f in OUTPUT_DIR.glob("scene_*.json"):
        name = f.stem.replace("scene_", "")
        if name not in session_state["scenes"]:
            try:
                session_state["scenes"][name] = SceneConfig.load(str(f))
            except Exception:
                pass
    return gr.update(choices=list(session_state["scenes"].keys()))


def save_scene(name, model, conf, nms_iou, frame_skip, device,
               roi_text, roi_strategy, calib_src_text, calib_dst_text,
               alert_threshold, alert_w_count, alert_w_area, alert_max_count):
    if not name or not name.strip():
        return "请输入场景名称", gr.update(choices=[])
    name = name.strip()
    existing = name in session_state["scenes"]
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
    cfg.save(str(_scene_json_path(name)))
    session_state["scenes"][name] = cfg
    choices = list(session_state["scenes"].keys())
    suffix = " (已覆盖)" if existing else ""
    return f"场景 '{name}' 已保存{suffix}", gr.update(choices=choices, value=name)


def load_scene(name):
    if not name:
        empty = SceneConfig("_empty_")
        return _scene_to_ui(empty) + ("",)
    cfg = session_state["scenes"].get(name)
    if cfg is None:
        path = _scene_json_path(name)
        if path.exists():
            cfg = SceneConfig.load(str(path))
            session_state["scenes"][name] = cfg
        else:
            empty = SceneConfig("_empty_")
            return _scene_to_ui(empty) + ("",)
    model_msg = _switch_model(cfg.model_path, cfg.device)
    return _scene_to_ui(cfg) + (model_msg,)


def delete_scene(name):
    if not name:
        return "请先选择要删除的场景", gr.update()
    path = _scene_json_path(name)
    if path.exists():
        path.unlink()
    session_state["scenes"].pop(name, None)
    choices = list(session_state["scenes"].keys())
    return f"场景 '{name}' 已删除", gr.update(choices=choices, value=None)


def _scene_to_ui(cfg: SceneConfig):
    """将 SceneConfig 展开为 UI 组件更新 tuple"""
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


# ==============================================================================
# 4. Model switching
# ==============================================================================
def _switch_model(model_filename: str, device: str = "auto"):
    global detector, video_processor
    try:
        detector = Detector(model_filename, device=device)
        video_processor = VideoProcessor(detector)
        return f"{model_filename} 已就绪 (设备: {detector.device_str})"
    except Exception as e:
        return f"模型加载失败: {e}"


def on_model_change(model_filename, device):
    return _switch_model(model_filename, device)


# ==============================================================================
# 5. Preview & ROI interaction
# ==============================================================================
def load_preview_video(video_path: str):
    if not video_path or not os.path.isfile(video_path):
        return None, "请输入有效的视频路径", gr.update(), {}
    frame, info = load_video_frame(video_path, 0)
    if frame is None:
        return None, info, gr.update(), {}
    info_str = (
        f"分辨率: {info['width']}x{info['height']}  |  "
        f"FPS: {info['fps']:.1f}  |  "
        f"总帧数: {info['total_frames']}  |  "
        f"时长: {int(info['duration_sec']//60)}分{int(info['duration_sec']%60)}秒"
    )
    return (
        frame, info_str,
        gr.update(maximum=info["total_frames"] - 1, value=0, interactive=True),
        {"path": video_path, "info": info},
    )


def seek_preview_frame(video_meta: dict, frame_idx: int, roi_points: list):
    if not video_meta or not video_meta.get("path"):
        return None, "请先加载预览视频"
    info = video_meta.get("info", {})
    total = info.get("total_frames", 0)
    frame, result = load_video_frame(video_meta["path"], int(frame_idx))
    if frame is None:
        return None, result
    if roi_points and len(roi_points) >= 3:
        frame = draw_roi_on_frame(frame, roi_points)
    return frame, f"帧 {int(frame_idx)} / {total}"


def handle_image_click(frame_rgb, roi_points, selection_active, evt: gr.SelectData):
    if not selection_active:
        return frame_rgb, format_point_list(roi_points), roi_points, False
    if evt.index is None:
        return frame_rgb, format_point_list(roi_points), roi_points, selection_active
    x, y = evt.index
    new_points = list(roi_points) + [(float(x), float(y))]
    msg = ""
    if len(new_points) == 4:
        sorted_pts = sort_points_convex(new_points)
        if is_quadrilateral_valid(sorted_pts):
            new_points = sorted_pts
            selection_active = False
            msg = "4个点已选定，区域有效"
        else:
            new_points = []
            msg = "区域无效（面积太小或点距离太近），请重新选择"
    elif len(new_points) > 4:
        new_points = [(float(x), float(y))]
        msg = "已重置，请继续点击（第1/4个点）"
    coord_str = format_point_list(new_points)
    out_frame = draw_roi_on_frame(frame_rgb, new_points)
    return out_frame, f"{coord_str}\n{msg}" if msg else coord_str, new_points, selection_active


def start_roi_selection():
    return True, "请在预览图像上依次点击4个顶点（任意顺序均可）", []


def clear_roi_selection(video_meta, frame_idx, roi_points):
    if not video_meta or not video_meta.get("path"):
        return None, "区域已清除", False, []
    frame, _ = seek_preview_frame(video_meta, frame_idx, [])
    return frame, "区域已清除", False, []


def sync_manual_roi(text, video_meta, frame_idx):
    points = parse_point_list(text)
    if not points:
        return None, "", []
    if len(points) >= 3 and video_meta and video_meta.get("path"):
        if len(points) == 4:
            points = sort_points_convex(points)
        frame, _ = seek_preview_frame(video_meta, frame_idx, points)
        return frame, format_point_list(points), points
    return None, format_point_list(points), points


def autofill_calib_from_roi(roi_points):
    if not roi_points or len(roi_points) != 4:
        return "", ""
    return format_point_list(roi_points), "0,0  6,0  6,4  0,4"


# ==============================================================================
# 6. Tab 1: 实时监控
# ==============================================================================
def start_monitor(root_dir, conf, nms_iou, frame_skip, roi_text, roi_strategy,
                  poll_interval, stable_time):
    if not root_dir or not os.path.isdir(root_dir):
        return "监控目录不存在", _monitor_log_text()
    _stop_monitor_inner()
    session_state["monitor_logs"] = []
    session_state["monitor_logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] 监控启动: {root_dir}"
    )
    roi = None
    roi_points = parse_point_list(roi_text)
    if roi_points and len(roi_points) >= 3:
        roi = RoiMask(roi_points, (1920, 1080), strategy=roi_strategy)
    tracker = ProcessedTracker(str(OUTPUT_DIR / "monitor_processed.json"))
    stop_event = threading.Event()
    session_state["monitor_stop_event"] = stop_event

    def _loop():
        scanner = FileScanner(root_dir, tracker)
        known = set()
        for f in scanner.scan(skip_processed=False):
            known.add(f["path"])
        while not stop_event.is_set():
            try:
                current = set()
                root = Path(root_dir)
                if root.exists():
                    for fp in root.rglob("*.mp4"):
                        if fp.is_file():
                            current.add(str(fp))
                new_files = current - known
                for fp in sorted(new_files):
                    waited = 0
                    while not stop_event.is_set() and waited < stable_time:
                        if time.time() - os.path.getmtime(fp) >= stable_time:
                            break
                        stop_event.wait(1)
                        waited += 1
                    if stop_event.is_set():
                        break
                    known.add(fp)
                    session_state["monitor_logs"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 发现: {Path(fp).name}"
                    )
                    try:
                        result = video_processor.process_video(
                            video_path=fp, conf=conf, nms_iou=nms_iou,
                            frame_skip=frame_skip, roi=roi,
                        )
                        tracker.mark_processed(fp, {"detections": result.total_detections})
                        session_state["monitor_logs"].append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] OK {Path(fp).name}"
                            f" — {result.total_detections} 目标"
                        )
                        csv_dir = OUTPUT_DIR / "monitor_results"
                        csv_dir.mkdir(exist_ok=True)
                        csv_path = csv_dir / f"{Path(fp).stem}_detections.csv"
                        with open(csv_path, "w", encoding="utf-8") as f:
                            f.write("frame_idx,timestamp_sec,label,confidence,x1,y1,x2,y2\n")
                            for fr in result.frame_results:
                                for d in fr.detections:
                                    f.write(
                                        f"{fr.frame_idx},{fr.timestamp_sec},{d.label},"
                                        f"{d.confidence},{d.bbox[0]},{d.bbox[1]},"
                                        f"{d.bbox[2]},{d.bbox[3]}\n"
                                    )
                    except Exception as e:
                        session_state["monitor_logs"].append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] FAIL {Path(fp).name}: {e}"
                        )
                if len(session_state["monitor_logs"]) > 200:
                    del session_state["monitor_logs"][:len(session_state["monitor_logs"]) - 200]
            except Exception:
                pass
            stop_event.wait(poll_interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    session_state["monitor_thread"] = t
    return (
        f"监控运行中（轮询间隔 {poll_interval}s，稳定等待 {stable_time}s）",
        _monitor_log_text(),
    )


def _stop_monitor_inner():
    evt = session_state.get("monitor_stop_event")
    if evt:
        evt.set()
    session_state["monitor_stop_event"] = None


def stop_monitor():
    _stop_monitor_inner()
    session_state["monitor_logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] 监控已停止"
    )
    return "监控已停止", _monitor_log_text()


def _monitor_log_text():
    logs = session_state.get("monitor_logs", [])
    return "\n".join(logs) if logs else "-- 暂无日志 --"


# ==============================================================================
# 7. Tab 2: 单视频检测
# ==============================================================================
def scan_video_dir(directory: str):
    if not directory or not os.path.isdir(directory):
        return gr.update(choices=[], value=None), "目录不存在"
    files = []
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        for f in Path(directory).rglob(f"*{ext}"):
            files.append(str(f))
        for f in Path(directory).glob(f"*{ext}"):
            if str(f) not in files:
                files.append(str(f))
    files.sort()
    if not files:
        return gr.update(choices=[], value=None), "目录下没有视频文件"
    return gr.update(choices=files, value=files[0]), f"找到 {len(files)} 个视频文件"


def process_single_video(
    video_path, conf, nms_iou, frame_skip, start_sec, end_sec,
    roi_text, roi_strategy, calib_src_text, calib_dst_text,
    alert_threshold, alert_w_count, alert_w_area, alert_max_count,
    output_annotated, progress=gr.Progress(),
):
    if not video_path or not os.path.isfile(video_path):
        return None, "视频文件不存在", ""

    progress(0, desc="准备中...")

    cap_info = cv2.VideoCapture(video_path)
    frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    roi = None
    roi_points = parse_point_list(roi_text)
    if roi_points and len(roi_points) >= 3:
        roi = RoiMask(roi_points, (frame_w, frame_h), strategy=roi_strategy)

    calib = None
    calib_src = parse_point_list(calib_src_text)
    calib_dst = parse_point_list(calib_dst_text)
    if calib_src and calib_dst and len(calib_src) == 4 and len(calib_dst) == 4:
        calib = CameraCalibration()
        try:
            calib.set_homography(calib_src, calib_dst, (frame_w, frame_h))
        except Exception as e:
            return None, f"标定设置失败: {e}", ""

    alert_eval = AlertEvaluator(
        threshold=alert_threshold, weight_count=alert_w_count,
        weight_area=alert_w_area, max_expected_count=alert_max_count,
    )

    output_video = None
    if output_annotated:
        output_video = str(OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4")

    end = end_sec if end_sec > 0 else None

    def on_progress(current, total):
        progress(current / total, desc=f"检测中 {current}/{total} 帧")

    try:
        progress(0.05, desc="检测中...")
        result = video_processor.process_video(
            video_path=video_path, conf=conf, nms_iou=nms_iou,
            frame_skip=frame_skip, start_sec=start_sec, end_sec=end,
            roi=roi, calibration=calib,
            output_video_path=output_video, progress_callback=on_progress,
        )
    except Exception as e:
        return None, f"处理失败: {e}", ""

    progress(0.90, desc="生成报告...")

    csv_path = str(OUTPUT_DIR / f"{Path(video_path).stem}_detections.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["frame_idx", "timestamp_sec", "label", "confidence",
                   "x1", "y1", "x2", "y2", "world_x", "world_y",
                   "world_w", "world_h", "alert"]
        f.write(",".join(headers) + "\n")
        for fr in result.frame_results:
            triggered, _, _ = alert_eval.evaluate(fr.detections, frame_w, frame_h)
            for d in fr.detections:
                wx = d.world_position[0] if d.world_position else ""
                wy = d.world_position[1] if d.world_position else ""
                ws_w = d.world_size[0] if d.world_size else ""
                ws_h = d.world_size[1] if d.world_size else ""
                f.write(
                    f"{fr.frame_idx},{fr.timestamp_sec},{d.label},{d.confidence},"
                    f"{d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]},"
                    f"{wx},{wy},{ws_w},{ws_h},{int(triggered)}\n"
                )

    alert_frames = sum(
        1 for fr in result.frame_results
        if alert_eval.evaluate(fr.detections, frame_w, frame_h)[0]
    )
    summary = (
        f"处理帧数: {result.frames_processed}\n"
        f"检测目标总数: {result.total_detections}\n"
        f"告警帧数: {alert_frames} / {result.frames_processed}\n"
        f"类别分布: {json.dumps(result.class_counts, ensure_ascii=False)}\n"
        f"\nCSV: {csv_path}"
    )
    if output_video:
        summary += f"\n标注视频: {output_video}"

    preview = None
    for fr in reversed(result.frame_results):
        if fr.detections:
            cap_tmp = cv2.VideoCapture(video_path)
            cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_idx)
            ret, frame = cap_tmp.read()
            cap_tmp.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, _ = detector.detect(frame_rgb, conf=conf)
                if roi:
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    roi.draw(annotated_bgr)
                    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                preview = annotated
            break

    return preview, summary, csv_path


# ==============================================================================
# 8. Tab 3: 批量处理
# ==============================================================================
def scan_directory(root_dir, start_date, end_date, start_time, end_time):
    if not root_dir or not os.path.isdir(root_dir):
        return "目录不存在", gr.update(choices=[], value=[])
    scanner = FileScanner(root_dir)
    try:
        files = scanner.scan_by_date_range(start_date, end_date, start_time, end_time)
    except Exception as e:
        return f"扫描失败: {e}", gr.update(choices=[], value=[])
    if not files:
        return "未找到匹配的视频文件", gr.update(choices=[], value=[])
    choices = [f["path"] for f in files]
    info = f"扫描到 {len(files)} 个视频（{start_date} {start_time} ~ {end_date} {end_time}）"
    return info, gr.update(choices=choices, value=choices)


def process_batch(
    file_list, conf, nms_iou, frame_skip, roi_text, roi_strategy,
    calib_src_text, calib_dst_text, alert_threshold, alert_w_count,
    alert_w_area, alert_max_count, resume, progress=gr.Progress(),
):
    if not file_list:
        return "没有待处理的视频", "", ""

    session_state["batch_stop_flag"] = False
    progress(0, desc="开始批量处理...")
    total = len(file_list)

    roi_points = parse_point_list(roi_text)
    calib_src = parse_point_list(calib_src_text)
    calib_dst = parse_point_list(calib_dst_text)

    need_dims = (
        (roi_points and len(roi_points) >= 3)
        or (calib_src and calib_dst and len(calib_src) == 4 and len(calib_dst) == 4)
    )
    vid_w = vid_h = 0
    if need_dims and file_list:
        cap = cv2.VideoCapture(file_list[0])
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    roi = None
    if roi_points and len(roi_points) >= 3:
        roi = RoiMask(roi_points, (vid_w, vid_h), strategy=roi_strategy)

    calib = None
    if calib_src and calib_dst and len(calib_src) == 4 and len(calib_dst) == 4:
        calib = CameraCalibration()
        try:
            calib.set_homography(calib_src, calib_dst, (vid_w, vid_h))
        except Exception as e:
            return f"标定失败: {e}", "", ""

    tracker = ProcessedTracker(str(OUTPUT_DIR / "batch_processed.json"))
    all_csv_dir = OUTPUT_DIR / "batch_results"
    all_csv_dir.mkdir(exist_ok=True)

    log_lines = []
    success, skipped, failed = 0, 0, 0
    aggregated_counts: dict = {}

    for i, vp in enumerate(file_list):
        if session_state.get("batch_stop_flag"):
            log_lines.append("-- 用户停止 --")
            break
        progress(i / total, desc=f"({i+1}/{total}): {Path(vp).name}")
        if resume and tracker.is_processed(vp):
            skipped += 1
            log_lines.append(f"[跳过] {Path(vp).name}")
            continue
        try:
            result = video_processor.process_video(
                video_path=vp, conf=conf, nms_iou=nms_iou,
                frame_skip=frame_skip, roi=roi, calibration=calib,
            )
            csv_path = str(all_csv_dir / f"{Path(vp).stem}_detections.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("frame_idx,timestamp_sec,label,confidence,x1,y1,x2,y2\n")
                for fr in result.frame_results:
                    for d in fr.detections:
                        f.write(
                            f"{fr.frame_idx},{fr.timestamp_sec},{d.label},"
                            f"{d.confidence},{d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]}\n"
                        )
            tracker.mark_processed(
                vp, {"detections": result.total_detections, "frames": result.frames_processed}
            )
            success += 1
            log_lines.append(f"[OK] {Path(vp).name} — {result.total_detections} 个目标")
            for label, count in result.class_counts.items():
                aggregated_counts[label] = aggregated_counts.get(label, 0) + count
        except Exception as e:
            failed += 1
            log_lines.append(f"[FAIL] {Path(vp).name} — {e}")

    summary = (
        f"成功: {success}    跳过: {skipped}    失败: {failed}\n"
        f"累计检测目标: {sum(aggregated_counts.values())}\n"
        f"类别分布: {json.dumps(aggregated_counts, ensure_ascii=False)}\n"
        f"结果目录: {all_csv_dir}"
    )
    progress(1.0, desc="完成")
    return summary, "\n".join(log_lines), str(all_csv_dir)


def stop_batch():
    session_state["batch_stop_flag"] = True
    return "已发送停止信号..."


# ==============================================================================
# 9. UI
# ==============================================================================
with gr.Blocks(title="YOLO 目标检测系统") as app:

    gr.Markdown("# YOLO 目标检测系统")

    # ══════════════════════════════════════════════════════════════════════════
    # Scene (Tag) Management — 场景即 Tag，每个 Tag 拥有独立参数
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Group():
        gr.Markdown("## 场景 (Tag) 管理")
        with gr.Row():
            scene_list = gr.Dropdown(
                label="已保存场景", choices=[], interactive=True, scale=3,
            )
            scene_name = gr.Textbox(
                label="场景名称", placeholder="输入名称以保存...", scale=2,
            )
            scene_load_btn = gr.Button("加载", variant="secondary", scale=1)
            scene_save_btn = gr.Button("保存", variant="primary", scale=1)
        with gr.Row():
            scene_status = gr.Textbox(label="", interactive=False, container=False, scale=3)
            scene_refresh_btn = gr.Button("刷新列表", variant="secondary", scale=1)
            scene_delete_btn = gr.Button("删除", variant="stop", scale=1)

    # ══════════════════════════════════════════════════════════════════════════
    # Detection Parameters — 随当前 Tag 变化
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Group():
        gr.Markdown("## 检测参数")
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=MODEL_CHOICES, value=DEFAULT_MODEL,
                label="检测模型", scale=3,
            )
            device_selector = gr.Dropdown(
                choices=["auto", "cuda", "cpu"], value="auto",
                label="计算设备", scale=1,
            )
            model_status = gr.Textbox(
                label="状态", value=f"{DEFAULT_MODEL} 已就绪",
                interactive=False, scale=2,
            )
        with gr.Row():
            conf_slider = gr.Slider(0.1, 0.9, 0.25, 0.05, label="置信度阈值")
            nms_iou_slider = gr.Slider(0.1, 0.9, 0.3, 0.05, label="NMS IoU")
            frame_skip_slider = gr.Slider(1, 60, 10, 1, label="帧采样间隔")

    # ══════════════════════════════════════════════════════════════════════════
    # ROI & Calibration
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("ROI 区域 & 相机标定", open=False):
        with gr.Row():
            preview_video_path = gr.Textbox(
                label="预览视频", scale=4, placeholder="输入视频文件路径...",
            )
            load_preview_btn = gr.Button("加载预览", variant="secondary", scale=1)
            frame_position_slider = gr.Slider(
                0, 100, 0, 1, label="跳转到帧", scale=2, interactive=False,
            )
            frame_info_display = gr.Textbox(
                label="帧信息", interactive=False, scale=2, placeholder="请先加载预览视频",
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                preview_image = gr.Image(
                    label="点击图像选择 ROI 区域",
                    type="numpy", image_mode="RGB", height=400,
                )
            with gr.Column(scale=2):
                with gr.Row():
                    roi_start_btn = gr.Button("开始选择区域", variant="secondary", scale=1)
                    roi_clear_btn = gr.Button("清除区域", variant="secondary", scale=1)
                roi_coords_display = gr.Textbox(
                    label="ROI 顶点坐标", lines=3, interactive=False,
                    placeholder="点击「开始选择区域」后在图像上依次点击4个点",
                )
                roi_strategy = gr.Dropdown(
                    choices=["centroid", "overlap", "full"], value="centroid",
                    label="过滤策略",
                )
                roi_manual_text = gr.Textbox(
                    label="或手动输入坐标", lines=2,
                    placeholder="x1,y1  x2,y2  x3,y3  x4,y4",
                )

                with gr.Accordion("相机标定 · Homography 4点法", open=False):
                    calib_src = gr.Textbox(
                        label="像素坐标（4个地面点）", lines=3,
                        placeholder="例如: 200,500  800,500  800,100  200,100",
                    )
                    calib_dst = gr.Textbox(
                        label="世界坐标（4点，单位：米）", lines=3,
                        placeholder="例如: 0,0  6,0  6,4  0,4",
                    )
                    calib_autofill_btn = gr.Button(
                        "从 ROI 自动填充像素坐标", variant="secondary", size="sm",
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # Alert Settings
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Accordion("告警设置", open=False):
        with gr.Row():
            alert_threshold = gr.Slider(0.1, 1.0, 0.5, 0.05, label="告警阈值")
            alert_max_count = gr.Number(10, label="期望最大目标数", precision=0)
        with gr.Row():
            alert_w_count = gr.Slider(0.0, 1.0, 0.4, 0.1, label="数量权重")
            alert_w_area = gr.Slider(0.0, 1.0, 0.6, 0.1, label="面积权重")

    # --- Hidden States ---
    roi_points_state = gr.State([])
    roi_selection_active = gr.State(False)
    preview_video_meta = gr.State({})

    # ══════════════════════════════════════════════════════════════════════════
    # Task Tabs — 在当前 Tag 参数下执行任务
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tabs():
        # ── Tab 1: 实时监控 ──────────────────────────────────────────────
        with gr.TabItem("实时监控"):
            with gr.Group():
                gr.Markdown("### 监控执行")
                with gr.Row():
                    monitor_dir = gr.Textbox(
                        label="监控目录", scale=4, value=str(Path(__file__).parent),
                    )
                    poll_interval = gr.Number(
                        10, label="轮询间隔 (秒)", precision=0, minimum=1, maximum=300, scale=1,
                    )
                    stable_time = gr.Number(
                        5, label="文件稳定等待 (秒)", precision=0, minimum=1, maximum=60, scale=1,
                    )
                with gr.Row():
                    monitor_start_btn = gr.Button("启动监控", variant="primary", scale=2)
                    monitor_stop_btn = gr.Button("停止监控", variant="stop", scale=1)
                    monitor_refresh_btn = gr.Button("刷新日志", variant="secondary", scale=1)
                monitor_status = gr.Textbox(label="状态", interactive=False)
                monitor_log = gr.Textbox(
                    label="日志", lines=8, interactive=False, show_copy_button=True,
                )

        # ── Tab 2: 单视频检测 ────────────────────────────────────────────
        with gr.TabItem("单视频检测"):
            with gr.Group():
                gr.Markdown("### 视频处理")
                with gr.Row():
                    video_dir = gr.Textbox(
                        label="视频所在目录", scale=3, value=str(Path(__file__).parent),
                    )
                    scan_btn = gr.Button("扫描", variant="secondary", scale=1)
                with gr.Row():
                    video_selector = gr.Dropdown(
                        label="选择视频", choices=[], interactive=True, scale=4,
                        allow_custom_value=True,
                    )
                    scan_info = gr.Textbox(label="", interactive=False, container=False)
                with gr.Row():
                    start_sec = gr.Number(0, label="起始 (秒)", precision=1)
                    end_sec = gr.Number(0, label="结束 (秒, 0=末尾)", precision=1)
                    output_annotated = gr.Checkbox(True, label="输出标注视频")
                    process_btn = gr.Button("开始处理", variant="primary")
                with gr.Row():
                    with gr.Column(scale=3):
                        result_preview = gr.Image(
                            label="检测结果预览", type="numpy", image_mode="RGB",
                        )
                    with gr.Column(scale=2):
                        result_summary = gr.Textbox(
                            label="处理报告", lines=12, show_copy_button=True,
                        )
                        result_csv = gr.Textbox(label="输出文件", interactive=False)

        # ── Tab 3: 批量处理 ──────────────────────────────────────────────
        with gr.TabItem("批量处理"):
            with gr.Group():
                gr.Markdown("### 批量执行")
                with gr.Row():
                    batch_dir = gr.Textbox(
                        label="视频根目录", value=str(Path(__file__).parent),
                    )
                with gr.Row():
                    batch_start_date = gr.Textbox(
                        label="开始日期", value=datetime.now().strftime("%Y-%m-%d"),
                        placeholder="YYYY-MM-DD",
                    )
                    batch_end_date = gr.Textbox(
                        label="结束日期", value=datetime.now().strftime("%Y-%m-%d"),
                        placeholder="YYYY-MM-DD",
                    )
                    batch_start_time = gr.Textbox(label="开始时间", value="00:00:00")
                    batch_end_time = gr.Textbox(label="结束时间", value="23:59:59")
                    batch_scan_btn = gr.Button("扫描", variant="secondary")
                batch_scan_info = gr.Textbox(label="", interactive=False, container=False)
                batch_file_list = gr.CheckboxGroup(label="待处理文件", choices=[])
                with gr.Row():
                    batch_resume = gr.Checkbox(True, label="断点续跑")
                    batch_process_btn = gr.Button("开始批量处理", variant="primary")
                    batch_stop_btn = gr.Button("停止", variant="stop")
                with gr.Row():
                    batch_summary = gr.Textbox(
                        label="汇总", lines=5, show_copy_button=True, scale=1,
                    )
                    batch_log = gr.Textbox(label="处理日志", lines=8, scale=2)
                batch_output_dir = gr.Textbox(label="结果目录", interactive=False)

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: Scene Management
    # ══════════════════════════════════════════════════════════════════════════
    scene_save_btn.click(
        save_scene,
        inputs=[
            scene_name, model_selector, conf_slider, nms_iou_slider,
            frame_skip_slider, device_selector,
            roi_coords_display, roi_strategy, calib_src, calib_dst,
            alert_threshold, alert_w_count, alert_w_area, alert_max_count,
        ],
        outputs=[scene_status, scene_list],
    )
    scene_load_btn.click(
        load_scene, inputs=[scene_list],
        outputs=[
            model_selector, conf_slider, nms_iou_slider, frame_skip_slider,
            device_selector,
            roi_coords_display, roi_strategy,
            calib_src, calib_dst,
            alert_threshold, alert_w_count, alert_w_area, alert_max_count,
            roi_points_state, scene_status, model_status,
        ],
    )
    scene_refresh_btn.click(refresh_scene_list, outputs=[scene_list])
    scene_delete_btn.click(
        delete_scene, inputs=[scene_list],
        outputs=[scene_status, scene_list],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: Model
    # ══════════════════════════════════════════════════════════════════════════
    model_selector.change(
        on_model_change, inputs=[model_selector, device_selector],
        outputs=[model_status],
    )
    device_selector.change(
        on_model_change, inputs=[model_selector, device_selector],
        outputs=[model_status],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: ROI & Preview
    # ══════════════════════════════════════════════════════════════════════════
    load_preview_btn.click(
        load_preview_video, inputs=[preview_video_path],
        outputs=[preview_image, frame_info_display, frame_position_slider, preview_video_meta],
    )
    frame_position_slider.change(
        seek_preview_frame,
        inputs=[preview_video_meta, frame_position_slider, roi_points_state],
        outputs=[preview_image, frame_info_display],
    )
    preview_image.select(
        handle_image_click,
        inputs=[preview_image, roi_points_state, roi_selection_active],
        outputs=[preview_image, roi_coords_display, roi_points_state, roi_selection_active],
    )
    roi_start_btn.click(
        start_roi_selection,
        outputs=[roi_selection_active, roi_coords_display, roi_points_state],
    )
    roi_clear_btn.click(
        clear_roi_selection,
        inputs=[preview_video_meta, frame_position_slider, roi_points_state],
        outputs=[preview_image, roi_coords_display, roi_selection_active, roi_points_state],
    )
    roi_manual_text.change(
        sync_manual_roi,
        inputs=[roi_manual_text, preview_video_meta, frame_position_slider],
        outputs=[preview_image, roi_coords_display, roi_points_state],
    )
    calib_autofill_btn.click(
        autofill_calib_from_roi, inputs=[roi_points_state],
        outputs=[calib_src, calib_dst],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: Tab 1 — 实时监控
    # ══════════════════════════════════════════════════════════════════════════
    monitor_start_btn.click(
        start_monitor,
        inputs=[monitor_dir, conf_slider, nms_iou_slider, frame_skip_slider,
                roi_coords_display, roi_strategy, poll_interval, stable_time],
        outputs=[monitor_status, monitor_log],
    )
    monitor_stop_btn.click(stop_monitor, outputs=[monitor_status, monitor_log])
    monitor_refresh_btn.click(lambda: _monitor_log_text(), outputs=[monitor_log])

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: Tab 2 — 单视频检测
    # ══════════════════════════════════════════════════════════════════════════
    scan_btn.click(
        scan_video_dir, inputs=[video_dir],
        outputs=[video_selector, scan_info],
    )
    process_btn.click(
        process_single_video,
        inputs=[
            video_selector, conf_slider, nms_iou_slider, frame_skip_slider,
            start_sec, end_sec,
            roi_coords_display, roi_strategy, calib_src, calib_dst,
            alert_threshold, alert_w_count, alert_w_area, alert_max_count,
            output_annotated,
        ],
        outputs=[result_preview, result_summary, result_csv],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Event Bindings: Tab 3 — 批量处理
    # ══════════════════════════════════════════════════════════════════════════
    batch_scan_btn.click(
        scan_directory,
        inputs=[batch_dir, batch_start_date, batch_end_date,
                batch_start_time, batch_end_time],
        outputs=[batch_scan_info, batch_file_list],
    )
    batch_process_btn.click(
        process_batch,
        inputs=[
            batch_file_list, conf_slider, nms_iou_slider, frame_skip_slider,
            roi_coords_display, roi_strategy, calib_src, calib_dst,
            alert_threshold, alert_w_count, alert_w_area, alert_max_count,
            batch_resume,
        ],
        outputs=[batch_summary, batch_log, batch_output_dir],
    )
    batch_stop_btn.click(stop_batch, outputs=[batch_summary])

    # ══════════════════════════════════════════════════════════════════════════
    # App Load
    # ══════════════════════════════════════════════════════════════════════════
    app.load(refresh_scene_list, outputs=[scene_list])


# ==============================================================================
# 10. Launch
# ==============================================================================
if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
