"""Tab 1 — 实时监控: directory watch + auto-detect new videos.

Uses ProcessedDB for persistence across restarts: files processed in a
previous session are skipped when monitoring is restarted.
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import shared_state
from alert_mail import AlertMailer
from database import ProcessedDB
from geometry import parse_point_list
from video_processor import AlertEvaluator, RoiMask


def start_monitor(root_dir, conf, nms_iou, frame_skip, roi_text, roi_strategy,
                  poll_interval, stable_time,
                  alert_threshold, alert_w_count, alert_w_area, alert_max_count,
                  email_enabled, email_smtp_server, email_smtp_port,
                  email_sender, email_password, email_receivers,
                  log_callback=None, status_callback=None):
    """Start the monitor. Returns (status_str, log_text).

    log_callback(msg) and status_callback(msg) are called for UI updates.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return "监控目录不存在", _monitor_log_text()

    _stop_monitor_inner()

    shared_state.session_state["monitor_logs"] = []
    def _log(msg):
        shared_state.session_state["monitor_logs"].append(msg)
        if log_callback:
            log_callback(msg)

    _log(f"[{datetime.now().strftime('%H:%M:%S')}] 监控启动: {root_dir}")

    roi = None
    roi_points = parse_point_list(roi_text)
    if roi_points and len(roi_points) >= 3:
        roi = RoiMask(roi_points, (1920, 1080), strategy=roi_strategy)

    stop_event = threading.Event()
    shared_state.session_state["monitor_stop_event"] = stop_event

    alert_eval = AlertEvaluator(
        threshold=alert_threshold, weight_count=alert_w_count,
        weight_area=alert_w_area, max_expected_count=alert_max_count,
    )

    def _loop():
        known = set(ProcessedDB.get_processed_paths("monitor"))

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
                    _wait_stable(fp, stable_time, stop_event)
                    if stop_event.is_set():
                        break
                    known.add(fp)
                    _log(f"[{datetime.now().strftime('%H:%M:%S')}] 发现: {Path(fp).name}")
                    try:
                        result = shared_state.video_processor.process_video(
                            video_path=fp, conf=conf, nms_iou=nms_iou,
                            frame_skip=frame_skip, roi=roi,
                        )
                        cap_tmp = cv2.VideoCapture(fp)
                        frame_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap_tmp.release()

                        csv_dir = shared_state.OUTPUT_DIR / "monitor_results"
                        csv_dir.mkdir(exist_ok=True)
                        csv_path = str(csv_dir / f"{Path(fp).stem}_detections.csv")
                        alert_frames = 0
                        with open(csv_path, "w", encoding="utf-8") as f:
                            f.write("frame_idx,timestamp_sec,label,confidence,x1,y1,x2,y2,alert\n")
                            for fr in result.frame_results:
                                triggered, _, _ = alert_eval.evaluate(
                                    fr.detections, frame_w, frame_h)
                                if triggered:
                                    alert_frames += 1
                                for d in fr.detections:
                                    f.write(
                                        f"{fr.frame_idx},{fr.timestamp_sec},{d.label},"
                                        f"{d.confidence},{d.bbox[0]},{d.bbox[1]},"
                                        f"{d.bbox[2]},{d.bbox[3]},{int(triggered)}\n"
                                    )
                        ProcessedDB.mark_processed(
                            file_path=fp,
                            mode="monitor",
                            detection_count=result.total_detections,
                            frames_processed=result.frames_processed,
                            csv_path=csv_path,
                        )
                        if alert_frames > 0:
                            AlertMailer.try_send(
                                email_enabled, email_smtp_server, email_smtp_port,
                                email_sender, email_password, email_receivers,
                                video_name=Path(fp).name, video_path=fp,
                                frames_processed=result.frames_processed,
                                total_detections=result.total_detections,
                                alert_frames=alert_frames,
                                alert_threshold=alert_threshold,
                                alert_weights=(alert_w_count, alert_w_area),
                                class_counts=result.class_counts,
                            )
                        _log(f"[{datetime.now().strftime('%H:%M:%S')}] OK {Path(fp).name}"
                             f" — {result.total_detections} 目标"
                             f" (告警 {alert_frames}/{result.frames_processed} 帧)")
                    except Exception as e:
                        _log(f"[{datetime.now().strftime('%H:%M:%S')}] FAIL {Path(fp).name}: {e}")
                if len(shared_state.session_state["monitor_logs"]) > 200:
                    del shared_state.session_state["monitor_logs"][:len(shared_state.session_state["monitor_logs"]) - 200]
            except Exception:
                pass
            stop_event.wait(poll_interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    shared_state.session_state["monitor_thread"] = t

    n_processed = len(ProcessedDB.get_processed_paths("monitor"))
    status = f"监控运行中 (轮询间隔 {poll_interval}s, 稳定等待 {stable_time}s, 已处理 {n_processed} 文件)"
    if status_callback:
        status_callback(status)
    return status, _monitor_log_text()


def _wait_stable(fp, stable_time, stop_event):
    waited = 0
    while not stop_event.is_set() and waited < stable_time:
        if time.time() - os.path.getmtime(fp) >= stable_time:
            break
        stop_event.wait(1)
        waited += 1


def _stop_monitor_inner():
    evt = shared_state.session_state.get("monitor_stop_event")
    if evt:
        evt.set()
    shared_state.session_state["monitor_stop_event"] = None


def stop_monitor():
    _stop_monitor_inner()
    shared_state.session_state["monitor_logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] 监控已停止"
    )
    return "监控已停止", _monitor_log_text()


def _monitor_log_text():
    logs = shared_state.session_state.get("monitor_logs", [])
    return "\n".join(logs) if logs else "-- 暂无日志 --"
