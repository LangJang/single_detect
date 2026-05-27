"""QThread workers for long-running detection tasks."""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

import shared_state
from geometry import parse_point_list
from video_processor import AlertEvaluator, RoiMask


# ---------------------------------------------------------------------------
# Monitor Worker
# ---------------------------------------------------------------------------

class MonitorWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, root_dir, conf, nms_iou, frame_skip, roi_text, roi_strategy,
                 calib_method, calib_physical_width, calib_physical_height,
                 calib_origin_x, calib_origin_y, calib_dst_text,
                 poll_interval, stable_time,
                 alert_threshold, alert_w_count, alert_w_area, alert_max_count,
                 email_enabled, email_smtp_server, email_smtp_port,
                 email_sender, email_password, email_receivers,
                 parent=None):
        super().__init__(parent)
        self._params = {
            "root_dir": root_dir, "conf": conf, "nms_iou": nms_iou,
            "frame_skip": frame_skip, "roi_text": roi_text,
            "roi_strategy": roi_strategy,
            "calib_method": calib_method,
            "calib_physical_width": calib_physical_width,
            "calib_physical_height": calib_physical_height,
            "calib_origin_x": calib_origin_x,
            "calib_origin_y": calib_origin_y,
            "calib_dst_text": calib_dst_text,
            "poll_interval": poll_interval,
            "stable_time": stable_time, "alert_threshold": alert_threshold,
            "alert_w_count": alert_w_count, "alert_w_area": alert_w_area,
            "alert_max_count": alert_max_count, "email_enabled": email_enabled,
            "email_smtp_server": email_smtp_server,
            "email_smtp_port": email_smtp_port, "email_sender": email_sender,
            "email_password": email_password, "email_receivers": email_receivers,
        }
        self._stop_event: threading.Event | None = None

    def run(self):
        p = self._params
        root_dir = p["root_dir"]
        if not root_dir or not os.path.isdir(root_dir):
            self.status_signal.emit("监控目录不存在")
            return

        # Build ROI mask
        roi = None
        roi_points = parse_point_list(p["roi_text"])
        if roi_points and len(roi_points) >= 3:
            roi = RoiMask(roi_points, (1920, 1080), strategy=p["roi_strategy"])

        from calibration import CameraCalibration

        calib = None
        if roi_points and len(roi_points) == 4:
            if p["calib_method"] == "diagonal":
                if p["calib_physical_width"] > 0 and p["calib_physical_height"] > 0:
                    calib = CameraCalibration()
                    try:
                        calib.set_from_roi(
                            roi_points, p["calib_physical_width"],
                            p["calib_physical_height"], (1920, 1080),
                            p["calib_origin_x"], p["calib_origin_y"],
                        )
                    except Exception:
                        pass
            else:  # four_point
                dst = parse_point_list(p["calib_dst_text"])
                if dst and len(dst) == 4:
                    calib = CameraCalibration()
                    try:
                        calib.set_homography(roi_points, dst, (1920, 1080))
                    except Exception:
                        pass

        stop_event = threading.Event()
        self._stop_event = stop_event
        shared_state.session_state["monitor_stop_event"] = stop_event

        alert_eval = AlertEvaluator(
            threshold=p["alert_threshold"],
            weight_count=p["alert_w_count"],
            weight_area=p["alert_w_area"],
            max_expected_count=p["alert_max_count"],
        )

        def _log(msg):
            self.log_signal.emit(msg)
            shared_state.session_state.setdefault("monitor_logs", []).append(msg)

        _log(f"[{datetime.now().strftime('%H:%M:%S')}] 监控启动: {root_dir}")

        from alert_mail import AlertMailer
        from database import ProcessedDB

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
                    self._wait_stable(fp, p["stable_time"], stop_event)
                    if stop_event.is_set():
                        break
                    known.add(fp)
                    fname = Path(fp).name
                    _log(f"[{datetime.now().strftime('%H:%M:%S')}] 发现: {fname}")
                    try:
                        result = shared_state.video_processor.process_video(
                            video_path=fp, conf=p["conf"], nms_iou=p["nms_iou"],
                            frame_skip=p["frame_skip"], roi=roi, calibration=calib,
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
                            file_path=fp, mode="monitor",
                            detection_count=result.total_detections,
                            frames_processed=result.frames_processed,
                            csv_path=csv_path,
                        )
                        if alert_frames > 0:
                            AlertMailer.try_send(
                                p["email_enabled"], p["email_smtp_server"],
                                p["email_smtp_port"], p["email_sender"],
                                p["email_password"], p["email_receivers"],
                                video_name=fname, video_path=fp,
                                frames_processed=result.frames_processed,
                                total_detections=result.total_detections,
                                alert_frames=alert_frames,
                                alert_threshold=p["alert_threshold"],
                                alert_weights=(p["alert_w_count"], p["alert_w_area"]),
                                class_counts=result.class_counts,
                            )
                        _log(f"[{datetime.now().strftime('%H:%M:%S')}] OK {fname}"
                             f" — {result.total_detections} 目标"
                             f" (告警 {alert_frames}/{result.frames_processed} 帧)")
                    except Exception as e:
                        _log(f"[{datetime.now().strftime('%H:%M:%S')}] FAIL {fname}: {e}")
                logs = shared_state.session_state.get("monitor_logs", [])
                if len(logs) > 200:
                    del logs[:len(logs) - 200]
            except Exception:
                pass
            stop_event.wait(p["poll_interval"])

        _log(f"[{datetime.now().strftime('%H:%M:%S')}] 监控已停止")

    def stop(self):
        if self._stop_event:
            self._stop_event.set()

    def _wait_stable(self, fp, stable_time, stop_event):
        waited = 0
        while not stop_event.is_set() and waited < stable_time:
            if time.time() - os.path.getmtime(fp) >= stable_time:
                break
            stop_event.wait(1)
            waited += 1


# ---------------------------------------------------------------------------
# Single Video Worker
# ---------------------------------------------------------------------------

class SingleVideoWorker(QThread):
    progress_signal = Signal(int, int, str)          # current, total, desc
    finished_signal = Signal(object, str, str)        # preview, summary, csv_path
    error_signal = Signal(str)

    def __init__(self, video_path, conf, nms_iou, frame_skip, start_sec, end_sec,
                 roi_text, roi_strategy,
                 calib_method, calib_physical_width, calib_physical_height,
                 calib_origin_x, calib_origin_y, calib_dst_text,
                 alert_threshold, alert_w_count, alert_w_area, alert_max_count,
                 output_annotated,
                 email_enabled, email_smtp_server, email_smtp_port,
                 email_sender, email_password, email_receivers,
                 parent=None):
        super().__init__(parent)
        self._params = {
            "video_path": video_path, "conf": conf, "nms_iou": nms_iou,
            "frame_skip": frame_skip, "start_sec": start_sec, "end_sec": end_sec,
            "roi_text": roi_text, "roi_strategy": roi_strategy,
            "calib_method": calib_method,
            "calib_physical_width": calib_physical_width,
            "calib_physical_height": calib_physical_height,
            "calib_origin_x": calib_origin_x,
            "calib_origin_y": calib_origin_y,
            "calib_dst_text": calib_dst_text,
            "alert_threshold": alert_threshold, "alert_w_count": alert_w_count,
            "alert_w_area": alert_w_area, "alert_max_count": alert_max_count,
            "output_annotated": output_annotated,
            "email_enabled": email_enabled, "email_smtp_server": email_smtp_server,
            "email_smtp_port": email_smtp_port, "email_sender": email_sender,
            "email_password": email_password, "email_receivers": email_receivers,
        }

    def run(self):
        p = self._params
        import json
        from calibration import CameraCalibration
        from database import ProcessedDB
        from alert_mail import AlertMailer

        video_path = p["video_path"]
        if not video_path or not os.path.isfile(video_path):
            self.error_signal.emit("视频文件不存在")
            return

        self.progress_signal.emit(0, 100, "准备中...")

        cap_info = cv2.VideoCapture(video_path)
        frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()

        roi = None
        roi_points = parse_point_list(p["roi_text"])
        if roi_points and len(roi_points) >= 3:
            roi = RoiMask(roi_points, (frame_w, frame_h), strategy=p["roi_strategy"])

        calib = None
        if roi_points and len(roi_points) == 4:
            if p["calib_method"] == "diagonal":
                if p["calib_physical_width"] > 0 and p["calib_physical_height"] > 0:
                    calib = CameraCalibration()
                    try:
                        calib.set_from_roi(
                            roi_points, p["calib_physical_width"],
                            p["calib_physical_height"], (frame_w, frame_h),
                            p["calib_origin_x"], p["calib_origin_y"],
                        )
                    except Exception as e:
                        self.error_signal.emit(f"标定设置失败: {e}")
                        return
            else:  # four_point
                dst = parse_point_list(p["calib_dst_text"])
                if dst and len(dst) == 4:
                    calib = CameraCalibration()
                    try:
                        calib.set_homography(roi_points, dst, (frame_w, frame_h))
                    except Exception as e:
                        self.error_signal.emit(f"标定设置失败: {e}")
                        return

        alert_eval = AlertEvaluator(
            threshold=p["alert_threshold"], weight_count=p["alert_w_count"],
            weight_area=p["alert_w_area"], max_expected_count=p["alert_max_count"],
        )

        output_video = None
        if p["output_annotated"]:
            output_video = str(
                shared_state.OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4")

        end = p["end_sec"] if p["end_sec"] > 0 else None

        def on_progress(current, total, desc=""):
            self.progress_signal.emit(current, total, desc)

        try:
            self.progress_signal.emit(5, 100, "检测中...")
            result = shared_state.video_processor.process_video(
                video_path=video_path, conf=p["conf"], nms_iou=p["nms_iou"],
                frame_skip=p["frame_skip"], start_sec=p["start_sec"], end_sec=end,
                roi=roi, calibration=calib,
                output_video_path=output_video,
                progress_callback=on_progress,
            )
        except Exception as e:
            self.error_signal.emit(f"处理失败: {e}")
            return

        self.progress_signal.emit(90, 100, "生成报告...")

        csv_path = str(
            shared_state.OUTPUT_DIR / f"{Path(video_path).stem}_detections.csv")
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

        ProcessedDB.mark_processed(
            file_path=video_path, mode="single",
            detection_count=result.total_detections,
            frames_processed=result.frames_processed, csv_path=csv_path,
        )

        if alert_frames > 0:
            AlertMailer.try_send(
                p["email_enabled"], p["email_smtp_server"], p["email_smtp_port"],
                p["email_sender"], p["email_password"], p["email_receivers"],
                video_name=Path(video_path).name, video_path=video_path,
                frames_processed=result.frames_processed,
                total_detections=result.total_detections,
                alert_frames=alert_frames,
                alert_threshold=p["alert_threshold"],
                alert_weights=(p["alert_w_count"], p["alert_w_area"]),
                class_counts=result.class_counts,
            )

        # Extract preview frame with detections
        preview = None
        for fr in reversed(result.frame_results):
            if fr.detections:
                cap_tmp = cv2.VideoCapture(video_path)
                cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_idx)
                ret, frame = cap_tmp.read()
                cap_tmp.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated, _ = shared_state.detector.detect(
                        frame_rgb, conf=p["conf"])
                    if roi:
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        roi.draw(annotated_bgr)
                        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    preview = annotated
                break

        self.progress_signal.emit(100, 100, "完成")
        self.finished_signal.emit(preview, summary, csv_path)


# ---------------------------------------------------------------------------
# Batch Worker
# ---------------------------------------------------------------------------

class BatchWorker(QThread):
    progress_signal = Signal(int, int, str)          # current, total, desc
    log_signal = Signal(str)
    finished_signal = Signal(str, str, str)          # summary, log_text, output_dir

    def __init__(self, file_list, conf, nms_iou, frame_skip, roi_text, roi_strategy,
                 calib_method, calib_physical_width, calib_physical_height,
                 calib_origin_x, calib_origin_y, calib_dst_text,
                 alert_threshold, alert_w_count, alert_w_area, alert_max_count, resume,
                 email_enabled, email_smtp_server, email_smtp_port,
                 email_sender, email_password, email_receivers,
                 parent=None):
        super().__init__(parent)
        self._params = {
            "file_list": file_list, "conf": conf, "nms_iou": nms_iou,
            "frame_skip": frame_skip, "roi_text": roi_text,
            "roi_strategy": roi_strategy,
            "calib_method": calib_method,
            "calib_physical_width": calib_physical_width,
            "calib_physical_height": calib_physical_height,
            "calib_origin_x": calib_origin_x,
            "calib_origin_y": calib_origin_y,
            "calib_dst_text": calib_dst_text,
            "alert_threshold": alert_threshold, "alert_w_count": alert_w_count,
            "alert_w_area": alert_w_area, "alert_max_count": alert_max_count,
            "resume": resume,
            "email_enabled": email_enabled, "email_smtp_server": email_smtp_server,
            "email_smtp_port": email_smtp_port, "email_sender": email_sender,
            "email_password": email_password, "email_receivers": email_receivers,
        }
        shared_state.session_state["batch_stop_flag"] = False

    def run(self):
        p = self._params
        import json
        from calibration import CameraCalibration
        from database import ProcessedDB
        from alert_mail import AlertMailer

        file_list = p["file_list"]
        if not file_list:
            self.finished_signal.emit("没有待处理的视频", "", "")
            return

        self.progress_signal.emit(0, len(file_list), "开始批量处理...")

        roi_points = parse_point_list(p["roi_text"])

        need_calib = (
            (roi_points and len(roi_points) == 4)
            and (
                (p["calib_method"] == "diagonal"
                 and p["calib_physical_width"] > 0 and p["calib_physical_height"] > 0)
                or (p["calib_method"] == "four_point"
                    and p["calib_dst_text"] and len(parse_point_list(p["calib_dst_text"])) == 4)
            )
        )
        need_dims = (
            (roi_points and len(roi_points) >= 3) or need_calib
        )
        vid_w = vid_h = 0
        if need_dims and file_list:
            cap = cv2.VideoCapture(file_list[0])
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        roi = None
        if roi_points and len(roi_points) >= 3:
            roi = RoiMask(roi_points, (vid_w, vid_h), strategy=p["roi_strategy"])

        calib = None
        if roi_points and len(roi_points) == 4:
            if p["calib_method"] == "diagonal":
                if p["calib_physical_width"] > 0 and p["calib_physical_height"] > 0:
                    calib = CameraCalibration()
                    try:
                        calib.set_from_roi(
                            roi_points, p["calib_physical_width"],
                            p["calib_physical_height"], (vid_w, vid_h),
                            p["calib_origin_x"], p["calib_origin_y"],
                        )
                    except Exception as e:
                        self.finished_signal.emit(f"标定失败: {e}", "", "")
                        return
            else:  # four_point
                dst = parse_point_list(p["calib_dst_text"])
                if dst and len(dst) == 4:
                    calib = CameraCalibration()
                    try:
                        calib.set_homography(roi_points, dst, (vid_w, vid_h))
                    except Exception as e:
                        self.finished_signal.emit(f"标定失败: {e}", "", "")
                        return

        alert_eval = AlertEvaluator(
            threshold=p["alert_threshold"], weight_count=p["alert_w_count"],
            weight_area=p["alert_w_area"], max_expected_count=p["alert_max_count"],
        )

        all_csv_dir = shared_state.OUTPUT_DIR / "batch_results"
        all_csv_dir.mkdir(exist_ok=True)

        log_lines = []
        success, skipped, failed = 0, 0, 0
        alert_total = 0
        aggregated_counts: dict = {}

        total = len(file_list)
        for i, vp in enumerate(file_list):
            if shared_state.session_state.get("batch_stop_flag"):
                log_lines.append("-- 用户停止 --")
                break
            self.progress_signal.emit(i, total, f"({i+1}/{total}): {Path(vp).name}")
            if p["resume"] and ProcessedDB.is_processed(vp, "batch"):
                skipped += 1
                log_lines.append(f"[跳过] {Path(vp).name}")
                self.log_signal.emit(f"[跳过] {Path(vp).name}")
                continue
            try:
                result = shared_state.video_processor.process_video(
                    video_path=vp, conf=p["conf"], nms_iou=p["nms_iou"],
                    frame_skip=p["frame_skip"], roi=roi, calibration=calib,
                )
                csv_path = str(
                    all_csv_dir / f"{Path(vp).stem}_detections.csv")
                cap_tmp = cv2.VideoCapture(vp)
                fw = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_tmp.release()

                alert_frames = 0
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("frame_idx,timestamp_sec,label,confidence,x1,y1,x2,y2,alert\n")
                    for fr in result.frame_results:
                        triggered, _, _ = alert_eval.evaluate(fr.detections, fw, fh)
                        if triggered:
                            alert_frames += 1
                        for d in fr.detections:
                            f.write(
                                f"{fr.frame_idx},{fr.timestamp_sec},{d.label},"
                                f"{d.confidence},{d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]},"
                                f"{int(triggered)}\n"
                            )
                ProcessedDB.mark_processed(
                    file_path=vp, mode="batch",
                    detection_count=result.total_detections,
                    frames_processed=result.frames_processed, csv_path=csv_path,
                )
                success += 1
                alert_total += alert_frames
                if alert_frames > 0:
                    AlertMailer.try_send(
                        p["email_enabled"], p["email_smtp_server"],
                        p["email_smtp_port"], p["email_sender"],
                        p["email_password"], p["email_receivers"],
                        video_name=Path(vp).name, video_path=vp,
                        frames_processed=result.frames_processed,
                        total_detections=result.total_detections,
                        alert_frames=alert_frames,
                        alert_threshold=p["alert_threshold"],
                        alert_weights=(p["alert_w_count"], p["alert_w_area"]),
                        class_counts=result.class_counts,
                    )
                msg = (f"[OK] {Path(vp).name} — {result.total_detections} 目标"
                       f" (告警 {alert_frames}/{result.frames_processed} 帧)")
                log_lines.append(msg)
                self.log_signal.emit(msg)
                for label, count in result.class_counts.items():
                    aggregated_counts[label] = aggregated_counts.get(label, 0) + count
            except Exception as e:
                failed += 1
                msg = f"[FAIL] {Path(vp).name} — {e}"
                log_lines.append(msg)
                self.log_signal.emit(msg)

        summary = (
            f"成功: {success}    跳过: {skipped}    失败: {failed}\n"
            f"累计检测目标: {sum(aggregated_counts.values())}\n"
            f"累计告警帧: {alert_total}\n"
            f"类别分布: {json.dumps(aggregated_counts, ensure_ascii=False)}\n"
            f"结果目录: {all_csv_dir}"
        )
        self.progress_signal.emit(total, total, "完成")
        self.finished_signal.emit(summary, "\n".join(log_lines), str(all_csv_dir))

    def stop(self):
        shared_state.session_state["batch_stop_flag"] = True
