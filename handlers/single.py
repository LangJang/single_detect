"""Tab 2 — 单视频检测: scan directory, process one video with full options."""

import json
import os
from pathlib import Path

import cv2

import shared_state
from alert_mail import AlertMailer
from calibration import CameraCalibration
from database import ProcessedDB
from geometry import parse_point_list
from video_processor import AlertEvaluator, RoiMask


def scan_video_dir(directory: str):
    """Return (file_list, info_str)."""
    if not directory or not os.path.isdir(directory):
        return [], "目录不存在"
    files = []
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        for f in Path(directory).rglob(f"*{ext}"):
            files.append(str(f))
        for f in Path(directory).glob(f"*{ext}"):
            if str(f) not in files:
                files.append(str(f))
    files.sort()
    if not files:
        return [], "目录下没有视频文件"
    return files, f"找到 {len(files)} 个视频文件"


def process_single_video(
    video_path, conf, nms_iou, frame_skip, start_sec, end_sec,
    roi_text, roi_strategy,
    calib_method, calib_physical_width, calib_physical_height,
    calib_origin_x, calib_origin_y, calib_dst_text,
    alert_threshold, alert_w_count, alert_w_area, alert_max_count,
    output_annotated,
    email_enabled, email_smtp_server, email_smtp_port,
    email_sender, email_password, email_receivers,
    progress_callback=None,
):
    """Process a single video. Returns (preview_numpy, summary_str, csv_path).

    progress_callback(current, total, desc) is called for progress updates.
    """
    if not video_path or not os.path.isfile(video_path):
        return None, "视频文件不存在", ""

    if progress_callback:
        progress_callback(0, 100, "准备中...")

    cap_info = cv2.VideoCapture(video_path)
    frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    roi = None
    roi_points = parse_point_list(roi_text)
    if roi_points and len(roi_points) >= 3:
        roi = RoiMask(roi_points, (frame_w, frame_h), strategy=roi_strategy)

    calib = None
    if roi_points and len(roi_points) == 4:
        if calib_method == "diagonal":
            if calib_physical_width > 0 and calib_physical_height > 0:
                calib = CameraCalibration()
                try:
                    calib.set_from_roi(roi_points, calib_physical_width,
                                       calib_physical_height, (frame_w, frame_h),
                                       calib_origin_x, calib_origin_y)
                except Exception as e:
                    return None, f"标定设置失败: {e}", ""
        else:  # four_point
            dst = parse_point_list(calib_dst_text)
            if dst and len(dst) == 4:
                calib = CameraCalibration()
                try:
                    calib.set_homography(roi_points, dst, (frame_w, frame_h))
                except Exception as e:
                    return None, f"标定设置失败: {e}", ""

    alert_eval = AlertEvaluator(
        threshold=alert_threshold, weight_count=alert_w_count,
        weight_area=alert_w_area, max_expected_count=alert_max_count,
    )

    output_video = None
    if output_annotated:
        output_video = str(shared_state.OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4")

    end = end_sec if end_sec > 0 else None

    def on_progress(current, total):
        if progress_callback:
            progress_callback(current, total, f"检测中 {current}/{total} 帧")

    try:
        if progress_callback:
            progress_callback(5, 100, "检测中...")
        result = shared_state.video_processor.process_video(
            video_path=video_path, conf=conf, nms_iou=nms_iou,
            frame_skip=frame_skip, start_sec=start_sec, end_sec=end,
            roi=roi, calibration=calib,
            output_video_path=output_video, progress_callback=on_progress,
        )
    except Exception as e:
        return None, f"处理失败: {e}", ""

    if progress_callback:
        progress_callback(90, 100, "生成报告...")

    csv_path = str(shared_state.OUTPUT_DIR / f"{Path(video_path).stem}_detections.csv")
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
        file_path=video_path,
        mode="single",
        detection_count=result.total_detections,
        frames_processed=result.frames_processed,
        csv_path=csv_path,
    )

    if alert_frames > 0:
        AlertMailer.try_send(
            email_enabled, email_smtp_server, email_smtp_port,
            email_sender, email_password, email_receivers,
            video_name=Path(video_path).name, video_path=video_path,
            frames_processed=result.frames_processed,
            total_detections=result.total_detections,
            alert_frames=alert_frames,
            alert_threshold=alert_threshold,
            alert_weights=(alert_w_count, alert_w_area),
            class_counts=result.class_counts,
        )

    preview = None
    for fr in reversed(result.frame_results):
        if fr.detections:
            cap_tmp = cv2.VideoCapture(video_path)
            cap_tmp.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_idx)
            ret, frame = cap_tmp.read()
            cap_tmp.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, _ = shared_state.detector.detect(frame_rgb, conf=conf)
                if roi:
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    roi.draw(annotated_bgr)
                    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                preview = annotated
            break

    if progress_callback:
        progress_callback(100, 100, "完成")

    return preview, summary, csv_path
