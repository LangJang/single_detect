"""Tab 3 — 批量处理: scan date-range, process multiple videos, skip processed."""

import json
import os
from pathlib import Path

import cv2

import shared_state
from alert_mail import AlertMailer
from calibration import CameraCalibration
from file_scanner import FileScanner
from database import ProcessedDB
from geometry import parse_point_list
from video_processor import AlertEvaluator, RoiMask


def scan_directory(root_dir, start_date, end_date, start_time, end_time):
    """Return (info_str, file_list)."""
    if not root_dir or not os.path.isdir(root_dir):
        return "目录不存在", []
    scanner = FileScanner(root_dir)
    try:
        files = scanner.scan_by_date_range(start_date, end_date, start_time, end_time)
    except Exception as e:
        return f"扫描失败: {e}", []
    if not files:
        return "未找到匹配的视频文件", []
    choices = [f["path"] for f in files]
    info = f"扫描到 {len(files)} 个视频（{start_date} {start_time} ~ {end_date} {end_time}）"
    return info, choices


def process_batch(
    file_list, conf, nms_iou, frame_skip, roi_text, roi_strategy,
    calib_src_text, calib_dst_text, alert_threshold, alert_w_count,
    alert_w_area, alert_max_count, resume,
    email_enabled, email_smtp_server, email_smtp_port,
    email_sender, email_password, email_receivers,
    progress_callback=None,
):
    """Process a batch of videos. Returns (summary_str, log_str, output_dir).

    progress_callback(current, total, desc) is called for progress updates.
    """
    if not file_list:
        return "没有待处理的视频", "", ""

    shared_state.session_state["batch_stop_flag"] = False
    if progress_callback:
        progress_callback(0, len(file_list), "开始批量处理...")

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

    alert_eval = AlertEvaluator(
        threshold=alert_threshold, weight_count=alert_w_count,
        weight_area=alert_w_area, max_expected_count=alert_max_count,
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
        if progress_callback:
            progress_callback(i, total, f"({i+1}/{total}): {Path(vp).name}")
        if resume and ProcessedDB.is_processed(vp, "batch"):
            skipped += 1
            log_lines.append(f"[跳过] {Path(vp).name}")
            continue
        try:
            result = shared_state.video_processor.process_video(
                video_path=vp, conf=conf, nms_iou=nms_iou,
                frame_skip=frame_skip, roi=roi, calibration=calib,
            )
            csv_path = str(all_csv_dir / f"{Path(vp).stem}_detections.csv")
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
                file_path=vp,
                mode="batch",
                detection_count=result.total_detections,
                frames_processed=result.frames_processed,
                csv_path=csv_path,
            )
            success += 1
            alert_total += alert_frames
            if alert_frames > 0:
                AlertMailer.try_send(
                    email_enabled, email_smtp_server, email_smtp_port,
                    email_sender, email_password, email_receivers,
                    video_name=Path(vp).name, video_path=vp,
                    frames_processed=result.frames_processed,
                    total_detections=result.total_detections,
                    alert_frames=alert_frames,
                    alert_threshold=alert_threshold,
                    alert_weights=(alert_w_count, alert_w_area),
                    class_counts=result.class_counts,
                )
            log_lines.append(
                f"[OK] {Path(vp).name} — {result.total_detections} 目标"
                f" (告警 {alert_frames}/{result.frames_processed} 帧)"
            )
            for label, count in result.class_counts.items():
                aggregated_counts[label] = aggregated_counts.get(label, 0) + count
        except Exception as e:
            failed += 1
            log_lines.append(f"[FAIL] {Path(vp).name} — {e}")

    summary = (
        f"成功: {success}    跳过: {skipped}    失败: {failed}\n"
        f"累计检测目标: {sum(aggregated_counts.values())}\n"
        f"累计告警帧: {alert_total}\n"
        f"类别分布: {json.dumps(aggregated_counts, ensure_ascii=False)}\n"
        f"结果目录: {all_csv_dir}"
    )
    if progress_callback:
        progress_callback(total, total, "完成")
    return summary, "\n".join(log_lines), str(all_csv_dir)


def stop_batch():
    """Set the batch stop flag."""
    shared_state.session_state["batch_stop_flag"] = True
