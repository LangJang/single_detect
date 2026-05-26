"""
UI event bindings for the YOLO detection Gradio app.

All event wiring is centralized here to keep layout.py focused on
component construction.
"""

import gradio as gr

from handlers.batch import process_batch, scan_directory, stop_batch
from handlers.monitor import _monitor_log_text, start_monitor, stop_monitor
from handlers.preview import (
    autofill_calib_from_roi,
    clear_roi_selection,
    handle_image_click,
    load_preview_video,
    seek_preview_frame,
    start_roi_selection,
    sync_manual_roi,
)
from handlers.single import process_single_video, scan_video_dir
from scene_manager import (
    delete_scene,
    load_scene,
    on_model_change,
    refresh_scene_list,
    save_scene,
)


# ruff: noqa: E704 — lambdas in .click() chains are intentional
def wire_events(app: gr.Blocks, c: dict):
    # ── Scene Management ──
    c["scene_save_btn"].click(
        save_scene,
        inputs=[
            c["scene_name"], c["model_selector"], c["conf_slider"],
            c["nms_iou_slider"], c["frame_skip_slider"], c["device_selector"],
            c["roi_coords_display"], c["roi_strategy"],
            c["calib_src"], c["calib_dst"],
            c["alert_threshold"], c["alert_w_count"],
            c["alert_w_area"], c["alert_max_count"],
        ],
        outputs=[c["scene_status"], c["scene_list"]],
    )
    c["scene_load_btn"].click(
        load_scene, inputs=[c["scene_list"]],
        outputs=[
            c["model_selector"], c["conf_slider"], c["nms_iou_slider"],
            c["frame_skip_slider"], c["device_selector"],
            c["roi_coords_display"], c["roi_strategy"],
            c["calib_src"], c["calib_dst"],
            c["alert_threshold"], c["alert_w_count"],
            c["alert_w_area"], c["alert_max_count"],
            c["roi_points_state"], c["scene_status"], c["model_status"],
        ],
    )
    c["scene_refresh_btn"].click(
        refresh_scene_list, outputs=[c["scene_list"]])
    c["scene_delete_btn"].click(
        delete_scene, inputs=[c["scene_list"]],
        outputs=[c["scene_status"], c["scene_list"]],
    )

    # ── Model ──
    c["model_selector"].change(
        on_model_change,
        inputs=[c["model_selector"], c["device_selector"]],
        outputs=[c["model_status"]],
    )
    c["device_selector"].change(
        on_model_change,
        inputs=[c["model_selector"], c["device_selector"]],
        outputs=[c["model_status"]],
    )

    # ── ROI & Preview ──
    c["load_preview_btn"].click(
        load_preview_video, inputs=[c["preview_video_path"]],
        outputs=[
            c["preview_image"], c["frame_info_display"],
            c["frame_position_slider"], c["preview_video_meta"],
        ],
    )
    c["frame_position_slider"].change(
        seek_preview_frame,
        inputs=[
            c["preview_video_meta"], c["frame_position_slider"],
            c["roi_points_state"],
        ],
        outputs=[c["preview_image"], c["frame_info_display"]],
    )
    c["preview_image"].select(
        handle_image_click,
        inputs=[
            c["preview_image"], c["roi_points_state"],
            c["roi_selection_active"],
        ],
        outputs=[
            c["preview_image"], c["roi_coords_display"],
            c["roi_points_state"], c["roi_selection_active"],
        ],
    )
    c["roi_start_btn"].click(
        start_roi_selection,
        outputs=[
            c["roi_selection_active"], c["roi_coords_display"],
            c["roi_points_state"],
        ],
    )
    c["roi_clear_btn"].click(
        clear_roi_selection,
        inputs=[
            c["preview_video_meta"], c["frame_position_slider"],
            c["roi_points_state"],
        ],
        outputs=[
            c["preview_image"], c["roi_coords_display"],
            c["roi_selection_active"], c["roi_points_state"],
        ],
    )
    c["roi_manual_text"].change(
        sync_manual_roi,
        inputs=[
            c["roi_manual_text"], c["preview_video_meta"],
            c["frame_position_slider"],
        ],
        outputs=[
            c["preview_image"], c["roi_coords_display"],
            c["roi_points_state"],
        ],
    )
    c["calib_autofill_btn"].click(
        autofill_calib_from_roi, inputs=[c["roi_points_state"]],
        outputs=[c["calib_src"], c["calib_dst"]],
    )

    # ── Tab 1: 实时监控 ──
    c["monitor_start_btn"].click(
        start_monitor,
        inputs=[
            c["monitor_dir"], c["conf_slider"], c["nms_iou_slider"],
            c["frame_skip_slider"], c["roi_coords_display"],
            c["roi_strategy"], c["poll_interval"], c["stable_time"],
            c["alert_threshold"], c["alert_w_count"],
            c["alert_w_area"], c["alert_max_count"],
            c["email_enabled"], c["email_smtp_server"],
            c["email_smtp_port"], c["email_sender"],
            c["email_password"], c["email_receivers"],
        ],
        outputs=[c["monitor_status"], c["monitor_log"]],
    )
    c["monitor_stop_btn"].click(
        stop_monitor,
        outputs=[c["monitor_status"], c["monitor_log"]],
    )
    c["monitor_refresh_btn"].click(
        lambda: _monitor_log_text(), outputs=[c["monitor_log"]])

    # ── Tab 2: 单视频检测 ──
    c["scan_btn"].click(
        scan_video_dir, inputs=[c["video_dir"]],
        outputs=[c["video_selector"], c["scan_info"]],
    )
    c["process_btn"].click(
        process_single_video,
        inputs=[
            c["video_selector"], c["conf_slider"], c["nms_iou_slider"],
            c["frame_skip_slider"], c["start_sec"], c["end_sec"],
            c["roi_coords_display"], c["roi_strategy"],
            c["calib_src"], c["calib_dst"],
            c["alert_threshold"], c["alert_w_count"],
            c["alert_w_area"], c["alert_max_count"],
            c["output_annotated"],
            c["email_enabled"], c["email_smtp_server"],
            c["email_smtp_port"], c["email_sender"],
            c["email_password"], c["email_receivers"],
        ],
        outputs=[c["result_preview"], c["result_summary"], c["result_csv"]],
    )

    # ── Tab 3: 批量处理 ──
    c["batch_scan_btn"].click(
        scan_directory,
        inputs=[
            c["batch_dir"], c["batch_start_date"], c["batch_end_date"],
            c["batch_start_time"], c["batch_end_time"],
        ],
        outputs=[c["batch_scan_info"], c["batch_file_list"]],
    )
    c["batch_process_btn"].click(
        process_batch,
        inputs=[
            c["batch_file_list"], c["conf_slider"], c["nms_iou_slider"],
            c["frame_skip_slider"], c["roi_coords_display"],
            c["roi_strategy"], c["calib_src"], c["calib_dst"],
            c["alert_threshold"], c["alert_w_count"],
            c["alert_w_area"], c["alert_max_count"],
            c["batch_resume"],
            c["email_enabled"], c["email_smtp_server"],
            c["email_smtp_port"], c["email_sender"],
            c["email_password"], c["email_receivers"],
        ],
        outputs=[c["batch_summary"], c["batch_log"], c["batch_output_dir"]],
    )
    c["batch_stop_btn"].click(
        stop_batch, outputs=[c["batch_summary"]])

    # ── App Load ──
    app.load(refresh_scene_list, outputs=[c["scene_list"]])
