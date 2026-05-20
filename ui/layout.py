"""Gradio UI layout and event bindings.

Imports handlers from their respective modules and wires them to the
Gradio components.  The ``app`` instance is created at module level so
``main.py`` can import and launch it.
"""

from datetime import datetime
from pathlib import Path

import gradio as gr

import shared_state
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


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks app (event bindings wired)."""

    with gr.Blocks(title="YOLO 目标检测系统") as app:

        gr.Markdown("# YOLO 目标检测系统")

        # ═══════════════════════════════════════════════════════════════
        # Scene (Tag) Management
        # ═══════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════
        # Detection Parameters
        # ═══════════════════════════════════════════════════════════════
        with gr.Group():
            gr.Markdown("## 检测参数")
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=shared_state.MODEL_CHOICES, value=shared_state.DEFAULT_MODEL,
                    label="检测模型", scale=3,
                )
                device_selector = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"], value="auto",
                    label="计算设备", scale=1,
                )
                model_status = gr.Textbox(
                    label="状态", value=f"{shared_state.DEFAULT_MODEL} 已就绪",
                    interactive=False, scale=2,
                )
            with gr.Row():
                conf_slider = gr.Slider(0.1, 0.9, 0.25, 0.05, label="置信度阈值")
                nms_iou_slider = gr.Slider(0.1, 0.9, 0.3, 0.05, label="NMS IoU")
                frame_skip_slider = gr.Slider(1, 60, 10, 1, label="帧采样间隔")

        # ═══════════════════════════════════════════════════════════════
        # ROI & Calibration
        # ═══════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════
        # Alert Settings
        # ═══════════════════════════════════════════════════════════════
        with gr.Accordion("告警设置", open=False):
            with gr.Row():
                alert_threshold = gr.Slider(0.1, 1.0, 0.5, 0.05, label="告警阈值")
                alert_max_count = gr.Number(10, label="期望最大目标数", precision=0)
            with gr.Row():
                alert_w_count = gr.Slider(0.0, 1.0, 0.4, 0.1, label="数量权重")
                alert_w_area = gr.Slider(0.0, 1.0, 0.6, 0.1, label="面积权重")

            with gr.Accordion("邮件告警 (SMTP)", open=False):
                email_enabled = gr.Checkbox(False, label="启用邮件告警")
                with gr.Row():
                    email_smtp_server = gr.Textbox(
                        "smtp.163.com", label="SMTP 服务器", scale=2,
                    )
                    email_smtp_port = gr.Number(465, label="端口", precision=0, scale=1)
                with gr.Row():
                    email_sender = gr.Textbox(
                        "", label="发件邮箱", placeholder="your@163.com", scale=2,
                    )
                    email_password = gr.Textbox(
                        "", label="授权码", type="password", placeholder="SMTP 授权码（非登录密码）", scale=2,
                    )
                email_receivers = gr.Textbox(
                    "", label="收件邮箱",
                    placeholder="receiver@qq.com (多个用逗号或分号分隔)",
                )

        # --- Hidden States ---
        roi_points_state = gr.State([])
        roi_selection_active = gr.State(False)
        preview_video_meta = gr.State({})

        # ═══════════════════════════════════════════════════════════════
        # Task Tabs
        # ═══════════════════════════════════════════════════════════════
        with gr.Tabs():
            # ── Tab 1: 实时监控 ────────────────────────────────────
            with gr.TabItem("实时监控"):
                with gr.Group():
                    gr.Markdown("### 监控执行")
                    with gr.Row():
                        monitor_dir = gr.Textbox(
                            label="监控目录", scale=4, value=str(Path(__file__).parent.parent),
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
                        label="日志", lines=8, interactive=False,
                    )

            # ── Tab 2: 单视频检测 ──────────────────────────────────
            with gr.TabItem("单视频检测"):
                with gr.Group():
                    gr.Markdown("### 视频处理")
                    with gr.Row():
                        video_dir = gr.Textbox(
                            label="视频所在目录", scale=3, value=str(Path(__file__).parent.parent),
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
                                label="处理报告", lines=12,
                            )
                            result_csv = gr.Textbox(label="输出文件", interactive=False)

            # ── Tab 3: 批量处理 ────────────────────────────────────
            with gr.TabItem("批量处理"):
                with gr.Group():
                    gr.Markdown("### 批量执行")
                    with gr.Row():
                        batch_dir = gr.Textbox(
                            label="视频根目录", value=str(Path(__file__).parent.parent),
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
                            label="汇总", lines=5, scale=1,
                        )
                        batch_log = gr.Textbox(label="处理日志", lines=8, scale=2)
                    batch_output_dir = gr.Textbox(label="结果目录", interactive=False)

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: Scene Management
        # ═══════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: Model
        # ═══════════════════════════════════════════════════════════════
        model_selector.change(
            on_model_change, inputs=[model_selector, device_selector],
            outputs=[model_status],
        )
        device_selector.change(
            on_model_change, inputs=[model_selector, device_selector],
            outputs=[model_status],
        )

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: ROI & Preview
        # ═══════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: Tab 1 — 实时监控
        # ═══════════════════════════════════════════════════════════════
        monitor_start_btn.click(
            start_monitor,
            inputs=[monitor_dir, conf_slider, nms_iou_slider, frame_skip_slider,
                    roi_coords_display, roi_strategy, poll_interval, stable_time,
                    alert_threshold, alert_w_count, alert_w_area, alert_max_count,
                    email_enabled, email_smtp_server, email_smtp_port,
                    email_sender, email_password, email_receivers],
            outputs=[monitor_status, monitor_log],
        )
        monitor_stop_btn.click(stop_monitor, outputs=[monitor_status, monitor_log])
        monitor_refresh_btn.click(lambda: _monitor_log_text(), outputs=[monitor_log])

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: Tab 2 — 单视频检测
        # ═══════════════════════════════════════════════════════════════
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
                email_enabled, email_smtp_server, email_smtp_port,
                email_sender, email_password, email_receivers,
            ],
            outputs=[result_preview, result_summary, result_csv],
        )

        # ═══════════════════════════════════════════════════════════════
        # Event Bindings: Tab 3 — 批量处理
        # ═══════════════════════════════════════════════════════════════
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
                email_enabled, email_smtp_server, email_smtp_port,
                email_sender, email_password, email_receivers,
            ],
            outputs=[batch_summary, batch_log, batch_output_dir],
        )
        batch_stop_btn.click(stop_batch, outputs=[batch_summary])

        # ═══════════════════════════════════════════════════════════════
        # App Load
        # ═══════════════════════════════════════════════════════════════
        app.load(refresh_scene_list, outputs=[scene_list])

    return app


# Module-level singleton — main.py imports this
app = build_app()
