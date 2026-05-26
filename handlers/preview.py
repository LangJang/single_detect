"""Preview-video loading and interactive ROI selection handlers."""

import os

from geometry import (
    draw_roi_on_frame,
    format_point_list,
    is_quadrilateral_valid,
    load_video_frame,
    parse_point_list,
    sort_points_convex,
)


def load_preview_video(video_path: str):
    """Return (frame_rgb, info_str, total_frames, meta_dict)."""
    if not video_path or not os.path.isfile(video_path):
        return None, "请输入有效的视频路径", 0, {}
    frame, info = load_video_frame(video_path, 0)
    if frame is None:
        return None, info, 0, {}
    info_str = (
        f"分辨率: {info['width']}x{info['height']}  |  "
        f"FPS: {info['fps']:.1f}  |  "
        f"总帧数: {info['total_frames']}  |  "
        f"时长: {int(info['duration_sec']//60)}分{int(info['duration_sec']%60)}秒"
    )
    return frame, info_str, info["total_frames"], {"path": video_path, "info": info}


def seek_preview_frame(video_meta: dict, frame_idx: int, roi_points: list):
    """Return (frame_rgb, info_str)."""
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


def handle_image_click(frame_rgb, roi_points, selection_active, click_x: float, click_y: float):
    """Handle an image click at (click_x, click_y) in original image coordinates.

    Returns (frame_rgb, coord_text, new_points, new_active).
    """
    if not selection_active:
        return frame_rgb, format_point_list(roi_points), roi_points, False
    new_points = list(roi_points) + [(click_x, click_y)]
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
        new_points = [(click_x, click_y)]
        msg = "已重置，请继续点击（第1/4个点）"
    coord_str = format_point_list(new_points)
    out_frame = draw_roi_on_frame(frame_rgb, new_points)
    return out_frame, f"{coord_str}\n{msg}" if msg else coord_str, new_points, selection_active


def start_roi_selection():
    """Return (selection_active, status_msg, empty_points)."""
    return True, "请在预览图像上依次点击4个顶点（任意顺序均可）", []


def clear_roi_selection(video_meta, frame_idx, roi_points):
    """Return (frame_rgb, status_msg, selection_active, empty_points)."""
    if not video_meta or not video_meta.get("path"):
        return None, "区域已清除", False, []
    frame, _ = seek_preview_frame(video_meta, frame_idx, [])
    return frame, "区域已清除", False, []


def sync_manual_roi(text, video_meta, frame_idx):
    """Return (frame_rgb, coord_text, points)."""
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
    """Return (calib_src_text, calib_dst_text)."""
    if not roi_points or len(roi_points) != 4:
        return "", ""
    return format_point_list(roi_points), "0,0  6,0  6,4  0,4"
