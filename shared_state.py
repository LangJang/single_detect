"""Shared globals: detector, video_processor, session state.

All modules that need to reassign the detector/video_processor (e.g.
scene_manager) should ``import shared_state`` and set attributes on it.
"""

from pathlib import Path

from detector import Detector
from video_processor import VideoProcessor

DEFAULT_MODEL = "ep950-loss0.050-val_loss0.055.pth"
MODEL_CHOICES = [
    "ep950-loss0.050-val_loss0.055.pth",
    "ep400-loss0.049-val_loss0.034.pth",
]

detector = Detector(DEFAULT_MODEL)
video_processor = VideoProcessor(detector)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

from database import init_db  # noqa: E402
init_db(OUTPUT_DIR)

session_state: dict = {
    "scenes": {},                # {name: SceneConfig}
    "monitor_stop_event": None,
    "monitor_thread": None,
    "batch_stop_flag": False,
    "monitor_logs": [],
}
