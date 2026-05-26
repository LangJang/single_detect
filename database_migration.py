"""One-shot JSON→SQLite migration — reads old scene/progress files into the DB."""

import json
from pathlib import Path

from scene_config import SceneConfig


type SaveSceneFn = object  # (dict) -> None


def migrate_json_data(output_dir: Path, save_scene: SaveSceneFn, conn) -> str:
    """Migrate old JSON files into the database. Returns a status message."""
    migrated_scenes = 0

    for f in sorted(output_dir.glob("scene_*.json")):
        try:
            with open(str(f), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = SceneConfig.from_dict(data)
            save_scene(cfg.to_dict())
            migrated_scenes += 1
        except Exception:
            pass

    monitor = _migrate_processed(conn, output_dir / "monitor_processed.json", "monitor")
    batch = _migrate_processed(conn, output_dir / "batch_processed.json", "batch")

    if migrated_scenes or monitor or batch:
        return f"[DB migrate] scenes={migrated_scenes}  monitor={monitor}  batch={batch}"
    return ""


def _migrate_processed(conn, path: Path, mode: str) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return 0

    count = 0
    for file_path, record in data.items():
        summary = record.get("summary", {}) if isinstance(record, dict) else {}
        try:
            conn.execute(
                """INSERT OR IGNORE INTO processed_files
                   (file_path, mode, detection_count, frames_processed, processed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    file_path, mode,
                    summary.get("detections", 0),
                    summary.get("frames", 0),
                    record.get("processed_at", "") if isinstance(record, dict) else "",
                ),
            )
            count += 1
        except Exception:
            pass
    return count
