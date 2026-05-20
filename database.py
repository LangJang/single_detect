"""SQLite persistence layer — replaces scattered JSON files with a single DB.

Thread-safe: each thread gets its own sqlite3.Connection via threading.local().
WAL mode allows concurrent reads during writes.

Tables
------
scenes          — scene (Tag) configs (replaces output/scene_*.json)
processed_files — which videos have been processed (replaces *_processed.json)
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

_local = threading.local()
DB_PATH: str | None = None


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    """Return the thread-local connection, auto-creating it on first access."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


# ---------------------------------------------------------------------------
# Initialisation & migration
# ---------------------------------------------------------------------------
def init_db(output_dir: Path):
    """Create tables and migrate old JSON data on first run.

    Must be called once at startup before any DB queries.
    """
    global DB_PATH
    DB_PATH = str(output_dir / "app.db")
    is_new = not os.path.exists(DB_PATH)

    conn = get_db()
    _create_tables(conn)

    if is_new:
        _migrate_json_data(conn, output_dir)

    # Return early — keep the startup connection; it will be re-used by the
    # same thread or a new one will be created for other threads.
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scenes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT UNIQUE NOT NULL,
            model_path TEXT DEFAULT 'ep950-loss0.050-val_loss0.055.pth',
            confidence REAL DEFAULT 0.25,
            nms_iou    REAL DEFAULT 0.3,
            frame_skip INTEGER DEFAULT 10,
            device     TEXT DEFAULT 'auto',
            roi_points     TEXT DEFAULT '[]',
            roi_strategy   TEXT DEFAULT 'centroid',
            roi_resolution TEXT,
            calib_src      TEXT DEFAULT '[]',
            calib_dst      TEXT DEFAULT '[]',
            calib_resolution TEXT,
            alert_threshold    REAL DEFAULT 0.5,
            alert_weight_count REAL DEFAULT 0.4,
            alert_weight_area  REAL DEFAULT 0.6,
            alert_max_count    INTEGER DEFAULT 10,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS processed_files (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path        TEXT NOT NULL,
            mode             TEXT NOT NULL CHECK(mode IN ('monitor','batch','single')),
            detection_count  INTEGER DEFAULT 0,
            frames_processed INTEGER DEFAULT 0,
            csv_path         TEXT,
            processed_at     TEXT NOT NULL,
            UNIQUE(file_path, mode)
        );

        CREATE INDEX IF NOT EXISTS idx_processed_mode
            ON processed_files(mode);
        CREATE INDEX IF NOT EXISTS idx_processed_path
            ON processed_files(file_path);
    """)
    conn.commit()


def _migrate_json_data(conn: sqlite3.Connection, output_dir: Path):
    """One-shot: read old JSON files into the new DB tables."""
    from file_scanner import SceneConfig

    # --- scene_*.json ---
    migrated_scenes = 0
    for f in sorted(output_dir.glob("scene_*.json")):
        try:
            with open(str(f), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = SceneConfig.from_dict(data)
            _insert_scene(conn, cfg.to_dict())
            migrated_scenes += 1
        except Exception:
            pass

    # --- monitor_processed.json ---
    monitor_json = output_dir / "monitor_processed.json"
    migrated_monitor = _migrate_processed_json(conn, monitor_json, "monitor")

    # --- batch_processed.json ---
    batch_json = output_dir / "batch_processed.json"
    migrated_batch = _migrate_processed_json(conn, batch_json, "batch")

    conn.commit()

    if migrated_scenes or migrated_monitor or migrated_batch:
        print(
            f"[DB migrate] scenes={migrated_scenes}  "
            f"monitor={migrated_monitor}  batch={migrated_batch}"
        )


def _migrate_processed_json(conn: sqlite3.Connection, path: Path, mode: str) -> int:
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
                    file_path,
                    mode,
                    summary.get("detections", 0),
                    summary.get("frames", 0),
                    record.get("processed_at", datetime.now().isoformat()) if isinstance(record, dict) else datetime.now().isoformat(),
                ),
            )
            count += 1
        except Exception:
            pass
    return count


def _insert_scene(conn: sqlite3.Connection, fields: dict):
    """Insert one scene row. *fields* is a SceneConfig.to_dict() result."""
    ts = datetime.now().isoformat()
    list_fields = ["roi_points", "calib_src", "calib_dst"]
    json_fields = {k: json.dumps(fields.get(k, [])) for k in list_fields}
    json_fields["roi_resolution"] = (
        json.dumps(fields["roi_resolution"]) if fields.get("roi_resolution") else None
    )
    json_fields["calib_resolution"] = (
        json.dumps(fields["calib_resolution"]) if fields.get("calib_resolution") else None
    )

    conn.execute(
        """INSERT OR REPLACE INTO scenes
           (name, model_path, confidence, nms_iou, frame_skip, device,
            roi_points, roi_strategy, roi_resolution,
            calib_src, calib_dst, calib_resolution,
            alert_threshold, alert_weight_count, alert_weight_area, alert_max_count,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fields["name"],
            fields.get("model_path", "ep950-loss0.050-val_loss0.055.pth"),
            fields.get("confidence", 0.25),
            fields.get("nms_iou", 0.3),
            fields.get("frame_skip", 10),
            fields.get("device", "auto"),
            json_fields["roi_points"],
            fields.get("roi_strategy", "centroid"),
            json_fields["roi_resolution"],
            json_fields["calib_src"],
            json_fields["calib_dst"],
            json_fields["calib_resolution"],
            fields.get("alert_threshold", 0.5),
            fields.get("alert_weight_count", 0.4),
            fields.get("alert_weight_area", 0.6),
            fields.get("alert_max_count", 10),
            ts,
            ts,
        ),
    )


# ---------------------------------------------------------------------------
# Scene CRUD
# ---------------------------------------------------------------------------
class SceneDB:
    @staticmethod
    def list_all() -> list[str]:
        rows = get_db().execute("SELECT name FROM scenes ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    @staticmethod
    def get(name: str) -> dict | None:
        row = get_db().execute("SELECT * FROM scenes WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        # Decode JSON columns back to Python lists
        for col in ("roi_points", "calib_src", "calib_dst"):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                d[col] = []
        for col in ("roi_resolution", "calib_resolution"):
            try:
                d[col] = json.loads(d[col]) if d[col] else None
            except (json.JSONDecodeError, TypeError):
                d[col] = None
        return d

    @staticmethod
    def save(name: str, fields: dict):
        """INSERT or REPLACE a scene. *fields* comes from SceneConfig.to_dict()."""
        now = datetime.now().isoformat()
        existing = SceneDB.get(name)
        created = existing["created_at"] if existing else now
        fields["name"] = name
        fields["created_at"] = created
        fields["updated_at"] = now
        _insert_scene(get_db(), fields)
        get_db().commit()

    @staticmethod
    def delete(name: str) -> bool:
        conn = get_db()
        cur = conn.execute("DELETE FROM scenes WHERE name = ?", (name,))
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Processed-file tracking
# ---------------------------------------------------------------------------
class ProcessedDB:
    @staticmethod
    def is_processed(file_path: str, mode: str) -> bool:
        row = get_db().execute(
            "SELECT 1 FROM processed_files WHERE file_path = ? AND mode = ?",
            (file_path, mode),
        ).fetchone()
        return row is not None

    @staticmethod
    def mark_processed(
        file_path: str,
        mode: str,
        detection_count: int = 0,
        frames_processed: int = 0,
        csv_path: str | None = None,
    ):
        conn = get_db()
        conn.execute(
            """INSERT OR REPLACE INTO processed_files
               (file_path, mode, detection_count, frames_processed, csv_path, processed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_path, mode, detection_count, frames_processed, csv_path, datetime.now().isoformat()),
        )
        conn.commit()

    @staticmethod
    def get_processed_paths(mode: str) -> list[str]:
        rows = get_db().execute(
            "SELECT file_path FROM processed_files WHERE mode = ?", (mode,)
        ).fetchall()
        return [r["file_path"] for r in rows]

    @staticmethod
    def clear(mode: str | None = None):
        conn = get_db()
        if mode:
            conn.execute("DELETE FROM processed_files WHERE mode = ?", (mode,))
        else:
            conn.execute("DELETE FROM processed_files")
        conn.commit()
