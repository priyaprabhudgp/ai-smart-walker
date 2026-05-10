"""
ai/audit_log.py

Writes one JSON line per processing cycle to a daily rotating log file.

Each entry records:
  - ts          : ISO timestamp
  - sensors     : raw distances from all ultrasonic sensors (input before processing)
  - detections  : raw YOLO outputs — every object seen with confidence and distance
  - scene       : interpreted obstacles after urgency ranking (AI decision output)
  - alert       : spoken alert text, or null if nothing was said
  - nav_step    : active navigation instruction, or null if not navigating

Log location: <project_root>/logs/audit_YYYY-MM-DD.log
One JSON object per line — readable with any JSON-lines tool or plain text editor.
"""

import json
import os
import datetime
from typing import Optional


_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _log_path() -> str:
    return os.path.join(_LOG_DIR, f"audit_{datetime.date.today().isoformat()}.log")


def log_cycle(
    distances: dict,
    detections: list,
    scene,
    alert: Optional[str],
    nav_step: Optional[str],
) -> None:
    """
    Write one audit entry for a processing cycle.

    Args:
        distances:  Raw sensor readings from UltrasonicArray.read_all().
        detections: Raw Detection objects after assign_distances().
        scene:      SceneSummary produced by SceneInterpreter (AI decision output).
        alert:      The spoken alert string, or None if nothing was said.
        nav_step:   Current navigation instruction, or None if not navigating.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)

    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "sensors": dict(distances),
        "detections": [
            {
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "bbox": list(d.bbox),
                "distance_m": d.distance_m,
            }
            for d in detections
        ],
        "scene": [
            {
                "label": obs.label,
                "position": obs.position,
                "urgency": obs.urgency,
                "distance_m": obs.distance_m,
            }
            for obs in scene.obstacles
        ],
        "alert": alert,
        "nav_step": nav_step,
    }

    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
