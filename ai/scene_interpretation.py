"""
ai/scene_interpretation.py

Takes a list of Detection objects and produces a structured SceneSummary --
priority-ranked obstacles, rough position (left/center/right), and urgency level.

No LLM needed -- intentionally fast and rule-based.
The language_generation.py module turns these summaries into actual speech.
"""

from dataclasses import dataclass, field
from typing import Optional
from object_detection import Detection


# ── Object priority config ─────────────────────────────────────────────────────
# Higher number = more urgent to announce.
# Anything not listed gets priority 0 and is ignored.

PRIORITY_MAP: dict[str, int] = {
    "person":        10,
    "bicycle":        9,
    "motorcycle":     9,
    "car":            8,
    "truck":          8,
    "bus":            8,
    "dog":            7,
    "cat":            6,
    "chair":          5,
    "bench":          5,
    "dining table":   4,
    "potted plant":   3,
    "backpack":       2,
    "suitcase":       2,
    "umbrella":       2,
}

# Distance thresholds in meters -- used once ultrasonic sensors are connected.
# For now, distance_m on detections will be None and urgency defaults to "low".
URGENCY_CRITICAL = 0.8   # stop immediately
URGENCY_HIGH     = 2.0   # slow down
URGENCY_MEDIUM   = 4.0   # heads up


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ObstacleSummary:
    label: str
    priority: int
    position: str                  # "left", "center", or "right"
    distance_m: Optional[float] = None
    urgency: str = "low"           # "critical", "high", "medium", "low"
    confidence: float = 0.0


@dataclass
class SceneSummary:
    obstacles: list[ObstacleSummary] = field(default_factory=list)
    frame_width: int = 640

    @property
    def top_obstacle(self) -> Optional[ObstacleSummary]:
        """The single highest-priority obstacle in the scene."""
        return self.obstacles[0] if self.obstacles else None

    @property
    def is_clear(self) -> bool:
        return len(self.obstacles) == 0


# ── Interpreter ────────────────────────────────────────────────────────────────

class SceneInterpreter:
    """
    Converts raw detections into a SceneSummary.

    Args:
        frame_width:    Pixel width of the camera frame (default 640).
        min_priority:   Skip objects with priority below this value.
        max_obstacles:  Cap how many obstacles to include (top N by priority).
    """

    def __init__(
        self,
        frame_width: int = 640,
        min_priority: int = 1,
        max_obstacles: int = 3,
    ):
        self.frame_width = frame_width
        self.min_priority = min_priority
        self.max_obstacles = max_obstacles

    def interpret(self, detections: list[Detection]) -> SceneSummary:
        summary = SceneSummary(frame_width=self.frame_width)

        for d in detections:
            priority = PRIORITY_MAP.get(d.label, 0)
            if priority < self.min_priority:
                continue

            position = self._horizontal_position(d.bbox)
            urgency = self._urgency(d.distance_m)

            summary.obstacles.append(ObstacleSummary(
                label=d.label,
                priority=priority,
                position=position,
                distance_m=d.distance_m,
                urgency=urgency,
                confidence=d.confidence,
            ))

        # Sort: priority first, then urgency
        urgency_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        summary.obstacles.sort(
            key=lambda o: (o.priority, urgency_rank.get(o.urgency, 0)),
            reverse=True,
        )
        summary.obstacles = summary.obstacles[:self.max_obstacles]
        return summary

    def _horizontal_position(self, bbox: tuple) -> str:
        """Classify bbox center as left / center / right third of frame."""
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2
        third = self.frame_width / 3
        if cx < third:
            return "left"
        elif cx < 2 * third:
            return "center"
        else:
            return "right"

    def _urgency(self, distance_m: Optional[float]) -> str:
        """Assign urgency from distance. Returns low if no distance data yet."""
        if distance_m is None:
            return "low"
        if distance_m <= URGENCY_CRITICAL:
            return "critical"
        if distance_m <= URGENCY_HIGH:
            return "high"
        if distance_m <= URGENCY_MEDIUM:
            return "medium"
        return "low"


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Simulate what the detector would return from the bus.jpg test image
    mock_detections = [
        Detection(label="bus",    confidence=0.87, bbox=(22,  231, 805, 756), distance_m=None),
        Detection(label="person", confidence=0.87, bbox=(48,  398, 245, 902), distance_m=1.2),
        Detection(label="person", confidence=0.85, bbox=(669, 392, 809, 877), distance_m=3.5),
        Detection(label="person", confidence=0.83, bbox=(221, 405, 344, 857), distance_m=0.6),
    ]

    # Note: the image from your test was 809px wide, not 640 -- set frame_width to match
    interpreter = SceneInterpreter(frame_width=809)
    scene = interpreter.interpret(mock_detections)

    print(f"Path clear: {scene.is_clear}")
    print(f"Top obstacle: {scene.top_obstacle.label} ({scene.top_obstacle.urgency})\n")
    print("All ranked obstacles:")
    for obs in scene.obstacles:
        dist = f"{obs.distance_m:.1f}m" if obs.distance_m else "no distance yet"
        print(f"  [{obs.urgency.upper():<8}] {obs.label:<10} @ {obs.position:<6}  {dist}  (priority {obs.priority})")
