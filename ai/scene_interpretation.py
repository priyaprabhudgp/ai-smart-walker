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


# ----- OBJECT PRIORITY CONFIG -----
# Higher number = more urgent to announce.
# Anything not listed gets priority 0 and is ignored.

PRIORITY_MAP: dict[str, int] = {
    # People -- always highest, they move unpredictably
    "person":           10,

    # Stairs / drop hazards -- critical fall risk
    # NOTE: we need to custom train this
    "stairs":            9,

    # Pets -- move fast, low to ground, easy to trip over
    "dog":               8,
    "cat":               8,

    # Furniture at walking height -- direct collision risk
    "chair":             7,
    "dining table":      7,
    "bench":             6,
    "couch":             6,
    "bed":               5,

    # Bathroom hazards
    "toilet":            5,
    "sink":              4,

    # Clutter on the floor -- trip hazards
    "suitcase":          5,
    "backpack":          5,
    "handbag":           4,
    "umbrella":          4,
    "sports ball":       4,
    "skateboard":        4,

    # Low-priority awareness objects
    "potted plant":      3,
    "bottle":            2,
    "cup":               2,
    "book":              2,
    "laptop":            2,
    "tv":                2,
    "refrigerator":      2,
    "door":              2,
}

# Distance thresholds in meters -- used once ultrasonic sensors are connected.
# For now, distance_m on detections will be None and urgency defaults to "low".
URGENCY_CRITICAL = 0.8   # stop immediately
URGENCY_HIGH     = 2.0   # slow down
URGENCY_MEDIUM   = 4.0   # heads up


# ----- DATA STRUCTURES -----

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


# ----- INTERPRETER -----

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


# ----- QUICK TEST -----

if __name__ == "__main__":

    # Simulate what the detector would return from the bus.jpg test image
    mock_detections = [
    Detection(label="chair",   confidence=0.91, bbox=(50,  300, 200, 500), distance_m=0.7),
    Detection(label="person",  confidence=0.88, bbox=(300, 100, 500, 600), distance_m=1.5),
    Detection(label="cat",     confidence=0.76, bbox=(550, 400, 650, 520), distance_m=0.9),
    Detection(label="bottle",  confidence=0.65, bbox=(100, 450, 160, 550), distance_m=2.1),
    Detection(label="couch",   confidence=0.82, bbox=(400, 200, 700, 500), distance_m=3.0),
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
