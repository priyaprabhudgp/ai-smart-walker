"""
ai/object_detection.py

Runs YOLOv8 object detection on a single frame (numpy array or image path).
Designed to work standalone now (test images) and with Pi camera later --
just swap in a real frame from camera_stream.py, same interface.

Install deps first:
    pip install ultralytics opencv-python
"""

from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os


# ----- DETECTOR RESULT CONTAINER -----

@dataclass
class Detection:
    label: str          # e.g. "person", "chair", "bicycle"
    confidence: float   # scale of 0.0 – 1.0
    bbox: tuple         # (x1, y1, x2, y2) in pixels
    distance_m: Optional[float] = None  # filled in later by ultrasonic data


# ----- DETECTOR CLASS -----

class ObjectDetector:
    """
    Wraps YOLOv8 models. Call detect() with any frame or image path.

    Args:
        model_path:       Path to primary .pt weights (default yolov8n.pt, all COCO classes).
        stairs_model_path: Optional path to custom stairs-only model. Results are merged.
        confidence_threshold: Minimum score to include a detection.
        device: "cpu", "cuda", or "mps" (Apple Silicon). Defaults to cpu.
    """

    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        stairs_model_path: Optional[str] = None,
        confidence_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        print(f"[ObjectDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)

        self.stairs_model = None
        if stairs_model_path and os.path.exists(stairs_model_path):
            print(f"[ObjectDetector] Loading stairs model: {stairs_model_path}")
            self.stairs_model = YOLO(stairs_model_path)

        door_model_path = os.path.join(os.path.dirname(__file__), "door.pt")
        self.door_model = None
        if os.path.exists(door_model_path):
            print(f"[ObjectDetector] Loading door classifier: {door_model_path}")
            self.door_model = YOLO(door_model_path)

        print("[ObjectDetector] Ready.")

    def detect(self, frame) -> list[Detection]:
        """
        Run detection on a frame.

        Args:
            frame: numpy BGR array (from cv2 or picamera2) OR a file path string.

        Returns:
            List of Detection objects, sorted by confidence descending.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                ))

        # Merge stairs detections from custom model if available
        if self.stairs_model is not None:
            stairs_results = self.stairs_model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
            for result in stairs_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(Detection(
                        label="stairs",
                        confidence=float(box.conf),
                        bbox=(x1, y1, x2, y2),
                    ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_from_path(self, image_path: str) -> list[Detection]:
        """Convenience wrapper for a file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.detect(image_path)

    def classify_door(self, frame: np.ndarray) -> Optional[str]:
        """
        Classifies the door state from the center third of the frame.
        Returns "Open", "Closed", "Semi", or None if no door model loaded.
        """
        if self.door_model is None:
            return None
        w = frame.shape[1]
        x1, x2 = w // 3, 2 * w // 3
        crop = frame[:, x1:x2]
        results = self.door_model(crop, verbose=False, device=self.device)
        top = results[0].probs.top1
        label = results[0].names[top]
        conf = float(results[0].probs.top1conf)
        return label if conf >= 0.6 else None

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the frame.
        Useful for debugging -- save the result or cv2.imshow() it.
        """
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            label = f"{d.label} {d.confidence:.0%}"
            if d.distance_m is not None:
                label += f" ~{d.distance_m:.1f}m"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2
            )
        return annotated


def assign_distances(
    detections: list[Detection],
    distances: dict[str, Optional[float]],
    frame_width: int,
) -> list[Detection]:
    """
    Assigns sensor distances to detections based on horizontal position.
    Each detection's bbox center is mapped to the left/front/right sensor zone.
    Mutates detections in place and returns them.
    """
    third = frame_width / 3
    for d in detections:
        cx = (d.bbox[0] + d.bbox[2]) / 2
        if cx < third:
            d.distance_m = distances.get("left")
        elif cx < 2 * third:
            d.distance_m = distances.get("front")
        else:
            d.distance_m = distances.get("right")
    return detections


# -----QUICK TEST-----

if __name__ == "__main__":
    import urllib.request

    # Download a sample street scene if no test image exists alr
    TEST_IMAGE = "test_scene.jpg"
    if not os.path.exists(TEST_IMAGE):
        print("Downloading test image...")
        urllib.request.urlretrieve(
            "https://ultralytics.com/images/bus.jpg",
            TEST_IMAGE,
        )

    detector = ObjectDetector()
    detections = detector.detect_from_path(TEST_IMAGE)

    print(f"\nFound {len(detections)} object(s):\n")
    for d in detections:
        print(f"  {d.label:<15} conf={d.confidence:.0%}  bbox={d.bbox}")

    # Save annotated image
    frame = cv2.imread(TEST_IMAGE)
    annotated = detector.annotate(frame, detections)
    out_path = "test_scene_annotated.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")
