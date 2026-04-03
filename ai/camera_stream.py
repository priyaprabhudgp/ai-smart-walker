"""
ai/camera_stream.py

Live camera loop for Raspberry Pi + Camera Module 3.
Captures frames from picamera2, runs the full AI pipeline,
and prints the spoken alert.

Usage:
    python camera_stream.py              # default settings
    python camera_stream.py --no-llm    # offline mode, template fallback only

Press Ctrl+C to stop.

NOTE: This file only runs on Raspberry Pi OS with picamera2 installed.
      For dev/testing on Windows use pipeline_test.py with static images.
"""

import time
import argparse
import sys
import os
import pyttsx3

# Guard: fail early with a clear message if not on Pi
try:
    from picamera2 import Picamera2
except ImportError:
    print("[camera_stream] picamera2 not found -- this script only runs on Raspberry Pi.")
    print("                Use pipeline_test.py for testing on your dev machine.")
    sys.exit(1)

from object_detection import ObjectDetector
from scene_interpretation import SceneInterpreter
from language_generation import LanguageGenerator
from ultrasonic import UltrasonicArray


# ----- CONFIG -----

FRAME_WIDTH  = 640          # capture resolution width
FRAME_HEIGHT = 480          # capture resolution height
LOOP_INTERVAL = 0.5         # seconds between pipeline runs (2 fps processing)


# ----- CAMERA SETUP -----

def init_camera() -> Picamera2:
    """Configure and start the Camera Module 3."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)  # let the CM3 autofocus initialize before first capture
    print(f"[camera_stream] Camera ready ({FRAME_WIDTH}x{FRAME_HEIGHT})")
    return picam2


# ----- MAIN LOOP -----

def run(use_llm: bool = True):
    print("[camera_stream] Starting pipeline. Press Ctrl+C to stop.\n")

    tts = pyttsx3.init()
    tts.setProperty("rate", 150)  # slightly slower than default, easier to understand

    cam        = init_camera()
    sensors    = UltrasonicArray()
    detector   = ObjectDetector(stairs_model_path="best.pt", device="cpu")
    interpreter = SceneInterpreter(frame_width=FRAME_WIDTH)
    generator  = LanguageGenerator(
        cooldown_seconds=4.0,
        speak_clear_path=True,
        clear_path_cooldown=10.0,
        use_llm=use_llm,
    )

    try:
        while True:
            loop_start = time.monotonic()

            # 1. Capture frame (RGB numpy array from picamera2)
            frame_rgb = cam.capture_array()

            # picamera2 gives RGB --> convert to BGR for YOLO/cv2 compatibility
            frame_bgr = frame_rgb[:, :, ::-1]

            # 2. Read ultrasonic distances
            distances = sensors.read_all()
            # distances = {"left": 0.42, "right": 1.1, "front": None, "down": None}

            # 3. Detect objects
            detections = detector.detect(frame_bgr)

            # Assign distance to each detection based on its horizontal position
            # detected objs gets their distance based on sensor zone
            # feeds into interpretation and urgency is set based on obj dist
            for d in detections:
                position = interpreter._horizontal_position(d.bbox)
                d.distance_m = distances.get(position)

            # 4. Interpret scene
            scene = interpreter.interpret(detections)

            # 5. Generate alert
            alert = generator.generate(scene)

            # 6. Speak alert through speaker (text to speech)
            if alert:
                print(f"[ALERT] {alert}")
                tts.say(alert)
                tts.runAndWait()

            # Throttle loop to LOOP_INTERVAL
            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[camera_stream] Stopped.")
    finally:
        cam.stop()
        sensors.cleanup()


# ----- ENTRY POINT -----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI smart walker live camera pipeline")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Gemini API calls and use template fallback only (faster, offline)"
    )
    args = parser.parse_args()

    run(use_llm=not args.no_llm)
