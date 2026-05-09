"""
ai/camera_stream.py


THIS IS THE MAIN PIPELINE.


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
import threading
import pyttsx3

# fail early with a clear message if not on Pi
try:
    from picamera2 import Picamera2
except ImportError:
    print("[camera_stream] picamera2 not found -- this script only runs on Raspberry Pi.")
    print("                Use pipeline_test.py for testing on your dev machine.")
    sys.exit(1)

from object_detection import ObjectDetector, assign_distances
from scene_interpretation import SceneInterpreter
from language_generation import LanguageGenerator
from ultrasonic import UltrasonicArray
from navigation import Navigator
from voice_input import listen


# ----- CONFIG -----

FRAME_WIDTH  = 640          # capture resolution width
FRAME_HEIGHT = 480          # capture resolution height
LOOP_INTERVAL = 0.5         # seconds between pipeline runs (2 fps processing)
DOOR_COOLDOWN = 20.0        # seconds between repeated door announcements
SENSOR_WARN_COOLDOWN = 30.0 # seconds between repeated sensor failure warnings
REPEAT_CUES   = ["repeat", "say that again", "what did you say", "come again", "pardon", "say again"]

DOOR_MESSAGES = {
    "Open": "The door ahead is open, go through.",
    "Semi": "The door ahead is partially open, push it open to continue.",
    "Closed": "The door ahead is closed, please open it to continue.",
}


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

    tts = pyttsx3.init() # init tts 
    tts.setProperty("rate", 150)
    tts_lock = threading.Lock()
    last_spoken = [""]

    def speak(text: str):
        last_spoken[0] = text
        print(f"[SPEAK] {text}")
        with tts_lock:
            tts.say(text)
            tts.runAndWait()

    cam = init_camera() #init camera
    sensors = UltrasonicArray()

    # UNCOMMENT TO ENABLE STAIRS DETECTION. and replace
    # detector = ObjectDetector(stairs_model_path="best.pt", device="cpu") 

    detector = ObjectDetector(device="cpu")
    interpreter = SceneInterpreter(frame_width=FRAME_WIDTH)
    generator = LanguageGenerator(
        cooldown_seconds=4.0,
        speak_clear_path=True,
        clear_path_cooldown=10.0,
        use_llm=use_llm,
    )
    navigator = Navigator()

    # ----- VOICE + NAVIGATION THREAD -----
    def voice_thread():
        speak("Hello! Before we get started, please tell me where you are right now.")
        while True:
            text = listen()
            if not text:
                continue

            if any(cue in text.lower() for cue in REPEAT_CUES):
                if last_spoken[0]:
                    speak(last_spoken[0])
                continue

            result = navigator.handle(text)
            if result:
                generator.set_navigating(navigator._current_destination is not None)
                speak(result)
                queued = generator.pop_queued_alert()
                if queued:
                    speak(queued)

    threading.Thread(target=voice_thread, daemon=True).start()

    # ----- DETECTION LOOP (main thread) -----
    last_door_spoken = 0.0
    last_sensor_warn = 0.0
    door_consecutive = {"state": None, "count": 0}  # must see same state 3x in a row
    DOOR_CONFIRM_FRAMES = 3
    try:
        while True:
            loop_start = time.monotonic()

            # 1. Capture frame
            frame_rgb = cam.capture_array()
            frame_bgr = frame_rgb[:, :, ::-1]

            # 2. Sensor distances + object detection
            distances  = sensors.read_all()
            failed = sensors.failed_sensors()
            if failed and (time.monotonic() - last_sensor_warn) >= SENSOR_WARN_COOLDOWN:
                last_sensor_warn = time.monotonic()
                names = " and ".join(failed)
                speak(f"Warning: the {names} sensor{'s are' if len(failed) > 1 else ' is'} not responding. Please proceed with extra caution.")
            detections = detector.detect(frame_bgr)
            assign_distances(detections, distances, FRAME_WIDTH)

            # 3. Door classification (only when current instruction involves a door)
            current_step = navigator.current_step()
            door_expected = current_step is not None and "door" in current_step.lower()
            if not door_expected:
                door_consecutive["state"] = None
                door_consecutive["count"] = 0
            if generator.is_navigating and door_expected:
                door_state = detector.classify_door(frame_bgr)
                if door_state == door_consecutive["state"]:
                    door_consecutive["count"] += 1
                else:
                    door_consecutive["state"] = door_state
                    door_consecutive["count"] = 1
                confirmed = (
                    door_state is not None
                    and door_consecutive["count"] >= DOOR_CONFIRM_FRAMES
                    and (time.monotonic() - last_door_spoken) >= DOOR_COOLDOWN
                )
                if confirmed:
                    last_door_spoken = time.monotonic()
                    door_consecutive["count"] = 0
                    msg = DOOR_MESSAGES.get(door_state)
                    if msg:
                        speak(msg)

            # 4. Interpret scene
            scene = interpreter.interpret(detections)

            # 5. Generate alert (respects audio priority vs navigation)
            alert = generator.generate(scene)
            if alert:
                speak(alert)

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
