"""
ai/pipeline_test.py

End-to-end test of the full ML pipeline:
    real image → object_detection → scene_interpretation → language_generation → alert

Usage:
    python pipeline_test.py                  # runs on default test image
    python pipeline_test.py my_photo.jpg     # runs on your own image
"""

import sys
import os
import urllib.request

from object_detection import ObjectDetector
from scene_interpretation import SceneInterpreter
from language_generation import LanguageGenerator

#print(f"[DEBUG] Key from env: {os.environ.get('GEMINI_API_KEY', 'NOT FOUND')[:8]}...")
# ----- CONFIG -----

DEFAULT_TEST_IMAGE = "test_scene.jpg"
DEFAULT_IMAGE_URL  = "https://ultralytics.com/images/bus.jpg"
FRAME_WIDTH        = 809   # width of the test image in pixels


# ----- PIPELINE RUNNER -----

def run_pipeline(image_path: str, use_llm: bool = True) -> str:
    """
    Runs the full pipeline on a single image.
    Returns the final spoken alert string.
    """

    print(f"\n{'='*55}")
    print(f"  Image: {image_path}")
    print(f"{'='*55}\n")

    # Step 1: Object detection 
    print("[ Step 1 ] Running object detection...")
    detector = ObjectDetector()
    detections = detector.detect_from_path(image_path)

    if not detections:
        print("  No objects detected.\n")
        return "Path is clear."

    print(f"  Detected {len(detections)} object(s):")
    for d in detections:
        print(f"    {d.label:<15} conf={d.confidence:.0%}  bbox={d.bbox}")

    # Step 2: Scene interpretation 
    print(f"\n[ Step 2 ] Interpreting scene...")
    interpreter = SceneInterpreter(frame_width=FRAME_WIDTH)
    scene = interpreter.interpret(detections)

    if scene.is_clear:
        print("  No relevant obstacles found after filtering.")
        return "Path is clear."

    print(f"  {len(scene.obstacles)} relevant obstacle(s) after filtering:")
    for obs in scene.obstacles:
        dist = f"{obs.distance_m:.1f}m" if obs.distance_m else "no distance"
        print(f"    [{obs.urgency.upper():<8}] {obs.label:<12} @ {obs.position:<6}  {dist}  (priority {obs.priority})")

    # Step 3: Language generation 
    print(f"\n[ Step 3 ] Generating alert (use_llm={use_llm})...")
    generator = LanguageGenerator(cooldown_seconds=0, use_llm=use_llm)
    alert = generator.generate(scene)

    print(f"\n{'─'*55}")
    print(f"  FINAL ALERT: \"{alert}\"")
    print(f"{'─'*55}\n")

    return alert


# ----- SCENARIO TESTS -----

def run_scenario_tests():
    """
    Tests specific scenarios with mock data to verify pipeline logic.
    No image needed -- uses fake detections.
    """
    from object_detection import Detection
    from scene_interpretation import SceneSummary

    print(f"\n{'='*55}")
    print(f"  SCENARIO TESTS (mock data)")
    print(f"{'='*55}")

    interpreter = SceneInterpreter(frame_width=809)
    generator   = LanguageGenerator(cooldown_seconds=0, use_llm=False)

    scenarios = [
        {
            "name": "Clear path",
            "detections": [],
        },
        {
            "name": "Single person close ahead",
            "detections": [
                Detection(label="person", confidence=0.92, bbox=(300, 100, 500, 600), distance_m=0.5),
            ],
        },
        {
            "name": "Multiple indoor obstacles",
            "detections": [
                Detection(label="chair",  confidence=0.91, bbox=(50,  300, 200, 500), distance_m=0.7),
                Detection(label="person", confidence=0.88, bbox=(300, 100, 500, 600), distance_m=1.5),
                Detection(label="cat",    confidence=0.76, bbox=(550, 400, 650, 520), distance_m=0.9),
            ],
        },
        {
            "name": "Unknown objects only (should be clear)",
            "detections": [
                Detection(label="kite",       confidence=0.80, bbox=(100, 100, 200, 200), distance_m=2.0),
                Detection(label="surfboard",  confidence=0.75, bbox=(300, 200, 400, 350), distance_m=3.0),
            ],
        },
        {
            "name": "Pet on the floor",
            "detections": [
                Detection(label="dog", confidence=0.85, bbox=(200, 400, 400, 600), distance_m=1.1),
            ],
        },
        {
            "name": "Cooldown blocks repeat alert",
            "detections": [
                Detection(label="person", confidence=0.90, bbox=(300, 100, 500, 600), distance_m=2.0),
            ],
        },
    ]

    # Run cooldown test separately
    cooldown_generator = LanguageGenerator(cooldown_seconds=5.0, use_llm=False)

    for i, scenario in enumerate(scenarios):
        scene = interpreter.interpret(scenario["detections"])

        # use cooldown generator for last scenario
        gen = cooldown_generator if i == len(scenarios) - 1 else generator
        if i == len(scenarios) - 1:
            # call twice to test cooldown
            gen.generate(scene)
            alert = gen.generate(scene)
        else:
            alert = gen.generate(scene)

        status = f'"{alert}"' if alert else "None (cooldown blocked)"
        print(f"\n  [{i+1}] {scenario['name']}")
        print(f"       -> {status}")

    print()


# ----- Entry point -----

if __name__ == "__main__":

    # Get image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_IMAGE

    # Download default test image if needed
    if image_path == DEFAULT_TEST_IMAGE and not os.path.exists(DEFAULT_TEST_IMAGE):
        print("Downloading test image...")
        urllib.request.urlretrieve(DEFAULT_IMAGE_URL, DEFAULT_TEST_IMAGE)
        print("Done.\n")

    # RUN FULL PIPLINE
    run_pipeline(image_path, use_llm=True)

    # Run scenario tests with mock data
    run_scenario_tests()