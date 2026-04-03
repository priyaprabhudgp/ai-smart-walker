import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')  # system opencv

import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

# COCO class names
CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

def preprocess(frame, input_size=640):
    """Resize and normalize frame for YOLOv8"""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    # Pad to square
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    # Normalize and transpose to NCHW
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, nh, nw

def postprocess(outputs, scale, orig_h, orig_w, conf_thresh=0.5):
    """Parse YOLOv8 output boxes"""
    preds = outputs[0][0]  # shape: (84, 8400)
    preds = preds.T        # shape: (8400, 84)
    
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    
    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    
    mask = confidences > conf_thresh
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    detections = []
    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) / scale)
        y1 = int((cy - bh/2) / scale)
        x2 = int((cx + bw/2) / scale)
        y2 = int((cy + bh/2) / scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        detections.append((x1, y1, x2, y2, int(cls_id), float(conf)))
    
    return detections

# Load model
print("Loading YOLOv8n ONNX model...")
session = ort.InferenceSession(
    "yolov8n.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
print(f"Model loaded. Input: {input_name}")

# Setup camera
print("Starting camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()
time.sleep(2)
print("Camera ready - detecting... Ctrl+C to stop")

start = time.time()
count = 0

try:
    while True:
        frame = picam2.capture_array()
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        blob, scale, nh, nw = preprocess(frame)

        # Inference
        outputs = session.run(None, {input_name: blob})

        # Postprocess
        detections = postprocess(outputs, scale, orig_h, orig_w, conf_thresh=0.5)

        count += 1
        fps = count / (time.time() - start)

        # Print results
        if detections:
            for x1, y1, x2, y2, cls_id, conf in detections:
                label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                print(f"  {label}: {conf:.0%} [{x1},{y1},{x2},{y2}]")

        if count % 20 == 0:
            print(f"--- FPS: {fps:.1f} | Frame: {count} ---")

except KeyboardInterrupt:
    elapsed = time.time() - start
    print(f"\nStopped. Frames: {count} | Avg FPS: {count/elapsed:.1f}")
finally:
    picam2.stop()
