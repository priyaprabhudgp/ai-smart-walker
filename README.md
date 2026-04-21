# AI Smart Walker

An AI-powered smart walker that helps elderly users navigate indoor spaces using a Raspberry Pi, camera-based object detection, ultrasonic sensors, voice interaction, and spoken guidance.

**Pipeline:**
```
Camera + Microphone
    → YOLOv8 object detection + ultrasonic distance sensing
    → Scene interpretation (urgency ranking)
    → Obstacle alerts + turn-by-turn navigation
    → Spoken output through speaker
```

---

## Project Structure

```
ai-smart-walker/
├── ai/
│   ├── camera_stream.py         # main loop — runs on Raspberry Pi only
│   ├── object_detection.py      # YOLOv8n wrapper + custom model support
│   ├── scene_interpretation.py  # priority ranking, urgency, position
│   ├── language_generation.py   # Gemini API alerts + template fallback + audio priority
│   ├── navigation.py            # BFS routing, speech intent extraction, JSON map loader
│   ├── voice_input.py           # USB microphone → raw speech text
│   ├── ultrasonic.py            # 4x HC-SR04 sensor array
│   ├── pipeline_test.py         # end-to-end test with static images (Windows)
│   ├── yolov8n.pt               # base COCO model (80 classes — no doors, no stairs)
│   └── best.pt                  # custom stairs detection model
├── maps/
│   └── layout1.json             # house map for Layout 1 (see testhouselayouts/)
├── testhouselayouts/
│   └── layout1.pdf              # hand-drawn floor plan for Layout 1
├── requirements.txt             # Windows dev dependencies
├── requirements-pi.txt          # Raspberry Pi deployment dependencies
└── .env                         # GEMINI_API_KEY (never commit this)
```

---

## Team

| Role | Responsibility |
|---|---|
| ML / Perception | Object detection, scene interpretation, spoken alerts |
| Navigation | Indoor map, route planning, voice intent, instruction generation |
| Hardware | Raspberry Pi, camera, microphone, speaker, walker mounting |

---

## Dev Setup (Windows)

```cmd
cd C:\Users\<user>\VScode\ai-smart-walker
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Test the obstacle detection pipeline on a static image:
```cmd
cd ai
python pipeline_test.py
```

Test navigation routing:
```cmd
python navigation.py
```

---

## Raspberry Pi Deployment

### First time setup

**1. Copy files to Pi (from Windows):**
```cmd
scp -r "C:\Users\<user>\VScode\ai-smart-walker\ai" pipriya@<ip>:~/ai-smart-walker/
scp -r "C:\Users\<user>\VScode\ai-smart-walker\maps" pipriya@<ip>:~/ai-smart-walker/
scp "C:\Users\<user>\VScode\ai-smart-walker\requirements-pi.txt" pipriya@<ip>:~/ai-smart-walker/
scp "C:\Users\<user>\VScode\ai-smart-walker\.env" pipriya@<ip>:~/ai-smart-walker/
```

**2. SSH into Pi:**
```bash
ssh pipriya@<ip>
```

**3. Install dependencies:**
```bash
sudo apt update && sudo apt install espeak espeak-ng python3-dev portaudio19-dev
python3 -m venv --system-site-packages ~/ai-smart-walker/venv
echo "source ~/ai-smart-walker/venv/bin/activate" >> ~/.bashrc
source ~/ai-smart-walker/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
pip install -r ~/ai-smart-walker/requirements-pi.txt
```

### Running

```bash
cd ~/ai-smart-walker/ai

# Full pipeline (Gemini LLM + voice + navigation)
python camera_stream.py

# Offline mode — template fallback only, no internet needed
python camera_stream.py --no-llm
```

Press `Ctrl+C` to stop.

### Pushing updates to Pi (from Windows)

```cmd
scp -r "C:\Users\<user>\VScode\ai-smart-walker\ai" pipriya@<ip>:~/ai-smart-walker/
scp -r "C:\Users\<user>\VScode\ai-smart-walker\maps" pipriya@<ip>:~/ai-smart-walker/
```

---

## Setting Up a House Map

Navigation requires a pre-defined map of the house stored as a JSON file in `maps/`.

### Step 1 — Draw the floor plan

Draw a clear floor plan of the house (see `testhouselayouts/layout1.pdf` as an example):
- Use **thick bold lines** for walls
- Mark every **door as a labeled gap** in the wall (`Door`)
- Label every **room** clearly
- Mark the **front door** with an arrow

### Step 2 — Generate the JSON with an LLM

Feed the floor plan image to a vision LLM (Claude or Gemini) with this prompt:

```
Here is a floor plan of a house. Extract all rooms, connections (doors between rooms),
and walking directions into this exact JSON format:

{
  "name": "Layout Name",
  "pdf": "testhouselayouts/layoutN.pdf",
  "default_start": "front_door",
  "nodes": {
    "room_a": {
      "room_b": { "instruction": "turn left through the door into Room B" }
    }
  },
  "locations": {
    "room b": "room_b",
    "room b alias": "room_b"
  }
}

Rules:
- node keys use snake_case (e.g. "master_bedroom")
- every connection must be bidirectional (if A→B exists, B→A must also exist)
- instructions must be specific turn-by-turn directions (turn left, turn right, walk forward)
- locations maps every phrase a user might say to the node key
```

### Step 3 — Verify the directions

Run the navigation test to check all routes make sense:
```cmd
cd ai
python navigation.py
```

Edit `maps/layoutN.json` to correct any wrong turn directions. Direction labels
(left vs right) are the most common error — verify them against the actual floor plan.

### Step 4 — Add to the project

Save the file as `maps/layoutN.json` and copy the floor plan PDF to `testhouselayouts/`.

---

## Hardware Checklist

- [ ] Camera Module 3 ribbon cable connected
- [ ] USB speaker plugged in
- [ ] USB microphone plugged in
- [x] Left ultrasonic sensor wired (TRIG=23, ECHO=24)
- [x] Right ultrasonic sensor wired (TRIG=17, ECHO=27)
- [ ] Front ultrasonic sensor wired (TRIG=5, ECHO=6)
- [ ] Down ultrasonic sensor wired (TRIG=13, ECHO=19)
- [ ] Pi connected to WiFi
- [ ] `.env` present with `GEMINI_API_KEY`

---

## Custom Model Training — Doors

YOLOv8 (COCO) does **not** detect doors. A custom model is needed, the same way `best.pt` was trained for stairs.

When a door is detected in the camera frame during navigation, the system can say:
- **Open door ahead** → "The door is ahead of you, go through to reach the hallway"
- **Closed door ahead** → "The door is on your left, it appears to be closed — please open it"

### Step 1 — Find a dataset

Good sources for door images:

| Source | Notes |
|---|---|
| [Roboflow Universe](https://universe.roboflow.com) | Search "door detection" — several ready-to-use labeled datasets |
| [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) | Has a "Door" class — download via `openimages` pip package |
| Custom photos | Photograph doors in your own home — most relevant for this use case |

**Recommended:** Start with a Roboflow dataset (already labeled), then add 50–100 photos of the actual doors in the target home for best accuracy.

### Step 2 — Label the data (if collecting custom photos)

Use [Roboflow](https://roboflow.com) (free tier):
1. Create a new project → Object Detection
2. Upload images
3. Draw bounding boxes around:
   - `open_door` — door frame with door open/ajar
   - `closed_door` — door frame with door shut
4. Export as **YOLOv8 format**

### Step 3 — Train

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # start from base weights
model.train(
    data="path/to/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="door_model",
)
```

Training on a GPU (Google Colab free tier works). On CPU expect several hours.

### Step 4 — Test

```python
model = YOLO("runs/detect/door_model/weights/best.pt")
results = model("test_image.jpg")
results[0].show()
```

### Step 5 — Integrate

In `ai/object_detection.py`, load the door model alongside the stairs model:
```python
self.door_model = YOLO("door_best.pt")
```

Detections with label `open_door` or `closed_door` in the center of the frame
during active navigation trigger the next route instruction.

---

## Environment Notes

- `.env` is gitignored — never commit it
- When installing a new package on Windows: `venv\Scripts\python.exe -m pip freeze > requirements.txt`
- `picamera2` is a Pi system package — do not pip install it on Windows
- `best.pt` = custom stairs model (fine-tuned from yolov8n.pt)
- `yolov8n.pt` = base COCO model (persons, chairs, pets, etc. — no doors or stairs)
