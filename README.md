# AI Smart Walker

An AI-powered smart walker that helps elderly users navigate indoor spaces using a Raspberry Pi, camera-based object detection, ultrasonic sensors, voice interaction, and spoken guidance.

**Pipeline:**
```
Camera + Microphone
    → YOLOv8 object detection + ultrasonic distance sensing + door classification
    → Scene interpretation (urgency ranking)
    → Obstacle alerts + step-by-step navigation
    → Spoken output through speaker
```

---

## Project Structure

```
ai-smart-walker/
├── ai/
│   ├── camera_stream.py         # main loop — runs on Raspberry Pi only
│   ├── object_detection.py      # YOLOv8n wrapper + stairs model + door classifier
│   ├── scene_interpretation.py  # priority ranking, urgency, position
│   ├── language_generation.py   # Gemini API alerts + template fallback + audio priority
│   ├── navigation.py            # BFS routing, step-by-step instructions, voice intent
│   ├── voice_input.py           # USB microphone → raw speech text
│   ├── ultrasonic.py            # 4x HC-SR04 sensor array
│   ├── pipeline_test.py         # end-to-end test with static images (Windows)
│   ├── yolov8n.pt               # base COCO model (80 classes — no doors, no stairs)
│   ├── best.pt                  # custom stairs detection model
│   └── door.pt                  # custom door classifier (Open / Closed / Semi)
├── maps/
│   ├── layout1.json             # house map for Layout 1
│   └── raw diagrams/
│       └── layout1.pdf          # hand-drawn floor plan for Layout 1
├── requirements.txt             # Windows dev dependencies
└── requirements-pi.txt          # Raspberry Pi deployment dependencies
```

---

## What the System Can Do

### Obstacle Detection
- Detects 80 object classes (people, chairs, pets, furniture, etc.) via YOLOv8n
- Custom stairs model (`best.pt`) adds stair detection
- Urgency is determined by ultrasonic sensor distance:
  - **≤ 0.8m** → critical (stop immediately)
  - **≤ 2.0m** → high (slow down)
  - **≤ 4.0m** → medium (heads up)
  - **> 4.0m** → low (suppressed during navigation)
- Alerts use Gemini LLM for natural speech, falls back to templates if slow/offline

### Step-by-Step Navigation
- On boot, asks: *"Where are you right now?"*
- User declares position and destination by voice
- Instructions are given **one step at a time**
- Next step triggers automatically when an open door is detected, or on voice command
- Announces arrival and updates position automatically

### Door Classification
- `door.pt` classifies the center of the frame every frame during navigation
- **Open** → tells the user the door is open and to go through
- **Closed / Semi** → warns the user to open the door before continuing
- The user manually says "next" after passing through — auto-advance is disabled since the camera cannot distinguish which specific door it is seeing

### Audio Priority During Navigation
| Urgency | Behaviour |
|---|---|
| Critical / High | Interrupts immediately |
| Medium | Queued, spoken after current instruction |
| Low | Suppressed entirely |

### Voice Commands
| What you say | Response |
|---|---|
| *"I'm in the kitchen"* | Updates your position |
| *"Take me to the bathroom"* | Starts navigation, speaks first step |
| *"Next" / "Continue"* | Advances to the next step |
| *"Repeat that"* / *"Say that again"* | Replays the last spoken message |
| *"Where am I?"* | States your current position |
| *"Where am I going?"* | States your current destination |
| *"Cancel navigation"* | Cancels the active route |

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

Draw a clear floor plan of the house (see `maps/raw diagrams/layout1.pdf` as an example):
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

Save the file as `maps/layoutN.json` and copy the floor plan PDF to `maps/raw diagrams/`.

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
- [ ] `best.pt` (stairs model) in `ai/`
- [ ] `door.pt` (door classifier) in `ai/`

---

## Custom Model Training — Doors

YOLOv8 (COCO) does **not** detect doors. `door.pt` is a custom YOLOv8n-cls classification model trained on 3 classes:

| Class | Meaning |
|---|---|
| `Open` | Door frame with door open/ajar |
| `Closed` | Door frame with door shut |
| `Semi` | Door partially open |

> **Important:** The class names in your trained model must match exactly (`Open`, `Closed`, `Semi`).
> Verify with: `YOLO("door.pt").names` in a Python shell.

### Step 1 — Find a dataset

Good sources for door images:

| Source | Notes |
|---|---|
| [Roboflow Universe](https://universe.roboflow.com) | Search "door detection" — several ready-to-use labeled datasets |
| [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) | Has a "Door" class |
| Custom photos | Photograph doors in your own home — most relevant for this use case |

**Recommended:** Start with a Roboflow dataset (already labeled), then add 50–100 photos of the actual doors in the target home for best accuracy.

### Step 2 — Train (Google Colab)

```python
# Mount Drive first so weights survive runtime disconnects
from google.colab import drive
drive.mount('/content/drive')

from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
model.train(
    data="/content/dataset",
    epochs=30,
    imgsz=224,
    batch=32,
    project="/content/drive/MyDrive/door_classifier",
    name="run1",
)
```

Download immediately after training finishes (do not close the tab):
```python
from google.colab import files
files.download('/content/drive/MyDrive/door_classifier/run1/weights/best.pt')
```

Rename the downloaded file to `door.pt` and place it in `ai/`.

### Step 3 — Integrate

`door.pt` is loaded automatically by `ObjectDetector` if present in the `ai/` folder.
No code changes needed.

---

## Notes

- When installing a new package on Windows: `venv\Scripts\python.exe -m pip freeze > requirements.txt`
- `picamera2` is a Pi system package — do not pip install it on Windows
- `best.pt` = custom stairs model (fine-tuned from yolov8n.pt)
- `door.pt` = custom door classifier (YOLOv8n-cls, 3 classes: Open / Closed / Semi)
- `yolov8n.pt` = base COCO model (persons, chairs, pets, etc. — no doors or stairs)
- Voice recognition (`voice_input.py`) uses Google Speech API — requires internet on the Pi
