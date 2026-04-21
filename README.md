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

