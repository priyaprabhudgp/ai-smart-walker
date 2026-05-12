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
│   └── rawdiagrams/
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
- Next step triggers on voice command
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


## Dev Setup (Windows)

```cmd
cd C:\Users\<user>\ai-smart-walker
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

**1. SSH into Pi:**
```bash
ssh pipriya@<ip>
```

**2. Clone the repo:**
```bash
cd ~
git clone https://github.com/priyaprabhudgp/ai-smart-walker.git
cd ai-smart-walker
```

**3. Copy your `.env` file (Gemini API key — not in the repo):**
```cmd
scp "C:\Users\<user>\ai-smart-walker\.env" pipriya@<ip>:~/ai-smart-walker/
```

**4. Install dependencies:**
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

### Pushing updates to Pi

SSH in and pull:
```bash
ssh pipriya@<ip>
cd ~/ai-smart-walker
git pull origin main
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

## Notes

- When installing a new package on Windows: `venv\Scripts\python.exe -m pip freeze > requirements.txt`
- `picamera2` is a Pi system package — do not pip install it on Windows
- `best.pt` = custom stairs model (fine-tuned from yolov8n.pt)
- `door.pt` = custom door classifier (YOLOv8n-cls, 3 classes: Open / Closed / Semi)
- `yolov8n.pt` = base COCO model (persons, chairs, pets, etc. — no doors or stairs)
- Voice recognition (`voice_input.py`) uses Google Speech API — requires internet on the Pi





Default map layout:
# # # # # # # # #
. . . . . . . . .
. K . . . . . B . #   K = kitchen  B = bedroom
. . . . . . . . .
. . . E . . . . . #   E = entrance (start)
. . . . . . . . .
. L . . . . O . . #   L = living room  O = office
. . . . . . . . .
. . . . . . . . D #   D = bathroom
# # # # # # # # #

To change the map, edit `maps/house_map.json`. To add a room, add a new number to the grid and a matching entry in the `legend` object.

---

## Obstacle Detection

- Detects 80 object classes (people, chairs, pets, furniture, etc.) via YOLOv8n
- Custom stairs model (`best.pt`) adds stair detection
- Urgency is determined by ultrasonic sensor distance:

| Distance | Urgency | Behaviour |
|---|---|---|
| ≤ 0.8m | Critical | Stop immediately |
| ≤ 2.0m | High | Slow down warning |
| ≤ 4.0m | Medium | Heads-up alert |
| > 4.0m | Low | Suppressed during navigation |

Alerts use the Gemini LLM for natural speech; falls back to pre-written templates if offline or slow.

### Door Classification
`door.pt` classifies the centre of the camera frame every cycle during navigation:
- **Open** → tells the user to go through
- **Closed / Semi** → warns the user to open the door first

---

## Voice Commands

| Say... | What happens |
|---|---|
| *"Take me to the kitchen"* | Plans a route and speaks directions |
| *"I'm in the kitchen"* | Updates your current position |
| *"Next" / "Continue"* | Moves to the next navigation step |
| *"Repeat that" / "Say that again"* | Replays the last spoken message |
| *"Where am I?"* | States your current position |
| *"Where am I going?"* | States your current destination |
| *"Cancel navigation"* | Cancels the active route |
| *"Stop" / "Goodbye"* | Ends the voice navigation session |

---

## Dev Setup (Mac / Windows)

```bash
cd ai-smart-walker
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Test the obstacle detection pipeline on a static image:
```bash
cd ai
python pipeline_test.py
```

Test voice navigation:
```bash
python -m voice_navigation.nav_voice_system
```

---

## Raspberry Pi Deployment

### 1. Clone the repo

```bash
cd ~
git clone https://github.com/priyaprabhudgp/ai-smart-walker.git
cd ai-smart-walker
```

### 2. Install system packages

```bash
sudo apt update
sudo apt install espeak-ng python3-pyaudio python3-pip python3-venv -y
```

### 3. Set up a virtual environment

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 4. Install Python packages

```bash
pip install pyttsx3 vosk
```

For the full camera + detection pipeline:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pi.txt
```

### 5. Download the Vosk speech model

```bash
cd ~/ai-smart-walker
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### 6. Configure audio

Find your USB mic card number:
```bash
arecord -l
```

Create `/etc/asound.conf` (replace `hw:3,0` with your actual mic card number):
pcm.!default {
type asym
playback.pcm {
type plug
slave.pcm "hw:0,0"
}
capture.pcm {
type plug
slave.pcm "hw:3,0"
}
}
ctl.!default {
type hw
card 0
}

Force audio output to the 3.5mm jack (not HDMI):
```bash
sudo raspi-config
# → System Options → Audio → Headphones
```

### 7. Test hardware

```bash
# Test speaker
speaker-test -t wav -c 1

# Test microphone (speak for 3 seconds, then hear playback)
arecord -D hw:3,0 -f cd -d 3 test.wav && aplay test.wav
```

### 8. Run

**Voice navigation only (mic + map + speaker):**
```bash
cd ~/ai-smart-walker
source venv/bin/activate
python -m voice_navigation.nav_voice_system
```

**Full pipeline (camera + object detection + navigation):**
```bash
cd ~/ai-smart-walker/ai
source ~/ai-smart-walker/venv/bin/activate
python camera_stream.py

# Offline mode (no Gemini API needed):
python camera_stream.py --no-llm
```

Press `Ctrl+C` to stop.

### 9. Pull updates from GitHub

```bash
cd ~/ai-smart-walker
git pull origin main
```

---

## Setting Up a New House Map

### Step 1 — Draw the floor plan
Sketch the house on paper or in a drawing tool. Mark every room, wall, and doorway. See `maps/rawdiagrams/layout1.pdf` as an example.

### Step 2 — Generate the JSON with an LLM
Feed the floor plan image to Claude or Gemini with this prompt:
Here is a floor plan. Convert it into this JSON grid format:
{
"grid": [[0,0,...], ...],
"legend": {
"0": "blocked",
"1": "walkable",
"2": "kitchen",
...
}
}
Rules:

0 = wall/outside, 1 = open walkable floor
Each named room gets its own number (2, 3, 4...)
The room's number appears once in the grid at that room's centre cell
Every room must be reachable by walking through 1-cells


### Step 3 — Save and test
Save the file as `maps/house_map.json` and run:
```bash
python -m voice_navigation.nav_voice_system
```

### Step 4 — Update the start position
Edit `DEFAULT_START_POSITION` in `voice_navigation/nav_voice_system.py` to match where the user will be standing when the system starts.

---

## Notes

- `picamera2` is a Pi system package — do not pip install it
- `best.pt` — custom stairs model (fine-tuned from yolov8n.pt)
- `door.pt` — custom door classifier (YOLOv8n-cls, 3 classes: Open / Closed / Semi)
- `yolov8n.pt` — base COCO model (persons, chairs, pets, etc.)
- Voice recognition uses Vosk — fully offline, no internet required
- The ALSA warnings printed at startup are harmless — PyAudio probes all audio drivers on init
