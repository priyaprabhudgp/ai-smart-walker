# AI Smart Walker

An AI-powered smart walker that helps elderly users navigate indoor spaces safely. Built on a Raspberry Pi with a camera, ultrasonic sensors, and a speaker — it detects obstacles, classifies doors, and guides users room-to-room with spoken instructions.

```
Camera + Microphone
    → YOLOv8 object detection + ultrasonic distance sensing + door classification
    → Scene interpretation (urgency ranking)
    → Obstacle alerts + step-by-step navigation
    → Spoken output through speaker
```

---

## Features

**Obstacle Detection** — Detects 80 object classes via YOLOv8n plus a custom stair detection model. Urgency is determined by ultrasonic distance and alerts are spoken immediately at critical and high urgency, queued or suppressed at lower urgency during navigation.

| Distance | Urgency | Behaviour |
|---|---|---|
| ≤ 0.8m | Critical | Stop immediately |
| ≤ 2.0m | High | Slow down warning |
| ≤ 4.0m | Medium | Heads-up alert |
| > 4.0m | Low | Suppressed during navigation |

**Door Classification** — A custom `door.pt` classifier reads the camera frame every cycle during navigation and tells the user whether the door ahead is open, closed, or partially open.

**Step-by-Step Navigation** — On boot the system asks where the user is. They say their destination and receive turn-by-turn instructions one step at a time via BFS over a pre-mapped JSON floor plan. The system announces arrival and tracks position automatically.

**Voice Commands**

| Say... | What happens |
|---|---|
| *"I'm in the kitchen"* | Updates current position |
| *"Take me to the bathroom"* | Plans a route and speaks the first step |
| *"Next" / "Continue"* | Advances to the next step |
| *"Repeat that" / "Say that again"* | Replays the last spoken message |
| *"Where am I?"* | States current position |
| *"Where am I going?"* | States current destination |
| *"Cancel navigation"* | Cancels the active route |

---

## Project Structure

```
ai-smart-walker/
├── ai/
│   ├── camera_stream.py         # main pipeline loop — Raspberry Pi only
│   ├── object_detection.py      # YOLOv8n wrapper + stairs model + door classifier
│   ├── scene_interpretation.py  # urgency ranking and obstacle prioritisation
│   ├── language_generation.py   # Gemini API alerts with template fallback
│   ├── navigation.py            # BFS routing, step-by-step instructions, voice intent
│   ├── voice_input.py           # USB microphone → speech text via Vosk
│   ├── ultrasonic.py            # 4x HC-SR04 sensor array
│   ├── audit_log.py             # per-cycle JSON audit logging
│   ├── pipeline_test.py         # end-to-end test with static images (Windows)
│   ├── yolov8n.pt               # base COCO model (80 classes)
│   ├── best.pt                  # custom stairs detection model
│   └── door.pt                  # custom door classifier (Open / Closed / Semi)
├── maps/
│   ├── layout1.json             # house map for Layout 1
│   └── rawdiagrams/
│       └── layout1.pdf          # hand-drawn floor plan for Layout 1
├── WIRING_GUIDE.md              # HC-SR04 wiring with voltage dividers
├── requirements.txt             # Windows dev dependencies
└── requirements-pi.txt          # Raspberry Pi deployment dependencies
```

---

## Raspberry Pi Setup

**Clone the repo:**
```bash
ssh pipriya@<ip>
cd ~
git clone https://github.com/priyaprabhudgp/ai-smart-walker.git
cd ai-smart-walker
```

**Copy your `.env` file** (Gemini API key — not in the repo):
```cmd
scp "C:\Users\<user>\ai-smart-walker\.env" pipriya@<ip>:~/ai-smart-walker/
```

**Install system packages:**
```bash
sudo apt update && sudo apt install espeak python3-dev portaudio19-dev -y
```

**Set up the virtual environment:**
```bash
python3 -m venv --system-site-packages ~/ai-smart-walker/venv
echo "source ~/ai-smart-walker/venv/bin/activate" >> ~/.bashrc
source ~/ai-smart-walker/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
pip install -r ~/ai-smart-walker/requirements-pi.txt
pip install "numpy==1.26.4" --force-reinstall
```

> `requirements-pi.txt` pulls in opencv which upgrades numpy to 2.x, breaking picamera2. The final reinstall pins it back to 1.26.4.

**Download the Vosk speech model:**
```bash
cd ~/ai-smart-walker
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**Configure audio output to 3.5mm jack:**
```bash
sudo raspi-config
# → System Options → Audio → Headphones
```

---

## Running

```bash
cd ~/ai-smart-walker/ai

# Full pipeline with Gemini LLM
python camera_stream.py

# Offline mode — no internet required
python camera_stream.py --no-llm
```

### Autoboot

To start the walker automatically on power-on:

```bash
sudo nano /etc/systemd/system/smartwalker.service
```

```ini
[Unit]
Description=AI Smart Walker
After=local-fs.target

[Service]
User=pipriya
WorkingDirectory=/home/pipriya/ai-smart-walker/ai
ExecStart=/home/pipriya/ai-smart-walker/venv/bin/python camera_stream.py --no-llm
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable smartwalker
sudo systemctl start smartwalker
```

### Pulling updates

```bash
ssh pipriya@<ip>
cd ~/ai-smart-walker
git pull origin main
```

### Shutting down

Always shut down cleanly before unplugging — pulling power mid-write corrupts files on the SD card:

```bash
sudo shutdown now
```

Wait for the green activity LED to stop blinking before unplugging.

---

## Setting Up a New House Map

Navigation uses a pre-defined JSON map of the house stored in `maps/`. Draw a floor plan, then feed it to Claude or Gemini with this prompt:

```
Here is a floor plan of a house. Extract all rooms, connections (doors between rooms),
and walking directions into this exact JSON format:

{
  "name": "Layout Name",
  "pdf": "maps/rawdiagrams/layoutN.pdf",
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

Verify all routes with `python navigation.py` and correct any wrong turn directions before deploying. Left vs right is the most common error.

---

## Dev Setup (Windows)

```cmd
cd C:\Users\<user>\ai-smart-walker
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ai
python pipeline_test.py
```

---

## Notes

- `numpy==1.26.4` must be pinned — newer versions break picamera2 on Raspberry Pi OS Bookworm
- `picamera2` is a Pi system package — do not pip install it on Windows
- Voice input uses Vosk (fully offline) — model must be downloaded separately (see setup above)
- Voice output uses pyttsx3 + espeak
- ALSA warnings at startup are harmless
- If the camera shows "device or resource busy": `sudo pkill -f camera_stream.py`

---

## Documentation and Media

- [Project Docs](https://drive.google.com/drive/folders/1z78CIVQ6NRQ4izaFbOxU4GfkqnKC1GNC?usp=drive_link)
- [Media](https://drive.google.com/drive/folders/132ME-axQYogrenPG6eLqmw5hXbRn3p1d?usp=sharing)
