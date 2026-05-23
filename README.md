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
│   ├── audit_log.py             # per-cycle JSON audit logging
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

| Distance | Urgency | Behaviour |
|---|---|---|
| ≤ 0.8m | Critical | Stop immediately |
| ≤ 2.0m | High | Slow down warning |
| ≤ 4.0m | Medium | Heads-up alert |
| > 4.0m | Low | Suppressed during navigation |

- Alerts use Gemini LLM for natural speech, falls back to templates if slow/offline

### Door Classification
- `door.pt` classifies the centre of the camera frame each cycle during navigation
- **Open** → tells the user to go through
- **Closed / Semi** → warns the user to open the door first

### Step-by-Step Navigation
- On boot, asks: *"Where are you right now?"*
- User declares position and destination by voice
- Instructions are given **one step at a time**
- Next step triggers on voice command
- Announces arrival and updates position automatically

### Audio Priority During Navigation
| Urgency | Behaviour |
|---|---|
| Critical / High / Medium | Interrupts immediately |
| Low | Suppressed entirely |

### Voice Commands
| What you say | Response |
|---|---|
| *"I'm in the kitchen"* | Updates your position |
| *"Take me to the bathroom"* | Starts navigation, speaks first step |
| *"Next" / "Continue"* | Advances to the next step |
| *"Repeat that" / "Say that again"* | Replays the last spoken message |
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

**4. Install system packages:**
```bash
sudo apt update && sudo apt install espeak espeak-ng python3-dev portaudio19-dev -y
```

**5. Set up virtual environment and install Python packages:**
```bash
python3 -m venv --system-site-packages ~/ai-smart-walker/venv
echo "source ~/ai-smart-walker/venv/bin/activate" >> ~/.bashrc
source ~/ai-smart-walker/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
pip install -r ~/ai-smart-walker/requirements-pi.txt
```

**6. Configure audio output (3.5mm jack):**
```bash
sudo raspi-config
# → System Options → Audio → Headphones
```

**7. Test hardware:**
```bash
# Test speaker
speaker-test -t wav -c 1

# Test microphone (speak for 3 seconds, then hear playback)
arecord -D hw:3,0 -f cd -d 3 test.wav && aplay test.wav
```

### Running

```bash
cd ~/ai-smart-walker/ai

# Full pipeline (Gemini LLM + voice + navigation)
python camera_stream.py

# Offline mode — no internet needed
python camera_stream.py --no-llm
```

Press `Ctrl+C` to stop.

### Autoboot (systemd)

The walker can start automatically when the Pi powers on without needing a monitor or SSH:

```bash
sudo nano /etc/systemd/system/smartwalker.service
```

Paste:
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

Enable it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable smartwalker
sudo systemctl start smartwalker
```

Check status / logs:
```bash
sudo systemctl status smartwalker
journalctl -u smartwalker -f
```

Stop temporarily (survives reboot):
```bash
sudo systemctl stop smartwalker
```

Disable autoboot permanently:
```bash
sudo systemctl disable smartwalker
```

### Pulling updates from GitHub

```bash
ssh pipriya@<ip>
cd ~/ai-smart-walker
git pull origin main
```

---

## Setting Up a House Map

Navigation requires a pre-defined map of the house stored as a JSON file in `maps/`.

### Step 1 — Draw the floor plan

Draw a clear floor plan of the house (see `maps/rawdiagrams/layout1.pdf` as an example):
- Use **thick bold lines** for walls
- Mark every **door as a labeled gap** in the wall
- Label every **room** clearly

### Step 2 — Generate the JSON with an LLM

Feed the floor plan image to Claude or Gemini with this prompt:

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

### Step 3 — Verify the directions

```cmd
cd ai
python navigation.py
```

Edit `maps/layoutN.json` to correct any wrong directions. Left vs right is the most common error — verify against the actual floor plan.

### Step 4 — Save the file

Save as `maps/layoutN.json` and add the floor plan PDF to `maps/rawdiagrams/`.

---

## Notes

- When installing a new package on Windows: `venv\Scripts\python.exe -m pip freeze > requirements.txt`
- `picamera2` is a Pi system package — do not pip install it on Windows
- `best.pt` — custom stairs model (fine-tuned from yolov8n.pt)
- `door.pt` — custom door classifier (YOLOv8n-cls, 3 classes: Open / Closed / Semi)
- `yolov8n.pt` — base COCO model (persons, chairs, pets, etc. — no doors or stairs)
- Voice recognition uses Google Speech API — requires internet on the Pi
- The ALSA warnings printed at startup are harmless — PyAudio probes all audio drivers on init
- If the camera shows "device busy" on startup, a previous process wasn't stopped cleanly: `sudo pkill -f camera_stream.py`

##Documentation and Media

- Docs: https://drive.google.com/drive/folders/1z78CIVQ6NRQ4izaFbOxU4GfkqnKC1GNC?usp=drive_link 
- Media: https://drive.google.com/drive/folders/132ME-axQYogrenPG6eLqmw5hXbRn3p1d?usp=sharing
