# AI Smart Walker

An AI-powered smart walker that detects indoor obstacles and speaks natural alerts to the user via a built-in voice assistant.

**Pipeline:** Camera Module 3 → YOLOv8n object detection → scene interpretation → Gemini 2.5 LLM → spoken alert via speaker

---

## Project Structure

```
ai-smart-walker/
├── ai/
│   ├── object_detection.py      # YOLOv8n wrapper + custom stairs model
│   ├── scene_interpretation.py  # priority ranking, urgency, position
│   ├── language_generation.py   # Gemini API + template fallback
│   ├── camera_stream.py         # live Pi camera loop (runs on Pi only)
│   ├── pipeline_test.py         # end-to-end test with static images
│   ├── ultrasonic.py            # 4x HC-SR04 sensor array (left/right wired, front/down pending)
│   └── best.pt                  # custom trained stairs detection model
├── ios_app/                     # archived iOS app (not in use)
├── requirements.txt             # Windows dev dependencies
├── requirements-pi.txt          # Raspberry Pi deployment dependencies
└── .env                         # GEMINI_API_KEY (never commit this)
```

---

## Team

| Role | Responsibility |
|---|---|
| ML Pipeline | object detection, scene interpretation, LLM alerts |
| Hardware | ultrasonic sensors, GPIO, physical walker mounting |
| iOS | archived |

---

## Dev Setup (Windows)

```cmd
cd C:\Users\aylia\VScode\ai-smart-walker
venv\Scripts\activate
<<<<<<< HEAD
=======

for pi: 

source ~/ai-smart-walker/venv/bin/activate

cd ~/ai-smart-walker/ai

# First time only (or after pulling new changes) 
*WINDOWS*: 

>>>>>>> a17b23bfe6dd658812c0924921fc4a9bf1044db9
pip install -r requirements.txt
```

Test the pipeline on a static image:
```cmd
cd ai
python pipeline_test.py
```

---

## Raspberry Pi Deployment

### First time only

**1. Copy files to Pi (from Windows):**
```cmd
scp -r "C:\Users\aylia\VScode\ai-smart-walker\ai" pipriya@192.168.0.145:~/ai-smart-walker/
scp "C:\Users\aylia\VScode\ai-smart-walker\requirements-pi.txt" pipriya@192.168.0.145:~/ai-smart-walker/
scp "C:\Users\aylia\VScode\ai-smart-walker\.env" pipriya@192.168.0.145:~/ai-smart-walker/
```

**2. SSH into Pi:**
```bash
ssh pipriya@192.168.0.145
```

**3. Install dependencies:**
```bash
sudo apt update && sudo apt install espeak espeak-ng python3-dev
python3 -m venv --system-site-packages ~/ai-smart-walker/venv
echo "source ~/ai-smart-walker/venv/bin/activate" >> ~/.bashrc
source ~/ai-smart-walker/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.26.4
pip install -r ~/ai-smart-walker/requirements-pi.txt
```

### Running the pipeline

```bash
cd ~/ai-smart-walker/ai

# Full pipeline with Gemini LLM (needs internet + speaker)
python camera_stream.py

# Offline mode — template fallback, no speaker/internet needed
python camera_stream.py --no-llm
```

Press `Ctrl+C` to stop.

### Pushing code updates to Pi (from Windows)

```cmd
scp -r "C:\Users\aylia\VScode\ai-smart-walker\ai" pipriya@192.168.0.145:~/ai-smart-walker/
```

---

## Hardware Checklist

- [ ] Camera Module 3 ribbon cable connected
- [ ] USB speaker plugged in (on order)
- [ ] USB microphone plugged in (on order)
- [x] Left ultrasonic sensor wired (TRIG=23, ECHO=24)
- [x] Right ultrasonic sensor wired (TRIG=17, ECHO=27)
- [ ] Front ultrasonic sensor (hardware pending — update pins in `ai/ultrasonic.py`)
- [ ] Down ultrasonic sensor (hardware pending — update pins in `ai/ultrasonic.py`)
- [ ] Pi connected to WiFi
- [ ] `.env` present with `GEMINI_API_KEY`

---

## Environment Notes

- `.env` is gitignored — never commit it
- When installing a new package on Windows: `pip freeze > requirements.txt`
- `picamera2` is a Pi system package — do not pip install it on Windows
- `best.pt` is the custom stairs model (fine-tuned from yolov8n.pt)
- `yolov8n.pt` handles all other COCO classes (person, chair, dog, etc.)
