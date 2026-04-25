# Wiring Guide — Microphone + MonkMakes Amplified Speaker 2

This guide covers how to physically connect both the microphone and the
MonkMakes Amplified Speaker 2 to your Raspberry Pi so the voice navigation
system can listen for destinations and speak directions aloud.

---

## 1. MonkMakes Amplified Speaker 2

### What it does
A small amplified speaker board designed for Raspberry Pi and Arduino.  
It takes a standard audio signal (3.5mm AUX or header-pin line-in) and
amplifies it to fill a room — perfect for directing a walker user.

### Power connections (female-to-female jumper wires)

| Speaker board pin | Pi GPIO pin | Wire colour (convention) |
|-------------------|-------------|--------------------------|
| **5V**            | Pin 2 or 4  | Red                      |
| **GND**           | Pin 6 or 9  | Black                    |

> The board accepts 3.3 V–6 V; 5 V from the Pi gives the loudest output.  
> Max current draw is ~300 mA, well within the Pi's rail limit.

### Audio input — choose ONE method

#### Option A — 3.5mm AUX cable (Raspberry Pi 1, 2, 3, 4)
1. Connect a **3-pole or 4-pole 3.5mm AUX cable** from the Pi's headphone
   jack to the MonkMakes speaker's 3.5mm socket.
2. Run the following command once at boot to force audio to the jack
   (not HDMI):
   ```bash
   amixer cset numid=3 1
   ```
   The `nav_voice_system.py` script calls this automatically via
   `configure_audio_output()` at startup.

#### Option B — Header pins (Raspberry Pi 5, Pico, or Arduino)
The Pi 5 has no 3.5mm jack; use the speaker's **LINE IN** header pin with
a USB audio adapter or PWM audio GPIO output instead.

| Speaker header | Connect to                              |
|----------------|-----------------------------------------|
| LINE IN        | USB audio adapter's output, or PWM GPIO |
| GND            | Same GND as above                       |

---

## 2. USB Microphone

Any USB Audio Class (UAC) microphone works with Raspberry Pi — no extra
drivers needed.

### Connection
Plug the USB microphone into any of the Pi's USB ports.

### Verify the mic is detected
```bash
arecord -l
```
You should see an entry like `card 1: Device [USB PnP Sound Device]`.

### Set USB mic as the default audio input
Create or edit `/etc/asound.conf` on the Pi:

```
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:0,0"   # Built-in audio out → 3.5mm jack → MonkMakes speaker
    }
    capture.pcm {
        type plug
        slave.pcm "hw:1,0"   # USB microphone input
    }
}

ctl.!default {
    type hw
    card 0
}
```

> **Check your card numbers** with `aplay -l` (output) and `arecord -l`
> (input) — the numbers may differ on your system.  
> Swap `hw:0,0` / `hw:1,0` if the mic and speaker are detected in a
> different order.

### Test microphone capture
```bash
arecord -D hw:1,0 -f cd test.wav   # record 5 seconds
aplay test.wav                      # play back through speaker
```

---

## 3. Full breadboard / GPIO reference diagram

```
Raspberry Pi 4                MonkMakes Amp Speaker 2
─────────────────             ──────────────────────────
 Pin 2  (5V)  ──── red ────► 5V
 Pin 6  (GND) ── black ───► GND
 3.5mm jack   ── aux ──────► 3.5mm socket
                                   │
                               [amplifier]
                                   │
                              [speaker cone]

 USB port     ── USB mic ────────────────────────────────►  [records user voice]
```

---

## 4. Software setup on the Pi

### Install system packages
```bash
sudo apt update
sudo apt install espeak-ng portaudio19-dev -y
```

### Install Python packages
```bash
pip install pyttsx3 vosk pyaudio
```

### Download the Vosk speech model (offline STT)
```bash
cd /path/to/ai-smart-walker
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

---

## 5. Running the voice navigation system

From the project root:
```bash
python -m voice_navigation.nav_voice_system
```

Or directly:
```bash
python voice_navigation/nav_voice_system.py
```

The system will:
1. Route audio to the 3.5mm jack.
2. Announce the available rooms.
3. Ask "Where would you like to go?" through the speaker.
4. Listen via the USB mic (Vosk offline recognition).
5. Speak turn-by-turn directions through the MonkMakes speaker.

Say **"stop"** or **"goodbye"** to end the session.

---

## 6. Available rooms (default map)

| Room         | Grid position |
|--------------|---------------|
| entrance     | (4, 4)        |
| kitchen      | (2, 2)        |
| bedroom      | (2, 8)        |
| living room  | (6, 2)        |
| office       | (6, 7)        |
| bathroom     | (8, 9)        |

To add or change rooms, edit `maps/house_map.json`.
