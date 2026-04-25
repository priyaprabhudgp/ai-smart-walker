"""
voice_output.py
---------------
Text-to-speech output for the MonkMakes Amplified Speaker 2.

Hardware setup (see WIRING_GUIDE.md for the full diagram):
  - Speaker 5V  → Pi GPIO Pin 2  (5V)
  - Speaker GND → Pi GPIO Pin 6  (GND)
  - Audio in    → Pi 3.5mm audio jack  (3-pole or 4-pole AUX cable)

Audio routing:
  On first use, call `configure_audio_output()` once to force the Pi's
  audio output to the 3.5mm jack (default is auto/HDMI).
  This calls: amixer cset numid=3 1
  (0 = auto, 1 = headphone jack, 2 = HDMI)

Dependencies (already in requirements-pi.txt):
  pip install pyttsx3
  sudo apt install espeak-ng portaudio19-dev
"""

import subprocess
import time
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio routing
# ---------------------------------------------------------------------------

def configure_audio_output() -> None:
    """
    Force Raspberry Pi audio output to the 3.5mm headphone jack.
    Run this once at startup before calling speak().
    Has no effect on non-Pi systems (error is silently swallowed).
    """
    try:
        result = subprocess.run(
            ["amixer", "cset", "numid=3", "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("[voice_output] Audio routed to 3.5mm jack.")
        else:
            logger.warning(
                "[voice_output] amixer returned non-zero — "
                "may not be running on a Pi. Output: %s", result.stderr.strip()
            )
    except FileNotFoundError:
        logger.warning("[voice_output] amixer not found — skipping audio routing.")


# ---------------------------------------------------------------------------
# TTS engine (lazy init so import is fast)
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    """Return a cached pyttsx3 engine, initialising it on first call."""
    global _engine
    if _engine is None:
        try:
            import pyttsx3  # noqa: PLC0415
            _engine = pyttsx3.init()
            # Comfortable speaking rate for a walker user
            _engine.setProperty("rate", 145)
            # Full volume through the amplified speaker
            _engine.setProperty("volume", 1.0)
            logger.info("[voice_output] pyttsx3 engine ready.")
        except Exception as exc:
            logger.error("[voice_output] Failed to init pyttsx3: %s", exc)
            raise
    return _engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str, rate: int = 145, volume: float = 1.0) -> None:
    """
    Speak a single string through the MonkMakes speaker.

    Args:
        text:   The sentence to speak.
        rate:   Words per minute (default 145 — clear for accessibility).
        volume: 0.0–1.0 (default 1.0 — full amplified output).
    """
    if not text or not text.strip():
        return

    print(f"[voice_output] ▶  {text}")
    try:
        engine = _get_engine()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:
        logger.error("[voice_output] TTS error: %s", exc)
        # Fallback: try espeak directly via subprocess
        _espeak_fallback(text)


def speak_directions(directions: List[str], pause_seconds: float = 0.6) -> None:
    """
    Speak each navigation instruction in the list, with a short pause between.

    Args:
        directions:    List of instruction strings from instruction_generator.
        pause_seconds: Gap between instructions so the user can process each one.
    """
    for instruction in directions:
        speak(instruction)
        time.sleep(pause_seconds)


def _espeak_fallback(text: str) -> None:
    """
    Fallback: speak using espeak-ng subprocess if pyttsx3 fails.
    espeak-ng is the underlying engine pyttsx3 uses on Linux/Pi.
    """
    try:
        subprocess.run(
            ["espeak-ng", "-s", "145", text],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        logger.error("[voice_output] espeak-ng not found. Install with: sudo apt install espeak-ng")
