import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vosk-model-small-en-us-0.15"
)

_model = None

def _get_model() -> Model:
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model not found at {_MODEL_PATH}. "
                "Download it from https://alphacephei.com/vosk/models"
            )
        print("[voice_input] Loading Vosk model...")
        _model = Model(_MODEL_PATH)
        print("[voice_input] Vosk model ready.")
    return _model


def listen() -> str:
    """
    Record from the default microphone until a full phrase is detected.
    Returns the transcribed text, or "" if nothing was understood.
    Fully offline — no internet required.
    """
    rec = KaldiRecognizer(_get_model(), 16000)
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000,
        )
        stream.start_stream()
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"[voice_input] Heard: {text}")
                    return text
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
