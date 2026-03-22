"""
ai/language_generation.py

Turns a SceneSummary into a natural spoken alert string.
Tries the Claude API first for warm, companion-style responses.
If the API is too slow or fails, silently falls back to templates
so an alert always gets spoken in real time.
"""

import time
import threading
from typing import Optional
from scene_interpretation import SceneSummary, ObstacleSummary
from dotenv import load_dotenv
load_dotenv()


# ----- LLM personality prompt -----

SYSTEM_PROMPT = """You are a warm, concise voice assistant built into a smart walker 
for elderly people at home. Your job is to alert the user to nearby obstacles.

Rules:
- Sound warm and natural, not robotic
- Always mention what the object is and where it is (left, center, or right)
- If distance is given, include it naturally
- For critical urgency, be direct and clear -- safety comes first
- Never say "I detected" or "Object found" -- speak like a caring companion

Examples of good responses:
- "Careful, there's a chair just ahead on your left."
- "Someone's coming toward you from the right."
- "Watch out -- a chair is very close on your left."
- "Your cat is nearby on the right, about a meter away."
- "The path looks clear, you're good to go."
"""


# ----- FALLBACK TEMPLATES -----
# Used when the LLM times out or fails.

TEMPLATES: dict[tuple, str] = {
    ("critical", True):  "Stop. There is a {label} on your {position}, {distance}.",
    ("critical", False): "Stop. There is a {label} very close on your {position}.",
    ("high",     True):  "Careful, {label} on your {position}, {distance}.",
    ("high",     False): "Careful, {label} on your {position}.",
    ("medium",   True):  "There is a {label} ahead on your {position}, {distance}.",
    ("medium",   False): "There is a {label} on your {position}.",
    ("low",      True):  "There is a {label} nearby, {distance}.",
    ("low",      False): "{label} detected.",
}

CLEAR_PATH_MESSAGE = "The path looks clear, you're good to go."


# ----- COOLDOWN TRACKER -----

class AlertCooldown:
    def __init__(self, cooldown_seconds: float = 4.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken: dict[str, float] = {}

    def should_speak(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_spoken.get(key, 0.0)
        if now - last >= self.cooldown_seconds:
            self._last_spoken[key] = now
            return True
        return False

    def reset(self):
        self._last_spoken.clear()


# ----- LLM CALLER -----

import time

# Global variable to track the last time we actually hit the internet
_LAST_API_CALL_TIME = 0.0
API_MIN_INTERVAL = 4.0 # Force at least 4 seconds between ANY network calls

def _call_llm(prompt: str, timeout: float = 5.0) -> Optional[str]:
    import urllib.request
    import urllib.error
    import json
    import os
    
    global _LAST_API_CALL_TIME
    
    # 1. Physical Throttle: Don't even try if we just called it
    now = time.monotonic()
    if now - _LAST_API_CALL_TIME < API_MIN_INTERVAL:
        #print("[DEBUG] Throttled -- skipping API call")
        return None 
    
    #print("[DEBUG] Attempting API call...")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
    
    payload = json.dumps({
        "contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
        "generationConfig": {"maxOutputTokens": 500}
    }).encode()

    try:
        _LAST_API_CALL_TIME = now # Update timestamp before the call
        req = urllib.request.Request(
            url, data=payload, 
            headers={"Content-Type": "application/json"}, 
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            #rint(f"[DEBUG] Raw response: {data}")
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("[LanguageGenerator] 429 Error: Rate limit reached. Cooling down...")
            # Optional: Increase the global interval temporarily
            _LAST_API_CALL_TIME += 10 
            #print(f"[DEBUG] HTTP Error {e.code}: {e.read().decode()}")
        return None
    except Exception as e:
        print(f"[LanguageGenerator] Connection Error: {e}")
        return None


# ----- GENERATOR -----

class LanguageGenerator:
    """
    Converts a SceneSummary into a spoken alert string.
    Uses Claude API for natural companion-style responses,
    falls back to templates silently if LLM is too slow or unavailable.

    Args:
        cooldown_seconds:    Min seconds before repeating the same alert.
        speak_clear_path:    Whether to announce when the path is clear.
        clear_path_cooldown: Separate longer cooldown for clear path messages.
        use_llm:             Set to False to always use templates (faster, offline).
        llm_timeout:         Max seconds to wait for LLM before falling back.
    """

    def __init__(
        self,
        cooldown_seconds: float    = 4.0,
        speak_clear_path: bool     = True,
        clear_path_cooldown: float = 10.0,
        use_llm: bool              = True,
        llm_timeout: float         = 1.5,
    ):
        self.obstacle_cooldown = AlertCooldown(cooldown_seconds)
        self.clear_cooldown    = AlertCooldown(clear_path_cooldown)
        self.use_llm           = use_llm
        self.llm_timeout       = llm_timeout

    def generate(self, scene: SceneSummary) -> Optional[str]:
        if scene.is_clear:
            if self.clear_cooldown.should_speak("clear"):
                return CLEAR_PATH_MESSAGE
            return None

        if not self.obstacle_cooldown.should_speak("scene"):
            return None

        if self.use_llm:
            llm_result = _call_llm(self._build_prompt(scene), self.llm_timeout)
            if llm_result:
                return llm_result

        # fallback -- just speak the top obstacle
        alerts = []
        for obs in scene.obstacles:
            alerts.append(self._render_template(obs))
        return " ".join(alerts)

    def _build_prompt(self, scene: SceneSummary) -> str:
        """Build a prompt describing the full scene."""
        lines = []
        for obs in scene.obstacles:
            dist = f"{obs.distance_m:.1f} meters away" if obs.distance_m else "unknown distance"
            lines.append(f"- {obs.label} on the {obs.position}, {dist}, urgency: {obs.urgency}")
        obstacles_text = "\n".join(lines)
        return (
            f"Describe the surrounding obstacles to the user in one warm, natural sentence:\n"
            f"{obstacles_text}\n"
            f"You MUST mention each obstacle, its position, and distance."
        )

    def _render_template(self, obs: ObstacleSummary) -> str:
        """Fallback template renderer."""
        has_distance = obs.distance_m is not None
        template = TEMPLATES.get((obs.urgency, has_distance), "{label} detected.")

        distance_str = ""
        if has_distance:
            d = obs.distance_m
            if d < 1.0:
                distance_str = "less than one meter away"
            elif d < 2.0:
                distance_str = f"about {d:.1f} meters away"
            else:
                distance_str = f"{d:.0f} meters away"

        return template.format(
            label=obs.label,
            position=obs.position,
            distance=distance_str,
        ).strip()


# ----- QUICK TEST -----

if __name__ == "__main__":
    from scene_interpretation import SceneInterpreter, SceneSummary
    from object_detection import Detection

    mock_detections = [
        Detection(label="chair",  confidence=0.91, bbox=(50,  300, 200, 500), distance_m=0.7),
        Detection(label="person", confidence=0.88, bbox=(300, 100, 500, 600), distance_m=1.5),
        Detection(label="cat",    confidence=0.76, bbox=(550, 400, 650, 520), distance_m=0.9),
        Detection(label="couch",  confidence=0.82, bbox=(400, 200, 700, 500), distance_m=3.0),
    ]

    interpreter = SceneInterpreter(frame_width=809)
    scene       = interpreter.interpret(mock_detections)

    print("=== WITH LLM ===")
    gen_llm = LanguageGenerator(cooldown_seconds=0, use_llm=True)
    print(f'  -> "{gen_llm.generate(scene)}"\n')

    print("=== WITHOUT LLM (template fallback) ===")
    gen_template = LanguageGenerator(cooldown_seconds=0, use_llm=False)
    gen_template.obstacle_cooldown.reset()
    print(f'  -> "{gen_template.generate(scene)}"')

    print("\n=== clear path ===")
    empty = SceneSummary(obstacles=[])
    print(f'  -> "{gen_template.generate(empty)}"')

    print("\n=== cooldown test ===")
    gen_cd = LanguageGenerator(cooldown_seconds=3.0, use_llm=False)
    print(f'  First call:  "{gen_cd.generate(scene)}"')
    print(f'  Second call: "{gen_cd.generate(scene)}"  <- None means cooldown blocked it')