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


# ----- LLM personality prompt -----

SYSTEM_PROMPT = """You are a warm, concise voice assistant built into a smart walker 
for elderly people at home. Your job is to alert the user to nearby obstacles.

Rules:
- Keep responses to one short sentence (under 12 words)
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

def _call_llm(prompt: str, timeout: float = 1.5) -> Optional[str]:
    import urllib.request
    import json
    import os

    result = [None]
    error  = [None]

    def call():
        try:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            payload = json.dumps({
                "system_instruction": {
                    "parts": [{"text": SYSTEM_PROMPT}]
                },
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": 60,
                }
            }).encode()

            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                result[0] = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if result[0]:
        return result[0]
    if error[0]:
        print(f"[LanguageGenerator] LLM error: {error[0]} -- using fallback")
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
        """
        Returns a string to speak, or None if nothing should be spoken right now.
        Always speaks the single highest priority obstacle only.
        """
        if scene.is_clear:
            if self.clear_cooldown.should_speak("clear"):
                return CLEAR_PATH_MESSAGE
            return None

        top = scene.top_obstacle
        if top is None:
            return None

        key = f"{top.label}:{top.position}"
        if not self.obstacle_cooldown.should_speak(key):
            return None

        return self._generate_alert(top)

    def generate_all(self, scene: SceneSummary) -> list[str]:
        """
        Generate alerts for ALL obstacles in the scene (cooldown-filtered).
        Returns a list -- speak them in order, most urgent first.
        """
        if scene.is_clear:
            msg = self.generate(scene)
            return [msg] if msg else []

        alerts = []
        for obs in scene.obstacles:
            key = f"{obs.label}:{obs.position}"
            if self.obstacle_cooldown.should_speak(key):
                alerts.append(self._generate_alert(obs))
        return alerts

    def _generate_alert(self, obs: ObstacleSummary) -> str:
        """Try LLM first, fall back to template silently."""
        if self.use_llm:
            llm_result = _call_llm(self._build_prompt(obs), self.llm_timeout)
            if llm_result:
                return llm_result

        return self._render_template(obs)

    def _build_prompt(self, obs: ObstacleSummary) -> str:
        """Build the prompt sent to the LLM."""
        dist = (
            f"{obs.distance_m:.1f} meters away"
            if obs.distance_m else "unknown distance"
        )
        return (
            f"Alert the user about this obstacle:\n"
            f"Object: {obs.label}\n"
            f"Position: {obs.position}\n"
            f"Distance: {dist}\n"
            f"Urgency: {obs.urgency}\n"
            f"Generate one short spoken alert."
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
    alert = gen_llm.generate(scene)
    print(f'  -> "{alert}"\n')

    print("=== WITHOUT LLM (template fallback) ===")
    gen_template = LanguageGenerator(cooldown_seconds=0, use_llm=False)
    gen_template.obstacle_cooldown.reset()
    alerts = gen_template.generate_all(scene)
    for a in alerts:
        print(f'  -> "{a}"')

    print("\n=== clear path ===")
    empty = SceneSummary(obstacles=[])
    print(f'  -> "{gen_template.generate(empty)}"')

    print("\n=== cooldown test ===")
    gen_cd = LanguageGenerator(cooldown_seconds=3.0, use_llm=False)
    print(f'  First call:  "{gen_cd.generate(scene)}"')
    print(f'  Second call: "{gen_cd.generate(scene)}"  <- None means cooldown blocked it')