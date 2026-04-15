from voice_input import listen
from language_generation import LanguageGenerator
from scene_interpretation import SceneSummary, ObstacleSummary

# fake scene for now
scene = SceneSummary(obstacles=[
    ObstacleSummary(label="chair", position="left", distance_m=0.7, urgency="high")
])

gen = LanguageGenerator()

while True:
    user_text = listen()

    if user_text:
        response = gen.generate(scene, user_text)
        print("Response:", response)
