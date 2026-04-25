"""
nav_voice_system.py
-------------------
Main voice-driven navigation loop for the AI Smart Walker.

Flow each cycle:
  1. MonkMakes speaker asks "Where would you like to go?"
  2. USB microphone captures the user's answer (via Vosk offline STT).
  3. destination_parser extracts the room name from the transcription.
  4. PathPlanner (BFS) finds the shortest route on the house grid map.
  5. instruction_generator converts the path to spoken sentences.
  6. MonkMakes speaker reads each instruction aloud.

Run from the project root:
    python -m voice_navigation.nav_voice_system
  or:
    python voice_navigation/nav_voice_system.py

Hardware required:
  - USB microphone (any UAS Audio class device)
  - MonkMakes Amplified Speaker 2 (see WIRING_GUIDE.md)
"""

import os
import sys
import logging

# ---------------------------------------------------------------------------
# Path setup — works whether run as a module or a script
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from navigation.map_module import MapModule                          # noqa: E402
from navigation.path_planner import PathPlanner                     # noqa: E402
from navigation.localization_module import LocalizationModule       # noqa: E402
from navigation.instruction_generator import (                      # noqa: E402
    path_to_user_friendly_directions,
)
from ai.voice_input import listen                                    # noqa: E402
from voice_navigation.voice_output import (                         # noqa: E402
    configure_audio_output,
    speak,
    speak_directions,
)
from voice_navigation.destination_parser import (                   # noqa: E402
    parse_destination,
    list_destinations_speech,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Path to the house grid map
MAP_FILE = os.path.join(_PROJECT_ROOT, "maps", "house_map.json")

# The cell the user starts on — must match a walkable cell in the map.
# (4, 4) = "entrance" in maps/house_map.json
DEFAULT_START_POSITION = (4, 4)

# Which direction the user is facing when the session starts.
# "up" means decreasing row index = forward on the grid.
DEFAULT_STARTING_FACING = "up"

# Words that end the navigation session
_QUIT_WORDS = {"quit", "exit", "stop", "bye", "goodbye", "end", "cancel"}


# ---------------------------------------------------------------------------
# Core session
# ---------------------------------------------------------------------------

def run(
    start_position: tuple = DEFAULT_START_POSITION,
    starting_facing: str = DEFAULT_STARTING_FACING,
    map_file: str = MAP_FILE,
) -> None:
    """
    Start the voice navigation session.

    Args:
        start_position:   (row, col) of the user's current location on the map.
        starting_facing:  Direction the user faces at startup ("up" / "down" /
                          "left" / "right").
        map_file:         Path to the house_map.json grid file.
    """
    # --- 1. Audio routing — send sound to the MonkMakes 3.5mm jack ---
    configure_audio_output()

    # --- 2. Load map ---
    if not os.path.exists(map_file):
        logger.error("Map file not found: %s", map_file)
        speak("Error: map file not found. Please check the setup.")
        return

    map_module = MapModule(map_file)
    map_module.load_map()

    known_destinations = list(map_module.destination_lookup.keys())
    destination_list_speech = list_destinations_speech(known_destinations)

    logger.info("Map loaded. Destinations: %s", known_destinations)

    # --- 3. Localisation — where is the user right now? ---
    try:
        localization = LocalizationModule(map_module, start_position)
    except ValueError as exc:
        logger.error("Bad start position %s: %s", start_position, exc)
        speak("Error: the starting position is not on the map.")
        return

    planner = PathPlanner(map_module)
    current_facing = starting_facing

    # --- 4. Greeting ---
    speak("Navigation system ready.")
    speak(f"Available rooms are: {destination_list_speech}.")

    # --- 5. Main loop ---
    while True:
        speak("Where would you like to go?")

        # Listen for destination
        logger.info("[nav] Listening …")
        try:
            text = listen()
        except Exception as exc:
            logger.error("Microphone error: %s", exc)
            speak("Microphone error. Please check the connection and try again.")
            continue

        if not text:
            speak("I didn't catch that. Please try again.")
            continue

        logger.info("[nav] Heard: %s", text)

        # Check for quit command
        words = set(text.lower().split())
        if words & _QUIT_WORDS:
            speak("Ending navigation. Goodbye.")
            break

        # Parse destination
        destination = parse_destination(text, known_destinations)

        if destination is None:
            speak(
                f"Sorry, I don't recognise that room. "
                f"You can say: {destination_list_speech}."
            )
            continue

        dest_display = destination.replace("_", " ")

        # Check if already there
        current_pos = localization.get_position()
        destination_pos = map_module.get_destination_position(destination)

        if destination_pos is None:
            speak(f"The {dest_display} is not on the current map.")
            continue

        if current_pos == destination_pos:
            speak(f"You are already at the {dest_display}.")
            continue

        # Plan path
        path = planner.find_path_bfs(current_pos, destination_pos)

        if path is None:
            speak(
                f"Sorry, I cannot find a route to the {dest_display} from here."
            )
            continue

        # Generate human-friendly directions
        directions = path_to_user_friendly_directions(
            path=path,
            destination_name=dest_display,
            starting_facing=current_facing,
        )

        step_count = len(path) - 1
        speak(
            f"Navigating to the {dest_display}. "
            f"That is {step_count} step{'s' if step_count != 1 else ''} away."
        )

        # Speak each direction through the MonkMakes speaker
        speak_directions(directions)

        # Update position and infer new facing from last step of path
        localization.set_position(destination_pos)
        if len(path) >= 2:
            current_facing = _infer_facing(path[-2], path[-1])

        speak("Have you arrived? Say a room name for your next destination.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _infer_facing(from_cell: tuple, to_cell: tuple) -> str:
    """Return the direction of travel between the last two path cells."""
    dr = to_cell[0] - from_cell[0]
    dc = to_cell[1] - from_cell[1]
    mapping = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    return mapping.get((dr, dc), "up")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
