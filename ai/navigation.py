"""
ai/navigation.py

Keyword-based intent extraction and BFS navigation for a pre-mapped home layout.

Flow:
    speech text → extract_intent() → Navigator.get_instructions() → spoken directions

Position is user-declared, not sensor-tracked. Defaults to front_door on boot.
User can re-declare position mid-session: "I'm in the kitchen, take me to the bedroom"
"""

from collections import deque
from typing import Optional


# ----- MAP -----
# node → {neighbor: walking instruction to get there}
# THIS IS BASED ON LAYOUT1 (view in testhouselayouts)

MAP: dict[str, dict[str, str]] = {
    "front_door": {
        "living_room": "walk forward",
    },
    "living_room": {
        "front_door":  "walk to the front door",
        "hallway":     "turn left into the hallway",
        "kitchen":     "turn right through the kitchen door",
        "backyard":    "walk forward through the backyard door",
    },
    "hallway": {
        "living_room":     "walk to the end of the hallway and go through the door into the living room",
        "bobs_room":       "turn right through the first door into Bob's room",
        "steves_room":     "turn right through the second door into Steve's room",
        "bathroom":        "walk straight to the end of the hallway through the bathroom door",
        "master_bathroom": "turn left through the door into the master bathroom",
        "master_bedroom":  "walk to the end of the hallway and turn left into the master bedroom",
    },
    "kitchen": {
        "living_room":  "go through the door into the living room",
        "laundry_room": "turn right, walk forward, and turn left through the laundry room door",
    },
    "laundry_room": {
        "kitchen":  "go through the door into the kitchen",
        "garage":   "turn right and walk forward into the garage",
        "backyard": "go through the door to the backyard",
    },
    "garage": {
        "laundry_room": "go through the door into the laundry room",
    },
    "backyard": {
        "living_room":  "go through the door into the living room",
        "laundry_room": "go through the door into the laundry room",
    },
    "bobs_room": {
        "hallway": "go out through the door to the hallway",
    },
    "steves_room": {
        "hallway": "go out through the door to the hallway",
    },
    "bathroom": {
        "hallway": "go out through the door to the hallway",
    },
    "master_bathroom": {
        "hallway":        "go through the door to the hallway",
        "master_bedroom": "go through the door into the master bedroom",
    },
    "master_bedroom": {
        "hallway":         "go through the door to the hallway",
        "master_bathroom": "go through the door into the master bathroom",
    },
}


# ----- KEYWORD TABLES -----
# Longest phrases first so "master bathroom" matches before "bathroom"

LOCATIONS: dict[str, str] = {
    "front door":      "front_door",
    "hallway":         "hallway",
    "living room":     "living_room",
    "kitchen":         "kitchen",
    "laundry room":    "laundry_room",
    "laundry":         "laundry_room",
    "garage":          "garage",
    "bob's room":      "bobs_room",
    "bobs room":       "bobs_room",
    "bob's":           "bobs_room",
    "steve's room":    "steves_room",
    "steves room":     "steves_room",
    "steve's":         "steves_room",
    "master bathroom": "master_bathroom",
    "master bedroom":  "master_bedroom",
    "bathroom":        "bathroom",
    "bedroom":         "master_bedroom",
    "back yard":       "backyard",
    "backyard":        "backyard",
}

_LOCATIONS_SORTED = sorted(LOCATIONS.items(), key=lambda x: -len(x[0]))

DESTINATION_CUES = [
    "take me to", "guide me to", "navigate to",
    "i want to go to", "i need to go to", "get me to",
    "how do i get to", "go to",
]

POSITION_CUES = [
    "i'm outside", "i am outside",
    "i'm near",    "i am near",
    "i'm in",      "i am in",
    "i'm at",      "i am at",
]


# ----- INTENT EXTRACTION -----

def _find_location(text: str) -> Optional[str]:
    for phrase, node in _LOCATIONS_SORTED:
        if phrase in text:
            return node
    return None


def extract_intent(text: str) -> dict[str, Optional[str]]:
    """
    Parse raw speech into current position and/or destination.

    Returns:
        {"current": node_id or None, "destination": node_id or None}

    Examples:
        "take me to the kitchen"
            → {"current": None, "destination": "kitchen"}

        "i'm in the living room, take me to the bathroom"
            → {"current": "living_room", "destination": "bathroom"}

        "i'm at the hallway"
            → {"current": "main_hallway", "destination": None}
    """
    text = text.lower()
    current = None
    destination = None

    pos_idx, pos_cue_len = -1, 0
    for cue in POSITION_CUES:
        idx = text.find(cue)
        if idx != -1:
            pos_idx, pos_cue_len = idx, len(cue)
            break

    dest_idx, dest_cue_len = -1, 0
    for cue in DESTINATION_CUES:
        idx = text.find(cue)
        if idx != -1:
            dest_idx, dest_cue_len = idx, len(cue)
            break

    if pos_idx != -1:
        pos_start = pos_idx + pos_cue_len
        pos_end = dest_idx if dest_idx > pos_idx else len(text)
        current = _find_location(text[pos_start:pos_end])

    if dest_idx != -1:
        destination = _find_location(text[dest_idx + dest_cue_len:])

    return {"current": current, "destination": destination}


# ----- PATHFINDING -----

def _bfs(start: str, end: str) -> Optional[list[str]]:
    """BFS over MAP. Returns ordered list of node IDs, or None if unreachable."""
    if start == end:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for neighbor in MAP.get(path[-1], {}):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


# ----- NAVIGATOR -----

class Navigator:
    """
    Tracks user's declared position and generates turn-by-turn spoken instructions.
    Position defaults to front_door on boot and updates whenever the user declares it.
    """

    def __init__(self, start: str = "front_door"):
        self.current_position = start

    def update_position(self, node: str):
        self.current_position = node

    def handle(self, text: str) -> Optional[str]:
        """
        Main entry point. Takes raw speech, updates position if declared,
        and returns spoken instructions if a destination was given.
        """
        intent = extract_intent(text)

        if intent["current"]:
            self.current_position = intent["current"]

        if intent["destination"]:
            return self.get_instructions(intent["destination"])

        if intent["current"] and not intent["destination"]:
            name = intent["current"].replace("_", " ")
            return f"Got it, I've updated your position to the {name}."

        return None

    def get_instructions(self, destination: str) -> str:
        if destination == self.current_position:
            return f"You are already at the {destination.replace('_', ' ')}."

        path = _bfs(self.current_position, destination)
        if path is None:
            return f"Sorry, I couldn't find a route to the {destination.replace('_', ' ')}."

        steps = [MAP[path[i]][path[i + 1]] for i in range(len(path) - 1)]
        return ". ".join(step.capitalize() for step in steps) + "."


# ----- QUICK TEST -----

if __name__ == "__main__":
    nav = Navigator()

    tests = [
        # from front door
        "take me to the master bedroom",
        "take me to the bathroom",
        "take me to the kitchen",
        "take me to bob's room",
        "take me to steve's room",
        "take me to the master bathroom",
        "take me to the garage",
        "take me to the backyard",
        "take me to the laundry room",
        # mid-session position updates
        "i'm in the kitchen, take me to the bathroom",
        "i'm in the kitchen, take me to the master bedroom",
        "i'm in steve's room, take me to the living room",
        "i'm in the master bedroom, take me to the bathroom",
        "i'm in bob's room, take me to the master bathroom",
        "i'm in the garage, take me to the bathroom",
        "i'm in the backyard, take me to the master bedroom",
        "i'm in the bathroom, take me to the garage",
        "i'm in the master bathroom, take me to bob's room",
        # already there
        "i'm in the kitchen, take me to the kitchen",
    ]

    for t in tests:
        print(f'  Input:    "{t}"')
        print(f'  Output:   "{nav.handle(t)}"')
        print(f'  Position: {nav.current_position}')
        print()
