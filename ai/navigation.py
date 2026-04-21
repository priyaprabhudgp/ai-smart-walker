"""
ai/navigation.py

Keyword-based intent extraction and BFS navigation for a pre-mapped home layout.

Flow:
    speech text → extract_intent() → Navigator.get_instructions() → spoken directions

Position is user-declared, not sensor-tracked. Defaults to the map's default_start on boot.
User can re-declare position mid-session: "I'm in the kitchen, take me to the bedroom"

Maps are loaded from JSON files in the maps/ directory.
"""

import json
import os
from collections import deque
from typing import Optional


# ----- MAP LOADER -----

def load_map(map_file: str) -> dict:
    """Load a house map from a JSON file. Returns the full map data dict."""
    with open(map_file, "r") as f:
        return json.load(f)


def _build_locations_sorted(locations: dict[str, str]) -> list[tuple[str, str]]:
    """Sort location keyword table longest-first to prevent short phrases shadowing long ones."""
    return sorted(locations.items(), key=lambda x: -len(x[0]))


# ----- DEFAULT MAP -----

_DEFAULT_MAP_FILE = os.path.join(
    os.path.dirname(__file__), "..", "maps", "layout1.json"
)


# ----- INTENT EXTRACTION -----

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


def _find_location(text: str, locations_sorted: list[tuple[str, str]]) -> Optional[str]:
    for phrase, node in locations_sorted:
        if phrase in text:
            return node
    return None


def extract_intent(
    text: str,
    locations_sorted: list[tuple[str, str]],
) -> dict[str, Optional[str]]:
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
            → {"current": "hallway", "destination": None}
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
        current = _find_location(text[pos_start:pos_end], locations_sorted)

    if dest_idx != -1:
        destination = _find_location(text[dest_idx + dest_cue_len:], locations_sorted)

    return {"current": current, "destination": destination}


# ----- PATHFINDING -----

def _bfs(start: str, end: str, graph: dict) -> Optional[list[str]]:
    """BFS over graph. Returns ordered list of node IDs, or None if unreachable."""
    if start == end:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        for neighbor in graph.get(path[-1], {}):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


# ----- NAVIGATOR -----

class Navigator:
    """
    Loads a house map from a JSON file and generates turn-by-turn spoken instructions.
    Position defaults to the map's default_start on boot and updates whenever
    the user declares their location.

    Args:
        map_file: Path to a map JSON file. Defaults to maps/layout1.json.
    """

    def __init__(self, map_file: str = _DEFAULT_MAP_FILE):
        data = load_map(map_file)
        self._graph: dict[str, dict[str, str]] = {
            node: {neighbor: edge["instruction"] for neighbor, edge in edges.items()}
            for node, edges in data["nodes"].items()
        }
        self._locations_sorted = _build_locations_sorted(data["locations"])
        self.current_position: str = data["default_start"]

    def update_position(self, node: str):
        self.current_position = node

    def handle(self, text: str) -> Optional[str]:
        """
        Main entry point. Takes raw speech, updates position if declared,
        and returns spoken instructions if a destination was given.
        """
        intent = extract_intent(text, self._locations_sorted)

        if intent["current"]:
            self.current_position = intent["current"]

        if intent["destination"]:
            return self.get_instructions(intent["destination"])

        if intent["current"] and not intent["destination"]:
            name = intent["current"].replace("_", " ")
            return f"Got it, I've updated your position to the {name}."

        return None

    def get_instructions(self, destination: str) -> str:
        current_name = self.current_position.replace("_", " ")
        dest_name    = destination.replace("_", " ")

        if destination == self.current_position:
            return f"You are already at the {dest_name}."

        path = _bfs(self.current_position, destination, self._graph)
        if path is None:
            return f"Sorry, I couldn't find a route to the {dest_name}."

        steps = [self._graph[path[i]][path[i + 1]] for i in range(len(path) - 1)]
        directions = ". ".join(step.capitalize() for step in steps) + "."
        return f"You are currently in the {current_name}, navigating to the {dest_name}. {directions}"


# ----- QUICK TEST -----

if __name__ == "__main__":
    nav = Navigator()

    tests = [
        "take me to the master bedroom",
        "take me to the bathroom",
        "take me to the kitchen",
        "take me to bob's room",
        "take me to steve's room",
        "take me to the master bathroom",
        "take me to the garage",
        "take me to the backyard",
        "take me to the laundry room",
        "i'm in the kitchen, take me to the bathroom",
        "i'm in the kitchen, take me to the master bedroom",
        "i'm in steve's room, take me to the living room",
        "i'm in the master bedroom, take me to the bathroom",
        "i'm in bob's room, take me to the master bathroom",
        "i'm in the garage, take me to the bathroom",
        "i'm in the backyard, take me to the master bedroom",
        "i'm in the bathroom, take me to the garage",
        "i'm in the master bathroom, take me to bob's room",
        "i'm in the kitchen, take me to the kitchen",
    ]

    for t in tests:
        print(f'  Input:    "{t}"')
        print(f'  Output:   "{nav.handle(t)}"')
        print(f'  Position: {nav.current_position}')
        print()
