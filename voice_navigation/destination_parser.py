"""
destination_parser.py
---------------------
Extracts a known destination name from free-form Vosk speech transcription.

How it works:
  1. Normalise the spoken text (lower-case, strip punctuation).
  2. Try an exact substring match against each known destination.
  3. If no exact match, try word-level fuzzy matching (handles "living room"
     matching the map key "living_room", etc.).

Usage:
    from voice_navigation.destination_parser import parse_destination

    destinations = ["kitchen", "bedroom", "living_room", "office", "bathroom", "entrance"]
    result = parse_destination("take me to the living room please", destinations)
    # → "living_room"
"""

import re
from typing import List, Optional


# Words the user may say that we can safely ignore when matching room names.
_FILLER_WORDS = {
    "go", "to", "the", "take", "me", "navigate", "please", "can", "you",
    "i", "want", "need", "find", "show", "get", "head", "walk", "move",
    "bring", "a", "an", "room", "let", "would", "like", "how", "do",
}


def _normalise(text: str) -> str:
    """Lower-case and strip everything that isn't a letter or space."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def _dest_tokens(destination: str) -> List[str]:
    """Split a destination key like 'living_room' into ['living', 'room']."""
    return re.split(r"[_\s]+", destination.lower())


def parse_destination(text: str, known_destinations: List[str]) -> Optional[str]:
    """
    Return the best matching destination key from *known_destinations*,
    or None if nothing matches.

    Args:
        text:               Raw Vosk transcription string.
        known_destinations: List of destination keys from the map legend
                            (e.g. ["kitchen", "living_room", "bedroom"]).

    Returns:
        The matched destination key (e.g. "living_room"), or None.
    """
    normalised = _normalise(text)

    # --- Pass 1: exact substring match (fastest) ---
    for dest in known_destinations:
        # Replace underscores with spaces for matching "living_room" → "living room"
        dest_display = dest.replace("_", " ")
        if dest_display in normalised:
            return dest

    # --- Pass 2: all tokens of destination appear in the spoken words ---
    spoken_words = set(normalised.split())
    for dest in known_destinations:
        tokens = _dest_tokens(dest)
        if all(t in spoken_words for t in tokens):
            return dest

    # --- Pass 3: partial word overlap (at least one non-filler token matches) ---
    meaningful_words = spoken_words - _FILLER_WORDS
    best_dest: Optional[str] = None
    best_score = 0

    for dest in known_destinations:
        tokens = set(_dest_tokens(dest))
        overlap = len(tokens & meaningful_words)
        if overlap > best_score:
            best_score = overlap
            best_dest = dest

    # Only accept if we got at least one real matching token
    return best_dest if best_score > 0 else None


def list_destinations_speech(known_destinations: List[str]) -> str:
    """
    Return a readable comma-separated list of destinations for speaking aloud.
    e.g. ["kitchen", "living_room"] → "kitchen and living room"
    """
    readable = [d.replace("_", " ") for d in known_destinations]
    if len(readable) == 0:
        return "no rooms available"
    if len(readable) == 1:
        return readable[0]
    return ", ".join(readable[:-1]) + ", and " + readable[-1]
