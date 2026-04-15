import json
from typing import List, Tuple, Dict, Optional


class MapModule:
    def __init__(self, map_file: str):
        self.map_file = map_file
        self.grid: List[List[int]] = []
        self.legend: Dict[int, str] = {}
        self.destination_lookup: Dict[str, Tuple[int, int]] = {}

    def load_map(self) -> None:
        """Load the map JSON file into memory."""
        with open(self.map_file, "r") as file:
            data = json.load(file)

        self.grid = data["grid"]
        self.legend = {int(k): v for k, v in data["legend"].items()}
        self._build_destination_lookup()

    def _build_destination_lookup(self) -> None:
        """Build a dictionary like {'kitchen': (1, 5)}."""
        self.destination_lookup.clear()

        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                cell_value = self.grid[row][col]

                if cell_value in self.legend:
                    label = self.legend[cell_value]

                    if label not in ("blocked", "walkable"):
                        self.destination_lookup[label] = (row, col)

    def is_walkable(self, position: Tuple[int, int]) -> bool:
        """Return True if a position is inside the grid and not blocked."""
        row, col = position

        if not self._is_in_bounds(position):
            return False

        cell_value = self.grid[row][col]
        label = self.legend.get(cell_value, "unknown")
        return label != "blocked"

    def _is_in_bounds(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside the grid."""
        row, col = position
        return 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0])

    def get_destination_position(self, destination_name: str) -> Optional[Tuple[int, int]]:
        """Return the coordinates of a destination like 'kitchen'."""
        return self.destination_lookup.get(destination_name)

    def get_grid(self) -> List[List[int]]:
        """Return the full map grid."""
        return self.grid

    def print_map(self) -> None:
        """Print the map grid."""
        for row in self.grid:
            print(row)

    def print_destinations(self) -> None:
        """Print all named destinations and their coordinates."""
        for name, coords in self.destination_lookup.items():
            print(f"{name}: {coords}")