from typing import Tuple
from navigation.map_module import MapModule


class LocalizationModule:
    def __init__(self, map_module: MapModule, start_position: Tuple[int, int]):
        self.map_module = map_module

        if not self.map_module.is_walkable(start_position):
            raise ValueError(f"Invalid start position: {start_position}")

        self.current_position = start_position

    def get_position(self) -> Tuple[int, int]:
        """Return the current user position."""
        return self.current_position

    def set_position(self, new_position: Tuple[int, int]) -> bool:
        """Set the position manually if valid."""
        if self.map_module.is_walkable(new_position):
            self.current_position = new_position
            return True
        return False

    def move_up(self) -> bool:
        return self._move((-1, 0))

    def move_down(self) -> bool:
        return self._move((1, 0))

    def move_left(self) -> bool:
        return self._move((0, -1))

    def move_right(self) -> bool:
        return self._move((0, 1))

    def _move(self, delta: Tuple[int, int]) -> bool:
        """Move the current position if the next cell is walkable."""
        row, col = self.current_position
        d_row, d_col = delta

        new_position = (row + d_row, col + d_col)

        if self.map_module.is_walkable(new_position):
            self.current_position = new_position
            return True

        return False