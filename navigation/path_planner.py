from collections import deque
from typing import List, Tuple, Optional, Dict
from navigation.map_module import MapModule


class PathPlanner:
    def __init__(self, map_module: MapModule):
        self.map_module = map_module

    def find_path_bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to goal using BFS.
        Returns a list of coordinates or None if no path exists.
        """
        if not self.map_module.is_walkable(start):
            return None

        if not self.map_module.is_walkable(goal):
            return None

        queue = deque([start])
        visited = {start}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self._reconstruct_path(parent, goal)

            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return None

    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid neighboring cells."""
        row, col = position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]

        return [pos for pos in candidates if self.map_module.is_walkable(pos)]

    def _reconstruct_path(
        self,
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Rebuild the path from goal back to start."""
        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path