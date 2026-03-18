from collections import deque

class PathPlanner:
    def __init__(self, environment_model):
        self.env = environment_model

    def find_path(self, start, goal):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            visited.add(current)

            for neighbor in self.env.get_neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None