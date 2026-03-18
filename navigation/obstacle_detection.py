class ObstacleDetection:
    def __init__(self, threshold=50):
        self.threshold = threshold  # cm

    def check_obstacles(self, front, left, right):
        if front < self.threshold:
            return "STOP: Obstacle ahead"

        if left < right - 10:
            return "MOVE RIGHT"

        if right < left - 10:
            return "MOVE LEFT"

        return "CLEAR"