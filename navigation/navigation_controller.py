from navigation.environment_model import EnvironmentModel
from navigation.localization import Localization
from navigation.path_planning import PathPlanner
from navigation.obstacle_detection import ObstacleDetection


class NavigationController:
    def __init__(self):
        self.env = EnvironmentModel()
        self.localization = Localization("hallway")
        self.planner = PathPlanner(self.env)
        self.obstacle = ObstacleDetection()

        self._setup_environment()

    def _setup_environment(self):
        self.env.add_location("hallway")
        self.env.add_location("kitchen")
        self.env.add_location("living_room")

        self.env.connect_locations("hallway", "kitchen")
        self.env.connect_locations("hallway", "living_room")

    def navigate_to(self, destination):
        start = self.localization.get_location()
        path = self.planner.find_path(start, destination)

        if not path:
            return ["No path found"]

        instructions = []
        for step in path[1:]:
            instructions.append(f"Move to {step}")

        return instructions

    def process_sensor_data(self, front, left, right):
        return self.obstacle.check_obstacles(front, left, right)