import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.navigation_controller import NavigationController

def test_navigation():
    nav = NavigationController()

    instructions = nav.navigate_to("kitchen")

    print("Navigation Instructions:")
    for instr in instructions:
        print(instr)


def test_obstacles():
    nav = NavigationController()

    result = nav.process_sensor_data(front=30, left=100, right=100)
    print("Obstacle Result:", result)


if __name__ == "__main__":
    test_navigation()
    test_obstacles()