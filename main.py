from navigation.map_module import MapModule
from navigation.localization_module import LocalizationModule
from navigation.path_planner import PathPlanner
from navigation.instruction_generator import path_to_user_friendly_directions


def main():
    # Step 1: Load the map
    map_module = MapModule("house_map.json")
    map_module.load_map()

    print("Loaded map:")
    map_module.print_map()

    print("\nKnown destinations:")
    map_module.print_destinations()

    # Step 2: Set the starting position manually
    # (4, 1) is the living_room in this example map
    start_position = (4, 1)
    localization = LocalizationModule(map_module, start_position)

    print("\nCurrent position:")
    print(localization.get_position())

    # Step 3: Pick a destination
    destination_name = "bedroom"
    destination_position = map_module.get_destination_position(destination_name)

    if destination_position is None:
        print(f"Destination '{destination_name}' not found.")
        return

    print(f"\nDestination '{destination_name}' is at {destination_position}")

    # Step 4: Plan a path
    planner = PathPlanner(map_module)
    path = planner.find_path_bfs(localization.get_position(), destination_position)

    if path is None:
        print("No path found.")
        return

    print("\nPath found:")
    print(path)

    # Step 5: Convert path to user-friendly directions
    directions = path_to_user_friendly_directions(
        path=path,
        destination_name=destination_name,
        starting_facing="up"
    )

    print("\nUser-friendly directions:")
    for step in directions:
        print(step)


if __name__ == "__main__":
    main()