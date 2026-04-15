from typing import List, Tuple


# Directions in row/col form
MOVE_DELTAS = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}

# Clockwise orientation order
ORIENTATIONS = ["up", "right", "down", "left"]


def get_move_direction(from_cell: Tuple[int, int], to_cell: Tuple[int, int]) -> str:
    """Return the movement direction from one cell to the next."""
    row1, col1 = from_cell
    row2, col2 = to_cell
    delta = (row2 - row1, col2 - col1)

    if delta not in MOVE_DELTAS:
        raise ValueError(f"Invalid move from {from_cell} to {to_cell}")

    return MOVE_DELTAS[delta]


def get_turn_command(current_facing: str, target_direction: str) -> List[str]:
    """
    Return turn instructions needed to change from current_facing
    to target_direction.
    """
    current_index = ORIENTATIONS.index(current_facing)
    target_index = ORIENTATIONS.index(target_direction)

    diff = (target_index - current_index) % 4

    if diff == 0:
        return []
    elif diff == 1:
        return ["Turn right."]
    elif diff == 2:
        return ["Turn around."]
    elif diff == 3:
        return ["Turn left."]

    return []


def path_to_user_friendly_directions(
    path: List[Tuple[int, int]],
    destination_name: str,
    starting_facing: str = "up"
) -> List[str]:
    """
    Convert a path into user-friendly navigation instructions.
    Example:
    - Go forward 3 steps.
    - Turn right.
    - Go forward 4 steps.
    - You have reached the kitchen.
    """
    if len(path) < 2:
        return [f"You are already at the {destination_name}."]

    instructions = []
    current_facing = starting_facing

    # Convert path into movement directions like ["up", "up", "right", ...]
    move_directions = []
    for i in range(1, len(path)):
        move_direction = get_move_direction(path[i - 1], path[i])
        move_directions.append(move_direction)

    i = 0
    while i < len(move_directions):
        target_direction = move_directions[i]

        # Add turn instruction if needed
        turn_commands = get_turn_command(current_facing, target_direction)
        instructions.extend(turn_commands)

        # Update facing after turn
        current_facing = target_direction

        # Count consecutive forward moves in same direction
        step_count = 1
        i += 1
        while i < len(move_directions) and move_directions[i] == target_direction:
            step_count += 1
            i += 1

        if step_count == 1:
            instructions.append("Go forward 1 step.")
        else:
            instructions.append(f"Go forward {step_count} steps.")

    instructions.append(f"You have reached the {destination_name}.")
    return instructions