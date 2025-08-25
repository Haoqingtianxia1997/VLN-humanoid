from src.execute.actions import execute_action_sequence

actions = [
        {"type": "perceive", "target": "cup"},
        {"type": "move", "target": "apple"},
        {"type": "grasp", "target": "apple"},
        {"type": "move", "target": "plate"},
        {"type": "place", "target": "plate"},
        {"type": "reset", "target": "home"}
    ]

execute_action_sequence(actions)