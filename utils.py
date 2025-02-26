import json
import os
from typing import List


def get_schedule_paths_from_dir(path_to_dir: str, ext=".json") -> List[str]:
    """Gets all schedule paths from a directory (non-recursive)."""
    print(f"Searching for schedule files in directory: {path_to_dir}")
    schedule_paths = []

    # Get only files in the specified directory
    try:
        files = os.listdir(path_to_dir)
        # Sort files to ensure consistent ordering
        files.sort()

        for file in files:
            full_path = os.path.join(path_to_dir, file)
            # Check if it's a file (not a directory) and has the correct extension
            if os.path.isfile(full_path) and file.endswith(ext):
                schedule_paths.append(full_path)
                print(f"Found schedule file: {full_path}")

        print(f"Total schedule files found: {len(schedule_paths)}")
        return schedule_paths
    except Exception as e:
        print(f"Error accessing directory {path_to_dir}: {str(e)}")
        raise


def read_schedule(path_to_schedule: str) -> str:
    """Reads the schedule from a file."""
    print(f"Reading schedule from: {path_to_schedule}")
    try:
        with open(path_to_schedule, "r", encoding="utf-8") as f:
            schedule = f.read()

        if not is_valid_json(schedule):
            print(f"Invalid JSON detected while trying to read schedule")
            raise ValueError(
                f"Error during reading schedule from {path_to_schedule}. Invalid JSON: \n{schedule}"
            )

        print(f"Successfully read schedule file: {path_to_schedule}")
        return schedule
    except Exception as e:
        print(f"Error reading schedule file {path_to_schedule}: {str(e)}")
        raise


def save_schedule(path_to_schedule: str, schedule_json: str) -> None:
    """Saves the schedule to a file."""
    print(f"Attempting to save schedule to: {path_to_schedule}")

    if not is_valid_json(schedule_json):
        print(f"Invalid JSON detected while trying to save schedule")
        raise ValueError(
            f"Error during saving schedule to {path_to_schedule}. Invalid JSON: \n{schedule_json}"
        )

    try:
        with open(path_to_schedule, "w", encoding="utf-8") as f:
            f.write(schedule_json)
        print(f"Successfully saved schedule to {path_to_schedule}")
    except Exception as e:
        print(f"Error saving schedule to {path_to_schedule}: {str(e)}")
        raise


def is_valid_json(json_str: str) -> bool:
    """Checks if a string is valid JSON."""
    try:
        json.loads(json_str)
        return True
    except ValueError as e:
        print(f"JSON validation failed: {str(e)}")
        return False
