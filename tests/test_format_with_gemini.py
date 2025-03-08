import json
import pytest

from format_with_gemini import format_schedule
from models.schedule import Schedule


def test_format_schedule():
    path_to_schedule_json = "tests/input/example.json"
    path_to_formatted_schedule_json = "tests/output/formatted_schedule.json"

    # read in the schedule
    with open(path_to_schedule_json, "r", encoding="utf-8") as f:
        schedule = f.read()

    format_schedule(schedule, path_to_formatted_schedule_json)

    # read in the formatted schedule
    with open(path_to_formatted_schedule_json, "r", encoding="utf-8") as f:
        formatted_schedule = f.read()
        formatted_schedule_json: Schedule = json.loads(formatted_schedule)

    monday_classes = formatted_schedule_json["monday"]["classes"]
    first_class = monday_classes[0]
    first_class_name = first_class["name"]

    first_class_teacher = first_class["teacher"]
    first_class_room = first_class["room"]

    assert first_class_room not in first_class_name

    assert first_class_teacher is not None
    assert first_class_teacher not in first_class_name
