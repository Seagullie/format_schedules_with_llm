from typing import List, Optional, TypedDict, Union


class Class(TypedDict):
    index: int
    name: str
    room: str
    qualification: Optional[str]
    teacher: Optional[str]
    label: Optional[Union[int, float]]
    isBiweekly: Optional[bool]
    week: Optional[int]


class Day(TypedDict):
    classes: List[Class]


Schedule = TypedDict(
    "Schedule",
    {
        "monday": Day,
        "tuesday": Day,
        "wednesday": Day,
        "thursday": Day,
        "friday": Day,
    },
)
