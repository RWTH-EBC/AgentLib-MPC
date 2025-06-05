"""
Package containing utils for agentlib_mpc.
"""

from typing import Literal

TimeConversionTypes = Literal["seconds", "minutes", "hours", "days"]
TIME_CONVERSION: dict[TimeConversionTypes, int] = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
}


def is_time_in_intervals(time: float, intervals: list[tuple[float, float]]) -> bool:
    """
    Check if given time is within any of the provided intervals.

    Args:
        time: The time value to check
        intervals: List of tuples, each containing (start_time, end_time)

    Returns:
        True if time falls within any interval, False otherwise
    """
    return any(start <= time <= end for start, end in intervals)
