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
