from dataclasses import dataclass
from datetime import datetime as dt

from enum import Enum


class PayoffType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class ExerciseType(Enum):
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


@dataclass
class Option:
    symbol: str
    strike: float
    expiry: dt
    payoff: PayoffType
    exercise: ExerciseType
