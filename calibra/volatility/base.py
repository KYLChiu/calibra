from abc import ABC, abstractmethod

from calibra.instrument.option import Option

from calibra.interpolation.base import VolatilityInterpolator


from dataclasses import dataclass
from typing import List
import numpy as np


class ImpliedVolatilityModel(ABC):
    @abstractmethod
    def iv(self, option: Option) -> float:
        pass


@dataclass
class VolatilitySurfacePoint:
    strike: float
    maturity: float
    market_vol: float


class VolatilitySurface:
    """Volatility surface using an abstract interpolator."""

    def __init__(
        self, points: List[VolatilitySurfacePoint], interpolator: VolatilityInterpolator
    ):
        self.points = points
        self.interpolator = interpolator
        strikes = np.array([p.strike for p in points])
        maturities = np.array([p.maturity for p in points])
        volatilities = np.array([p.market_volatility for p in points])
        self.interpolator.fit(strikes, maturities, volatilities)

    def get_volatility(self, strike: float, maturity: float) -> float:
        return self.interpolator(strike, maturity)
