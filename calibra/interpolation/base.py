from abc import ABC, abstractmethod


class VolatilityInterpolator(ABC):
    @abstractmethod
    def __call__(self, strike: float, maturity: float) -> float:
        """Return interpolated volatility for given strike and maturity"""
        pass
