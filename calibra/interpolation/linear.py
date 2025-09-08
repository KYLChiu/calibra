import numpy as np

from calibra.interpolation.base import VolatilityInterpolator
from scipy.interpolate import LinearNDInterpolator


class LinearVolatilityInterpolator(VolatilityInterpolator):
    def __init__(
        self, strikes: np.ndarray, maturities: np.ndarray, volatilities: np.ndarray
    ):
        self._interp = LinearNDInterpolator(
            list(zip(strikes, maturities)), volatilities
        )

    def __call__(self, strike: float, maturity: float) -> float:
        volatility = self._interp(strike, maturity)
        if np.isnan(volatility):
            # nearest neighbor fallback
            idx = np.argmin(
                np.abs(self._interp.points[:, 0] - strike)
                + np.abs(self._interp.points[:, 1] - maturity)
            )
            return float(self._interp.values[idx])
        return float(volatility)
