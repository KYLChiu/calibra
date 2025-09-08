import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from datetime import datetime as dt

from calibra.model.base import ImpliedVolatilityModel
from calibra.instrument.option import Option, PayoffType

from dataclasses import dataclass


@dataclass
class BlackScholesModel(ImpliedVolatilityModel):
    r: float  # Risk-free interest rate
    q: float  # Dividend yield

    def _d1(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
    ) -> float:
        sqrtT = np.sqrt(T)
        return (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (
            sigma * sqrtT
        )

    def _d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
    ) -> float:
        return self._d1(S, K, T, sigma) - sigma * np.sqrt(T)

    def _tau(self, option: Option) -> float:
        # TODO: observe day count convention
        return (option.expiry - dt.now()).days / 365.0

    def price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: PayoffType,
    ) -> float:
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        sign = 1 if option_type == PayoffType.CALL else -1
        return sign * (
            S * np.exp(-self.q * T) * norm.cdf(sign * d1)
            - K * np.exp(-self.r * T) * norm.cdf(sign * d2)
        )

    def iv(self, option: Option, underlying_price: float, market_price: float) -> float:
        def _iv_objective(sigma: float) -> float:
            return (
                self.price(
                    underlying_price,
                    option.strike,
                    self._tau(option),
                    sigma,
                    option.payoff,
                )
                - market_price
            )

        try:
            return brentq(_iv_objective, a=1e-6, b=5.0)
        except ValueError:
            return float("nan")
