from calibra.volatility.base import ImpliedVolatilityModel
from calibra.instrument.option import Option

from dataclasses import dataclass


@dataclass
class FlatVolatilityModel(ImpliedVolatilityModel):
    volatility: float

    def iv(self, option: Option) -> float:
        return self.volatility
