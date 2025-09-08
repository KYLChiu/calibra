from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy.optimize import minimize


@dataclass
class SABRModel:
    alpha: float = None
    beta: float = 0.5
    rho: float = None
    nu: float = None
    calibrated: bool = False

    def hagan_vol(self, F: float, K: float, T: float) -> float:
        """
        Hagan's SABR formula for implied volatility.

        Parameters
        ----------
        F : float
            Forward price of the underlying
        K : float
            Strike price
        T : float
            Time to maturity in years

        Returns
        -------
        float
            Implied volatility for the given strike and maturity
        """
        if F == K:
            return self.alpha / (F ** (1 - self.beta))
        z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * np.log(F / K)
        x_z = np.log(
            (np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho)
        )
        return self.alpha / ((F * K) ** ((1 - self.beta) / 2)) * z / x_z

    def get_vol(self, F: float, K: float, T: float) -> float:
        """
        Get the calibrated SABR implied volatility for a given strike and maturity.

        Parameters
        ----------
        F : float
            Forward price of the underlying
        K : float
            Strike price
        T : float
            Time to maturity in years

        Returns
        -------
        float
            SABR implied volatility

        Raises
        ------
        ValueError
            If the model has not been calibrated yet
        """
        if not self.calibrated:
            raise ValueError("Model must be calibrated first")
        return self.hagan_vol(F, K, T)

    def calibrate(
        self, vol_surface: List[Dict[str, float]], F: float = 100.0
    ) -> "SABRModel":
        strikes = np.array([d["K"] for d in vol_surface])
        maturities = np.array([d["T"] for d in vol_surface])
        market_vols = np.array([d["market_vol"] for d in vol_surface])

        def objective(params: np.ndarray) -> float:
            alpha, rho, nu = params
            self.alpha, self.rho, self.nu = alpha, rho, nu
            error = 0.0
            for i, K in enumerate(strikes):
                T = maturities[i]
                sigma_model = self.get_vol(F, K, T)
                sigma_market = market_vols[i]
                error += (sigma_model - sigma_market) ** 2
            return error

        res = minimize(
            objective,
            x0=[0.2, 0.0, 0.5],
            bounds=[(1e-4, 5), (-0.999, 0.999), (1e-4, 5)],
        )
        self.alpha, self.rho, self.nu = res.x
        self.calibrated = True
        return self
