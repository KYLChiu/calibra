import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp2d


class LocalVolModel:
    def __init__(self, strikes, maturities, vols, r, q):
        """
        strikes: 1D array of strikes
        maturities: 1D array of maturities
        vols: 2D array of market implied vols (shape: len(maturities) x len(strikes))
        r: risk-free rate
        q: dividend yield
        """
        self.strikes = strikes
        self.maturities = maturities
        self.vols = vols
        self.r = r
        self.q = q

        # Compute option price surface from IV surface
        self.price_surface = np.zeros_like(vols)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                sigma = vols[i, j]
                self.price_surface[i, j] = self.bs_price(
                    1.0, K, T, sigma
                )  # scale spot=1

        # Interpolator for price surface
        self.price_interp = interp2d(
            strikes, maturities, self.price_surface, kind="cubic"
        )

    def bs_price(self, S, K, T, sigma):
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-self.q * T) * S * norm.cdf(d1) - np.exp(
            -self.r * T
        ) * K * norm.cdf(d2)

    def get_vol(self, S, K, T, h=1e-4):
        """
        Compute local vol at spot S and time T using Dupire formula.
        """
        C = self.price_interp(K, T)
        dC_dT = (self.price_interp(K, T + h) - self.price_interp(K, T - h)) / (2 * h)
        dC_dK = (self.price_interp(K + h, T) - self.price_interp(K - h, T)) / (2 * h)
        d2C_dK2 = (
            self.price_interp(K + h, T) - 2 * C + self.price_interp(K - h, T)
        ) / (h**2)

        local_var = (dC_dT + self.r * K * dC_dK) / (0.5 * K**2 * d2C_dK2)
        return np.sqrt(max(local_var, 0.0))
