import torch
import numpy as np

class HestonVanilla:
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def _heston_cf(self, phi):
        a = -0.5 * phi**2 - 0.5j * phi
        b = self.kappa - self.rho * self.sigma * 1j * phi
        discriminant = b**2 - 2 * self.sigma**2 * a
        d = torch.sqrt(discriminant)
        g = (b - d) / (b + d)
        C = self.kappa * ((b - d) * self.T - 2 * torch.log((1 - g * torch.exp(-d * self.T)) / (1 - g))) / self.sigma**2
        D = ((b - d) / self.sigma**2) * (1 - torch.exp(-d * self.T)) / (1 - g * torch.exp(-d * self.T))
        cf = torch.exp(C * self.theta + D * self.v0 + 1j * phi * torch.log(self.S0 * torch.exp(self.r * self.T)))
        return cf

    def heston_price(self):
        P1 = 0.5
        P2 = 0.5
        umax = 50
        n = 100
        du = umax / n
        phi = du / 2

        for _ in range(n):
            factor1 = torch.exp(-1j * phi * torch.log(self.K))
            denominator = 1j * phi

            # P1 calculation
            cf1 = self._heston_cf(phi - 1j) / self._heston_cf(-1j)
            temp1 = factor1 * cf1 / denominator
            P1 += 1 / torch.pi * torch.real(temp1) * du

            # P2 calculation
            cf2 = self._heston_cf(phi)
            temp2 = factor1 * cf2 / denominator
            P2 += 1 / torch.pi * torch.real(temp2) * du

            phi += du

        price = self.S0 * P1 - torch.exp(-self.r * self.T) * self.K * P2
        return price
