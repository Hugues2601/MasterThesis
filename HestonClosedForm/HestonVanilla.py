import torch
import numpy as np
import pandas as pd

class HestonVanilla:
    def __init__(self,phi, S0, T, r, kappa, v0, theta, sigma, rho):
        self.phi = phi
        self.S0 = S0
        self.T=T
        self.r=r
        self.kappa=kappa
        self.v0=v0
        self.theta=theta
        self.sigma=sigma
        self.rho=rho

    def _heston_cf(self):
        a = -0.5 * self.phi ** 2 - 0.5j * self.phi
        b = self.kappa - self.rho * self.sigma * 1j * self.phi
        g = ((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) / (
                    (b + torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2)
        C = self.kappa * (((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) * T - 2 / self.sigma ** 2 * torch.log(
            (1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T)) / (1 - g)))
        D = ((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) * (
                    1 - torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T)) / (
                        1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T))
        cf = torch.exp(C * self.theta + D * self.v0 + 1j * self.phi * torch.log(self.S0 * torch.exp(self.r * self.T)))

        return cf

    def heston_price(self):
        params = (self.S0, self.T, self.r, self.kappa, self.v0, self.theta, self.sigma, self.rho)
        P1 = torch.full_like(self.S0, 0.5, device=self.S0.device)
        P2 = torch.full_like(self.S0, 0.5, device=self.S0.device)
        umax = 50
        n = 100
        du = umax / n
        phi = du / 2
        for _ in range(n):
            factor1 = torch.exp(-1j * self.phi * torch.log(self.K))
            denominator = 1j * self.phi
            cf1 = self._heston_cf(self.phi - 1j, *params) / self._heston_cf(-1j, *params)
            temp1 = factor1 * cf1 / denominator
            P1 += 1 / torch.pi * torch.real(temp1) * du
            cf2 = self._heston_cf(self.phi, *params)
            temp2 = factor1 * cf2 / denominator
            P2 += 1 / torch.pi * torch.real(temp2) * du
            phi += du
        price = self.S0 * P1 - torch.exp(-self.r * self.T) * self.K * P2

        return price