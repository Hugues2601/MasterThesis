import pandas as pd
import torch
import numpy as np
from HestonClosedForm.HestonVanilla import HestonVanilla

class HestonCalibrator:
    def __init__(self, csv_path, spot_price, risk_free_rate, device="cuda"):
        self.data = pd.read_csv(csv_path)
        self.S0 = torch.tensor(spot_price, device=device, dtype=torch.float32)
        self.r = torch.tensor(risk_free_rate, device=device, dtype=torch.float32)
        self.device = device
        self.calibrated_params = None

    def _heston_price(self, K, T, kappa, theta, sigma, rho, v0):

        heston = HestonVanilla(self.S0, K, T, self.r, kappa, v0, theta, sigma, rho)
        return heston.heston_price()

    def _loss_function(self, params):

        params = torch.tensor(params, device=self.device, requires_grad=True, dtype=torch.float32)
        kappa, theta, sigma, rho, v0 = params
        errors = []

        for _, row in self.data.iterrows():
            K = torch.tensor(row['Strike'], device=self.device, dtype=torch.float32)
            T = torch.tensor(row['Maturity'], device=self.device, dtype=torch.float32)
            market_price = torch.tensor(row['MarketPrice'], device=self.device, dtype=torch.float32)
            model_price = self._heston_price(K, T, kappa, theta, sigma, rho, v0)
            errors.append((market_price - model_price) ** 2)

        return torch.mean(torch.stack(errors))

    def calibrate(self, epochs=1000, learning_rate=0.01, initial_guess=None):

        if initial_guess is None:
            initial_guess = [2.0, 0.04, 0.5, -0.7, 0.04]  # kappa, theta, sigma, rho, v0

        params = torch.tensor(initial_guess, device=self.device, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([params], lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self._loss_function(params)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

        self.calibrated_params = params.detach().cpu().numpy()
        return self.calibrated_params

    def get_calibrated_params(self):

        if self.calibrated_params is None:
            raise ValueError("Model has not been calibrated yet.")
        return {
            'kappa': self.calibrated_params[0],
            'theta': self.calibrated_params[1],
            'sigma': self.calibrated_params[2],
            'rho': self.calibrated_params[3],
            'v0': self.calibrated_params[4]
        }

# Usage Example:
# Assuming a CSV file with columns: Strike, Maturity, MarketPrice
# calibrator = HestonCalibrator('option_data.csv', spot_price=100, risk_free_rate=0.05)
# calibrated_params = calibrator.calibrate(epochs=1000, learning_rate=0.01)
# print("Calibrated Parameters:", calibrator.get_calibrated_params())
