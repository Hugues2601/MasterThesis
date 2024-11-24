import pandas as pd
import torch
from datetime import datetime
from HestonClosedForm.HestonVanilla import HestonVanilla
from config import CONFIG

class HestonCalibrator:
    def __init__(self, csv_path, spot_price, risk_free_rate, heston_model_class=HestonVanilla):
        self.data = pd.read_csv(CONFIG.CSV_PATH)
        self.S0 = torch.tensor(spot_price, dtype=torch.float32, device="cuda")
        self.r = torch.tensor(risk_free_rate, dtype=torch.float32, device="cuda")
        self.heston_model_class = heston_model_class
        self.calibrated_params = None

    def _prepare_data(self):
        today = datetime.today()
        self.data["T"] = (pd.to_datetime(self.data["Expiration"]) - today).dt.days / 365.0
        self.K = torch.tensor(self.data["strike"].values, dtype=torch.float32, device="cuda")
        self.T = torch.tensor(self.data["T"].values, dtype=torch.float32, device="cuda")
        self.market_prices = torch.tensor(self.data["lastPrice"].values, dtype=torch.float32, device="cuda")

    def _loss_function(self, params):
        kappa, theta, sigma, rho, v0 = params
        model = self.heston_model_class(self.S0, self.K, self.T, self.r, kappa, v0, theta, sigma, rho)
        model_prices = model.heston_price()
        loss = torch.mean((self.market_prices - model_prices) ** 2)
        return loss

    def calibrate(self, epochs=1000, learning_rate=0.01, initial_guess=None):
        self._prepare_data()

        if initial_guess is None:
            initial_guess = [2.0, 0.02, 0.2, -0.5, 0.04]  # kappa, theta, sigma, rho, v0

        params = torch.tensor(initial_guess, dtype=torch.float32, requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([params], lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self._loss_function(params)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

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
