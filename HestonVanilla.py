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