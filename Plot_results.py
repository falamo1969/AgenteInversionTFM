import matplotlib
import numpy as np
import pandas as pd
import time

import gym
import EntornoAgenteInversion


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy



data_df = pd.read_csv(".\data\datos_fake.csv")
data_df = data_df.drop(columns=['date'])
precios_ETFs_df = pd.read_csv(".\data\ETFs_fake.csv")
precios_ETFs_df=precios_ETFs_df.drop(columns=['date'])


portfolio = [0 for i in range(precios_ETFs_df.shape[1])]

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015


env = gym.make('FinancialEconomicEnv-v0', fin_data_df=data_df, precios_ETFs_df=precios_ETFs_df, initial_amount=inversion, volume_trade = 1,
                                          trading_cost=cv_cost, custody_cost=custody_cost)


# Selecciono el modelo y la política
model = PPO.load("Models/PPO_fake_data.model")


print(model)
