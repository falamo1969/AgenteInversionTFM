import numpy as np
import pandas as pd
import time
import pickle
import os
import matplotlib.pyplot as plt
from Callbacks_module import MyCallback

import gym
import EntornoAgenteInversion

from ETL_data import ETL_data_df

#stable_baselines 3
from stable_baselines3 import A2C
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

#from wandb.integration.sb3 import WandbCallback

fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"


train_data = {}
test_data = {}

train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl)

portfolio = [0 for i in range(train_data['ETF_prices'].shape[1])]

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015

env = gym.make('FinancialEconomicEnv-v0', fin_data_df=train_data['fin_data'], 
                                          eco_data_df=train_data['eco_data'],
                                          ETF_prices_df=train_data['ETF_prices'],
                                          initial_amount=inversion, volume_trade = 1,
                                          trading_cost=cv_cost, custody_cost=custody_cost)

#print('Reward threshold: {}'.format(env.spec.reward_threshold))
#print('Reward range: {}'.format(env.reward_range))
#print('Maximum episodes steps: {}'.format(env.spec.max_episode_steps))
#print('Action space: {}'.format(env.action_space))
#print('Observation space: {}'.format(env.observation_space))


#check_env(env, True, True)

# Selecciono el modelo y la política
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
env =  Monitor(env, log_dir)

model = A2C(MultiInputPolicy, env, verbose=2, tensorboard_log=log_dir)

# Sin entrenar
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

#print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Train the model
#model.learn(n_episodes_train*n_steps_max, progress_bar=True,callback=WandbCallback())

#callback=MyCallback()
print("Empezando a entrenar")
model.learn(total_timesteps=1000000, progress_bar=True)
print("Fin entrenamiento y salvando modelo")
model.save("Models/a2c_data_1E6.model")