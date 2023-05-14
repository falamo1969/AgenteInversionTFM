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
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
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

env = gym.make('FinancialEconomicEnv-v0', fin_data_df=test_data['fin_data'], 
                                          eco_data_df=test_data['eco_data'],
                                          ETF_prices_df=test_data['ETF_prices'],
                                          initial_amount=inversion, volume_trade = 1,
                                          trading_cost=cv_cost, custody_cost=custody_cost)


model = A2C.load("Models/a2c_data_1E6.model")

# Evaluación del modelo
n_episodes = 10
l_exe_time =[]
l_total_reward = []
l_n_steps = []
l_info = []
N_STEPS_VERBOSE = 100
verbose = 1

for i in range(n_episodes):

    obs = env.reset()
    total_reward = 0
    done = False
    step = 0


    start_time = time.time()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action) #ejecución de la acción elegida
        total_reward += reward
        if ((step % N_STEPS_VERBOSE) == 0 ) & (verbose > 1):
            print("\n\nReward acumulada: {}".format(total_reward))
            print("Posición portfolio:")
            print(obs['portfolio'])
            print(info)
            print("\n")
        step +=1

    exe_time = (time.time()-start_time)/60
    l_exe_time.append(exe_time)
    l_total_reward.append(total_reward)
    l_n_steps.append(step)
    l_info.append(info)

    if verbose > 0:
        print("\n###########  INFO episodio {} #########".format(i))
        print(info)

print("\nEvolución del total reward:")
print("Media de total reward {:2f}, varianza {:2f}".format(np.mean(l_total_reward), np.std(l_total_reward)))


print("\nEvolución del tiempo de ejecución del entorno")
print("Media del tiempo de ejecución {:2f}, varianza {:2f}".format(np.mean(l_exe_time), np.std(l_exe_time)))

print("\nSteps para resolver cada episodio")
print("Media de pasos {:2f}, varianza {:2f}".format(np.mean(l_n_steps), np.std(l_n_steps)))

if verbose == 1:
    for i in range(n_episodes):
        print("\nInfo {}".format(i))
        print(l_info[i])


agente_A2C_pickle = {'l_total_reward': l_total_reward,
                        'l_exe_time': l_exe_time,
                        'l_n_steps': l_n_steps,
                        'l_info': l_info}

pickle.dump(agente_A2C_pickle,open('./results/agente_A2C_14-05_pickle.pickle', 'wb'))


        
