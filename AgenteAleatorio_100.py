import numpy as np
import pandas as pd
import time

import gym
import EntornoAgenteInversion
import pickle



data_df = pd.read_csv(".\data\datos_fake.csv")
data_df = data_df.drop(columns=['date'])
precios_ETFs_df = pd.read_csv(".\data\ETFs_fake.csv")
precios_ETFs_df=precios_ETFs_df.drop(columns=['date'])


inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015

verbose = 0

env = gym.make('FinancialEconomicEnv-v0', fin_data_df=data_df, precios_ETFs_df=precios_ETFs_df, initial_amount=inversion, volume_trade = 1,
                                          trading_cost=cv_cost, custody_cost=custody_cost)

if verbose == 1:
    print('Reward threshold: {}'.format(env.spec.reward_threshold))
    print('Reward range: {}'.format(env.reward_range))
    print('Maximum episodes steps: {}'.format(env.spec.max_episode_steps))
    print('Action space: {}'.format(env.action_space))
    print('Observation space: {}'.format(env.observation_space))

n_episodes = 10
l_exe_time =[]
l_total_reward = []
l_n_steps = []
l_info = []
N_STEPS_VERBOSE = 100

for i in range(n_episodes):

    obs = env.reset()
    total_reward = 0
    done = False
    step = 0
    verbose = 0

    start_time = time.time()
    while not done:
        action = env.action_space.sample() #acción aleatoria
        obs, reward, done, info = env.step(action) #ejecución de la acción elegida
        total_reward += reward
        if ((step % N_STEPS_VERBOSE) == 0 ) & (verbose ==1):
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
    print("\n###########  INFO episodio {} #########".format(i))
    print(info)

print("\nEvolución del total reward:")
print(l_total_reward)

print("\nEvolución del tiempo de ejecución del entorno")
print(l_exe_time)

print("\nSteps para resolver cada episodio")
print(l_n_steps)

print("\nInfo")
print(l_info)