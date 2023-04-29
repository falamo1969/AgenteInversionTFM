import numpy as np
import pandas as pd
import time

import gym
import EntornoAgenteInversion



data_df = pd.read_csv(".\data\datos_fake.csv")
data_df = data_df.drop(columns=['date'])
precios_ETFs_df = pd.read_csv(".\data\ETFs_fake.csv")
precios_ETFs_df=precios_ETFs_df.drop(columns=['date'])


portfolio = [0 for i in range(precios_ETFs_df.shape[1])]

inversion = 10000
cv_cost = 0.0025
custody_cost = 0.0015

env = gym.make('FinancialEconomicEnv-v0', fin_data_df=data_df, precios_ETFs_df=precios_ETFs_df, initial_amount=inversion, volume_trade = 1,
                                          trading_cost=cv_cost, custody_cost=custody_cost)

print('Reward threshold: {}'.format(env.spec.reward_threshold))
print('Reward range: {}'.format(env.reward_range))
print('Maximum episodes steps: {}'.format(env.spec.max_episode_steps))
print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))


obs = env.reset()
total_reward = 0
done = False
step = 0

start_time = time.time()

actions =  [[2,2,2,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0],
            [2,0,0,0,0,0,0,0,0,0,2],
            [0,0,1,0,1,0,1,0,0,0,2],
            [1,0,1,0,1,0,1,0,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,1]
          ]



for action in actions:
    obs, reward, done, info = env.step(action) #ejecución de la acción elegida
    total_reward += reward
    print("\n\nReward acumulada: {}".format(total_reward))
    print("Posición portfolio:")
    print(obs['portfolio'])
    print(info)


while not done:
    action = env.action_space.sample() #acción aleatoria
    obs, reward, done, info = env.step(action) #ejecución de la acción elegida
    total_reward += reward
    if (step % 50) == 0:
        print("\n\nReward acumulada: {}".format(total_reward))
        print("Posición portfolio:")
        print(obs['portfolio'])
        print(info)
        print("\n")
    step +=1

end_time = time.time()
train_time = (end_time-start_time)/60

print('Training time: {:.2f} minutes'.format(train_time))
