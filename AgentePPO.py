import numpy as np
import pandas as pd
import time

import gym
import EntornoAgenteInversion

#stable_baselines 3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

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


#check_env(env, True, True)

# Selecciono el modelo y la política
model = PPO(MultiInputPolicy, env, verbose=1)

# Sin entrenar
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

#print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



#model.learn(total_timesteps=precios_ETFs_df.shape[0])
#model.save("Models/PPO.model")

#del model # remove to demonstrate saving and loading

model = PPO.load("Models/PPO.model")

obs = env.reset()
i = 0
NUM_VERBOSE = 100
done = False
tot_rewards  = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    tot_rewards += reward
    i +=1
    if i % NUM_VERBOSE == 0 :
        print("##############  {} iteration  #######".format(i))
        print("Remaining  cash {:2f}\nPortfolio valuation {:.2f}\n".format(info['cash'], info["porfolio_value"]))
        print("Portfolio composition:")
        print(obs["portfolio"])
        print("Info:")
        print(info)
        check = info["porfolio_value"] - info["plus-minus"]
        print(check)
        print("\n")


print("Total reward: {:.2f}".format(tot_rewards))
print("Remaining cash {:2f}\nPortfolio valuation {:.2f}\n".format(info['cash'], info["porfolio_value"]))
print("Portfolio composition:")
print(info)


        
