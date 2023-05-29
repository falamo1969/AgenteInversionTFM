# Batch de pruebas

from datetime import datetime
from ETL_data import ETL_data_df
import numpy as np

from AgentDQN import AgentDQN
import gym
import EntornoAgenteInversion

def flatten_state(state):
    state_flat = np.zeros(0)
    if isinstance(state, dict):
        for (_, substate) in state.items():
            state_flat = np.concatenate((state_flat, np.array(substate)))
    else:
        state_flat = state
    return state_flat    


state_flat = np.zeros(0)


fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
#idx_l = [2927, 7310, 4388, 6580]
#test_name_l = ['2008-2009_v2', '2020-2022_v2', '2012-2014_v2', '2018-2020_v2']
idx = 7310
idx_l = [7310]
test_name_l = ['2020-2022_v3']
verbose  = 1

now = datetime.now()
print("Empezando batch.", now.strftime("%d-%b %H:%M:%S"))


################

# Creo el entorno
train_data = {}
test_data = {}

train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015

env = gym.make('FinancialEconomicEnv-v0', fin_data_df=train_data['fin_data'], 
                                        eco_data_df=train_data['eco_data'],
                                        ETF_prices_df=train_data['ETF_prices'],
                                        initial_amount=inversion, volume_trade = 1,
                                        trading_cost=cv_cost, custody_cost=custody_cost)


state_size = env.get_obs_size()
action_size = env.get_action_size()
agent = AgentDQN(state_size, action_size)

for episode in range(100):
    state = env.reset()
    state = flatten_state(state)
    done = False
    total_reward = 0
    step = 1
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        if not done:
            next_state = flatten_state(next_state)
            agent.update_q_network(state, action, reward, next_state, done)
            state = next_state
        total_reward += reward
        if (step%500==0): 
            print(f"Step {step}/{train_data['fin_data'].shape[0]}")
        step+=1
        
    print(f"Episode {episode+1} - Total Reward: {total_reward}")
###########

for idx, test_name in (zip(idx_l, test_name_l)):

    train_data = {}
    test_data = {}

    train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)

    #Entreno modelo A2C
    now = datetime.now()
    print("Entrenando modelo A2C_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
#    modelA2C = AgenteA2CTrain(test_name, train_data, total_timesteps=1000000, n_loops=10,verbose = verbose)

    #Test modelo A2C
    now = datetime.now()
    print("Test modelo A2C_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
#    AgenteA2CTest(modelA2C, test_data, test_name, verbose = verbose)
    
print("##############################################################")    
print("###################### FINALIZADO BACHT ######################")
print("##############################################################")    
