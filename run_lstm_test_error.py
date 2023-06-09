# Batch de pruebas

from datetime import datetime
from ETL_data import ETL_data_df
import numpy as np
import time
import pickle

from AgentDQN_LSTM import AgentDQNLSTM
import gym
import EntornoAgenteInversion

fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
idx_l = [2927, 7310, 4388, 6580]
test_name_l = ['2008-2009_lstm_v1', '2020-2022_lstm_v1', '2012-2014_lstm_v1', '2018-2020_lstm_v1']

now = datetime.now()
print("Empezando batch: ",now.strftime("%d-%b %H:%M:%S"))

################

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015
N_EPISODES = 400
VERBOSE = 1
train_data = {}
test_data = {}

idx = 2927
test_name = '2008-2009_lstm_v1'

train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)



f_name = "agentDQN-LSTM_" + test_name +"_" + str(N_EPISODES) + "_EPS"


#########  TEST  ###########
env_test= gym.make('FinancialEconomicEnv-v0', fin_data_df=test_data['fin_data'], 
                                        eco_data_df=test_data['eco_data'],
                                        ETF_prices_df=test_data['ETF_prices'],
                                        initial_amount=inversion, volume_trade = 1,
                                        trading_cost=cv_cost, custody_cost=custody_cost)

state_size = env_test.get_obs_size()
action_size = env_test.get_action_size()

lstm_agent_test = AgentDQNLSTM(env_test, state_size, action_size)
lstm_agent_test.load_model('Models/LSTM/'+f_name)

now = datetime.now()
print("Test modelo DQN-LSTM_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
test_rewards = lstm_agent_test.test()
print("Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_rewards), np.std(test_rewards)))

agente_DQN_LSTM_pickle = {'l_total_reward': test_rewards}

# Salvo resultados
pickle.dump(agente_DQN_LSTM_pickle,open('./results/LSTM_'+ test_name +'.pickle', 'wb'))    