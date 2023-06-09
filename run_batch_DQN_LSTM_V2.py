# Batch de pruebas

from datetime import datetime
from ETL_data import ETL_data_df
import numpy as np
import time
import pickle

from AgentDQN_LSTMV2 import AgentDQNLSTM_V2
import gym
import EntornoAgenteInversion

fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
#idx_l = [2927, 7310, 4388, 6580]
#test_name_l = ['2008-2009_lstm_v1', '2020-2022_lstm_v1', '2012-2014_lstm_v1', '2018-2020_lstm_v1']

idx_l = [2927, 7310, 4388, 6580]
test_name_l = ['2008-2009_lstm_v2', '2020-2022_lstm_v2', '2012-2014_lstm_v2', '2018-2020_lstm_v2']

now = datetime.now()
print("Empezando batch: ",now.strftime("%d-%b %H:%M:%S"))

################

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015
N_EPISODES = 300
VERBOSE = 1
TRAIN =  False
TEST =  True
N_TESTS = 10

for idx, test_name in (zip(idx_l, test_name_l)):

    train_data = {}
    test_data = {}
    f_name = "agentDQN-LSTM_" + test_name +"_" + str(N_EPISODES) + "_EPS"

    train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)

    if TRAIN == True:
        env = gym.make('FinancialEconomicEnv-v0', fin_data_df=train_data['fin_data'], 
                                                eco_data_df=train_data['eco_data'],
                                                ETF_prices_df=train_data['ETF_prices'],
                                                initial_amount=inversion, volume_trade = 1,
                                                trading_cost=cv_cost, custody_cost=custody_cost)


        state_size = env.get_obs_size()
        action_size = env.get_action_size()

        # Creo el modelo DQN-LSTM
        lstm_agent = AgentDQNLSTM_V2(env, state_size, action_size)

        # Entreno modelo DQN-LSTM
        now = datetime.now()
        print("Entrenando modelo DQN-LSTM_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
        lstm_agent.train(n_episodes= N_EPISODES, verbose=VERBOSE)

        # Salvo modelo DQN
        lstm_agent.save_model('Models/LSTM/'+f_name)
        
        # Salvo estadísticas
        lstm_agent.save_stats('logs/LSTM/'+f_name)

    if TEST == True:
        #########  TEST  ###########
        env_test= gym.make('FinancialEconomicEnv-v0', fin_data_df=test_data['fin_data'], 
                                                eco_data_df=test_data['eco_data'],
                                                ETF_prices_df=test_data['ETF_prices'],
                                                initial_amount=inversion, volume_trade = 1,
                                                trading_cost=cv_cost, custody_cost=custody_cost)

        state_size = env_test.get_obs_size()
        action_size = env_test.get_action_size()
        l_rewards = []
        l_info = []
        
        print("Test modelo DQN-LSTM_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))

        for i in range(N_TESTS):
            lstm_agent_test = AgentDQNLSTM_V2(env_test, state_size, action_size)
            lstm_agent_test.load_model('Models/LSTM/'+f_name)

            now = datetime.now()
            test_reward, test_info = lstm_agent_test.test()
            l_rewards.append(test_reward)
            l_info.append(test_info)
            if VERBOSE ==1:
                print("Episode {} Reward {:2f}".format(i, test_reward))
                print(test_info[-1])

        print("Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_reward), np.std(test_reward)))

        agente_DQN_LSTM_pickle = {'l_total_reward': l_rewards,
                                  'l_info': l_info}

        # Salvo resultados
        pickle.dump(agente_DQN_LSTM_pickle,open('./results/LSTM_'+ test_name + now.strftime("_%d-%b_%H_%M_%S")+ '.pickle', 'wb'))    
        
    
print("##############################################################")    
print("###################### FINALIZADO BACHT ######################")
print("##############################################################")    
