# Batch de pruebas

import numpy as np
import time
import pickle
import csv
import gym


from datetime import datetime
from ETL_data import ETL_data_df

from AgentDQN_v1 import AgentDQN
from AgentDQN_LSTMV2 import AgentDQNLSTM_V2
from AgenteA2C import AgenteA2CTest, A2CTrainedAgentLoad
from AgentePPO import AgentePPOTest, PPOTrainedAgentLoad


fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
idx_l = [2927, 4388, 6580, 7310]
test_name_l = ['2008-2009', '2012-2014', '2018-2020', '2020-2022']
VERBOSE  = 0
N_TESTS = 10

inversion = 100000
cv_cost = 0.0025
custody_cost = 0.0015

now = datetime.now()
print("Empezando batch.", now.strftime("%d-%b %H:%M:%S"))

fname_csv = './results/resumen_resultados_portfolio_inicial.csv'
file_csv = open(fname_csv, 'w', newline='')
writer = csv.writer(file_csv)
writer.writerow(['Periodo', 'Modelo', 'Media', 'Desviación', 'Rendimiento'])

for idx, test_name in (zip(idx_l, test_name_l)):

    train_data = {}
    test_data = {}

    train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)
    l_rewards = []
    l_info = []
    l_modelo = []
    
    coste_unitario = test_data['ETF_prices'].iloc[0].sum()
    nETF = int(inversion / coste_unitario)
    portfolio = [nETF] * test_data['ETF_prices'].shape[1]
    env_test= gym.make('FinancialEconomicEnv-v0', fin_data_df=test_data['fin_data'], 
                                            eco_data_df=test_data['eco_data'],
                                            ETF_prices_df=test_data['ETF_prices'],
                                            initial_amount=inversion - coste_unitario*nETF, volume_trade = 1,
                                            trading_cost=cv_cost, custody_cost=custody_cost, portfolio=portfolio)

    state_size = env_test.get_obs_size()
    action_size = env_test.get_action_size()
    
    
    # MODELO DQN_V2
    DQN_agent = AgentDQN(env_test, state_size, action_size)
    DQN_agent.load_model('./Models/DQN/agentDQN_'+ test_name +'_dqn_v2_400_EPS')
    
    print("Test modelo DQN_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))

    for i in range(N_TESTS):
        now = datetime.now()
        test_reward, test_info = DQN_agent.test()
        l_modelo.append('DQN_V2')
        l_rewards.append(test_reward)
        l_info.append(test_info)
        if VERBOSE ==1:
            print("Episode {} Reward {:2f}".format(i, test_reward))
            print(test_info[-1])

    print("DQN_V2 :Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_reward), np.std(test_reward)))
    writer.writerow([test_name, 'DQN_v2', np.mean(test_reward), np.std(test_reward), np.mean(test_reward)/inversion*100])

    # MODELO LSTM
    LSTM_agent = AgentDQNLSTM_V2(env_test, state_size, action_size)
    LSTM_agent.load_model('./Models/LSTM/agentDQN-LSTM_'+ test_name +'_lstm_v2_300_EPS')
    print("Test modelo LSTM_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))

    for i in range(N_TESTS):
        now = datetime.now()
        test_reward, test_info = LSTM_agent.test()
        l_modelo.append('LSTM_V2')
        l_rewards.append(test_reward)
        l_info.append(test_info)
        if VERBOSE ==1:
            print("Episode {} Reward {:2f}".format(i, test_reward))
            print(test_info[-1])

    print("LSTM_V2 :Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_reward), np.std(test_reward)))
    writer.writerow([test_name, 'LSTM_V2', np.mean(test_reward), np.std(test_reward), np.mean(test_reward)/inversion*100])


    # MODELO A2C
    A2C_agent = A2CTrainedAgentLoad(test_name)
    print("Test modelo A2C_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))

    for i in range(N_TESTS):
        now = datetime.now()
        obs = env_test.reset()
        test_reward = 0
        reward = 0
        done = False

        start_time = time.time()
        while not done:
            action, _ = A2C_agent.predict(obs)
            obs, reward, done, test_info = env_test.step(action) #ejecución de la acción elegida
            test_reward += reward

        l_modelo.append('A2C_V1')
        l_rewards.append(test_reward)
        l_info.append(test_info)
        if VERBOSE ==1:
            print("Episode {} Reward {:2f}".format(i, test_reward))
            print(test_info[-1])

    print("A2C :Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_reward), np.std(test_reward)))
    writer.writerow([test_name, 'A2C_V1', np.mean(test_reward), np.std(test_reward), np.mean(test_reward)/inversion*100])


    # MODELO PPO
    PPO_agent = PPOTrainedAgentLoad(test_name + '_v3_LR_0.0005')
    print("Test modelo PPO_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))

    for i in range(N_TESTS):
        now = datetime.now()
        obs = env_test.reset()
        test_reward = 0
        reward = 0
        done = False

        start_time = time.time()
        while not done:
            action, _ = PPO_agent.predict(obs)
            obs, reward, done, test_info = env_test.step(action) #ejecución de la acción elegida
            test_reward += reward

        l_modelo.append('PPO')
        l_rewards.append(test_reward)
        l_info.append(test_info)
        if VERBOSE ==1:
            print("Episode {} Reward {:2f}".format(i, test_reward))
            print(test_info[-1])

    print("PPO :Media de total reward {:2f}, varianza {:2f}".format(np.mean(test_reward), np.std(test_reward)))
    writer.writerow([test_name, 'PPO_V3_LR_0.0005', np.mean(test_reward), np.std(test_reward), np.mean(test_reward)/inversion*100])


    test_modelos_pickle = {'l_modelo' : l_modelo,
                            'l_total_reward': l_rewards,
                            'l_info': l_info}         
    # Salvo resultados
    pickle.dump(test_modelos_pickle,open('./results/test_con_portfolio_'+ test_name +'.pickle', 'wb'))    
file_csv.close()