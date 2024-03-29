import numpy as np
import time
import pickle
import os

import gym
import EntornoAgenteInversion


#stable_baselines 3
from stable_baselines3 import A2C
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_checker import check_env


def AgenteA2CTrain(model_name, train_data, total_timesteps=1000000,n_loops = 1,  inversion = 100000, cv_cost = 0.0025, custody_cost = 0.0015, verbose = 0):
    # Creo el entorno
    env = gym.make('FinancialEconomicEnv-v0', fin_data_df=train_data['fin_data'], 
                                            eco_data_df=train_data['eco_data'],
                                            ETF_prices_df=train_data['ETF_prices'],
                                            initial_amount=inversion, volume_trade = 1,
                                            trading_cost=cv_cost, custody_cost=custody_cost)

    # Check para el funcionamiento del entorno custom
    #check_env(env, True, True)
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Para el monitor de TensorBoard
    env =  Monitor(env, log_dir)

    model = A2C(MultiInputPolicy, env, verbose=0, tensorboard_log=log_dir)

    if verbose > 0:
        print("Empezando a entrenar modelo A2C_{}".format(model_name))
    
    for i in range(1,n_loops+1):
        model.learn(total_timesteps, reset_num_timesteps=False, progress_bar=True)
        model.save(f"Models/A2C_{model_name}_{i*total_timesteps}_data.model")
    
    if verbose > 0:
        print("Fin entrenamiento y salvando modelo A2C_{}".format(model_name))
    
    model.save("Models/A2C_"+ model_name +"_data.model")
    return model


def AgenteA2CTest(model, test_data, results_file, n_episodes=10, inversion = 100000, cv_cost = 0.0025, custody_cost = 0.0015, verbose = 0):
    # Creo el entorno
    env = gym.make('FinancialEconomicEnv-v0', fin_data_df=test_data['fin_data'], 
                                            eco_data_df=test_data['eco_data'],
                                            ETF_prices_df=test_data['ETF_prices'],
                                            initial_amount=inversion, volume_trade = 1,
                                            trading_cost=cv_cost, custody_cost=custody_cost)


    # Evaluación del modelo
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
            action, _ = model.predict(obs)
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


    agente_A2C_pickle = {'l_total_reward': l_total_reward,
                            'l_exe_time': l_exe_time,
                            'l_n_steps': l_n_steps,
                            'l_info': l_info}

    # Salvo resultados
    pickle.dump(agente_A2C_pickle,open('./results/A2C_'+ results_file +'_pickle.pickle', 'wb'))
    
def A2CTrainedAgentLoad(model_name):
    fname = "./Models/A2C_" + model_name +"_data.model"
    model = A2C.load(fname, env=None)
    return model