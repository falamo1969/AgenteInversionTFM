import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import csv
import datetime

#train_pickle = {'train_rewards': self.train_rewards,
#                'mean_train_rewards': self.mean_train_rewards,
#                'loss': self.train_losses,
#                'epsilon':self.l_epsilons}

#    agente_A2C_pickle = {'l_total_reward': l_total_reward,
#                            'l_exe_time': l_exe_time,
#                            'l_n_steps': l_n_steps,
#                            'l_info': l_info}

#        agente_DQN_pickle = {'l_total_reward': l_rewards,
#                             'info': l_info}

def plot_train_DQN(fnames, feature, labels=[], save='', path=['./logs/DQN/']):
    plt.figure(figsize=(8, 6))
    for j, f_name in enumerate(fnames):
        experiment = pickle.load(open(path+f_name,'rb'))
        x = [i for i in range(len(experiment[feature]))]
        y = experiment[feature]
        if labels!=[]:
            plt.plot(x, y, label=labels[j])
        else:
            plt.plot(x, y, label=f_name)

    plt.xlabel('Episodes')
    plt.title(feature)
    plt.tight_layout()
    plt.legend()
    if len(save)>=1:
        plt.savefig(save, format='png')
    plt.show()
    plt.close()

def plot_test(fnames, feature, path, labels=[], save=''):
    plt.figure(figsize=(8, 6))
    for j, f_name in enumerate(fnames):
        experiment = pickle.load(open(path + f_name,'rb'))
        x = [i for i in range(len(experiment[feature]))]
        y = experiment[feature]
        if len(labels)>0:
            lbl= labels[j]
        else:
            lbl=f_name
        plt.plot(x, y, label=lbl)
    plt.title(feature)
    plt.legend()
    plt.tight_layout()
    if len(save)>=1:
        plt.savefig(save, format='png')
    plt.show() 
    plt.close()

def plot_train_DQN_period(fnames, feature, labels=[], save='', path = './logs/DQN/'):
    plt.figure(figsize=(8, 6))
    for j, f_name in enumerate(fnames):
        experiment = pickle.load(open(path + f_name,'rb'))
        x = [i for i in range(len(experiment[feature]))]
        y = experiment[feature]
        if 'v3' in f_name and feature =='loss':
            y = [i/10 for i in y]            
        if labels!=[]:
            plt.plot(x, y, label=labels[j])
        else:
            plt.plot(x, y, label=f_name)

    plt.ylim(0,1500)
    plt.xlabel('Episodes')
    plt.title(feature)
    plt.legend()
    if len(save)>=1:
        plt.savefig(save, format='png')
    plt.show()
    plt.close()

def run_figures_trainLSTM(features, periodos, modelo):
    for feature in features:
        for periodo in periodos:
            labels = [modelo + '_v1_400_EPS', modelo +'_v2_300_EPS', modelo +'_v3_300_EPS']
            files = ['agentDQN-LSTM_'+ periodo +'_lstm_v1_400_EPS', 
                    'agentDQN-LSTM_'+ periodo +'_lstm_v2_300_EPS',
                    'agentDQN-LSTM_'+ periodo +'_lstm_v3_300_EPS']

            plot_train_DQN_period(files, feature, labels=labels, 
                                save = './figures/LSTM_'+periodo+'_'+feature+'_TRAIN.png', path='./logs/LSTM/')
            
def run_figures_trainDQN(features, periodos, modelo):
    for feature in features:
        for periodo in periodos:
            labels = [modelo + '_v1_200_EPS', modelo +'_v2_400_EPS', modelo +'_v3_200_EPS']
            files = ['agentDQN_'+ periodo +'_dqn_v1_200_EPS', 
                    'agentDQN_'+ periodo +'_dqn_v2_400_EPS',
                    'agentDQN_'+ periodo +'_dqn_v3_200_EPS']

#            plot_train_DQN_period(files, feature, labels=labels, path='./logs/DQN/')
            plot_train_DQN_period(files, feature, labels=labels, 
                                save = './figures/DQN_'+periodo+'_'+feature+'_TRAIN.png', path='./logs/DQN/')

 
def run_figures_test(periodos, modelo):
    path = './results/' + modelo + '/'
    files = os.listdir(path)

    for periodo in periodos:
        if modelo == 'DQN':
            pattern = "DQN_"+periodo+r"_dqn_v\d"
            matches = [fname for fname in files if re.search(pattern, fname)]
            labels = [label[14:20] for label in matches]
        else:
            pattern = "LSTM_"+periodo+r"_lstm_v\d+_"
            # Buscar los files que coinciden con el patrón
            matches = [fname for fname in files if re.search(pattern, fname)]
            labels = [label[15:22] for label in matches]


        plot_test(matches, 'l_total_reward', path=path, save = './figures/'+ modelo +'_'+periodo+'_reward_TEST.png', labels=labels)
        
# Configuración general    
features = ['loss', 'mean_train_rewards']
periodos = ['2008-2009', '2012-2014', '2018-2020', '2020-2022']
N_EPS_CALCULO = 50
modelo = 'LSTM'

#Configuración DQN
if modelo == 'DQN':
    path = './logs/DQN/'
    prefix = 'agentDQN_'
    suffixs = ['_dqn_v1_200_EPS', '_dqn_v2_400_EPS', '_dqn_v3_200_EPS']
elif modelo == 'LSTM':
    #Configuración LSTM
    path = './logs/LSTM/'
    prefix = 'agentDQN-LSTM_'
    suffixs = ['_lstm_v1_400_EPS', '_lstm_v2_300_EPS', '_lstm_v3_300_EPS']
    suffixs = ['_lstm_v4_300_EPS']



# Ruta del archivo CSV
fname_csv = path + modelo +'_resumen_train_v4.csv'
with open(fname_csv, 'w', newline='') as file:
    writer = csv.writer(file)

    cabecera= ["Periodo"] 
    for feature in features:
        cabecera += [feature, "Media", "Std"] 
    # Escribo cabecera
    writer.writerow(cabecera)

    for periodo in periodos:
        fnames = [prefix + periodo + suffix for suffix in suffixs] 

        for j, f_name in enumerate(fnames):
            experiment = pickle.load(open(path + f_name,'rb'))
            fila = [f_name]
            for feature in features:
                fila+=[feature]
                media = np.mean(experiment[feature][:-N_EPS_CALCULO])
                desv = np.std(experiment[feature][:-N_EPS_CALCULO])
                if (('v3' in f_name) or ('v4' in f_name)) and feature =='loss':
                    media /= 10
                    desv /= 10
                fila+=[media]
                fila+=[desv]
            writer.writerow(fila)        
        