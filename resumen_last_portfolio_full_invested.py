import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import datetime


#    test_modelos_pickle = {'l_modelo' : l_modelo,
#                            'l_total_reward': l_rewards,
#                            'l_info': l_info}         
    # Salvo resultados

raiz = './results/' 
#files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(raiz, x))) 

modelos = ['DQN_V2', 'DQN_LSTM_V2', 'A2C_V1', 'PPO_LR_0.0005']
periodos = ['2008-2009', '2020-2022', '2012-2014', '2018-2020']  
ETF_names = ['ETF-CONSUMER', 'ETF-CONSUMER-BASIS', 'ETF-ENERGY', 'ETF-FINANCIAL', 'ETF-HEALTH-CARE', 'ETF-INDUSTRIAL',
             'ETF-MATERIALS', 'ETF-REAL-STATE', 'ETF-TECHNOLOGY', 'ETF-UTILITIES']

results = []
for periodo in periodos:
    fname = raiz + 'test_con_portfolio_' + periodo + '.pickle'

#    pd.DataFrame(columns=['Fichero', 'l_total_reward', 'mean_reward', 'std_reward', 'last portfolio'])
    portfolios = []
    if os.path.isfile(fname):
        data = pickle.load(open(fname,'rb'))
        for i in range(len(data['l_modelo'])):
            if (i+1) % 10 == 0: 
                if ('DQN' in data['l_modelo'][i]) or ('LSTM' in data['l_modelo'][i]):
                    portfolios.append(data['l_info'][i][-1]['portfolio'])
                else:
                    portfolios.append(data['l_info'][i]['portfolio'])


    fig, ax = plt.subplots()
    n = 10
    x = np.arange(n)
    width = 1/5

    for i, modelo in enumerate(modelos):
        plt.bar(x + (i-1.5) * width,  portfolios[i], width=width, label = modelo)

    plt.xticks(x, ETF_names, rotation='vertical')
    plt.ylabel("Posición")
    plt.title("Portfolio final " + periodo)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./figures/comparativa_portfolio_final_full_invested_'+periodo+'.png', format='png')    
    plt.show()
