import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import datetime


raiz = './results/' 
# Obtener la lista de archivos en el directorio
#files = os.listdir(raiz)

# Ordenados por fecha 
#files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(raiz, x))) 
periodos = ['2008-2009', '2020-2022', '2012-2014', '2018-2020']  
prefixes = ['DQN/DQN_', 'LSTM/LSTM_', 'A2C/A2C_','PPO/PPO_'] 
suffixes = ['_dqn_v2_400-09-Jun.pickle', '_lstm_v2_14-Jun.pickle', '-09-Jun_v1_pickle.pickle','_v3_LR_0.0005.pickle', ]

results = []
for periodo in periodos:
    files = [raiz+prefix+periodo+suffix for prefix, suffix in zip(prefixes, suffixes)]

    pd.DataFrame(columns=['Fichero', 'l_total_reward', 'mean_reward', 'std_reward', 'last portfolio'])

    for fname in files:
        if os.path.isfile(fname):
            data = pickle.load(open(fname,'rb'))
            result = {}
            result['Fichero'] = fname
            result ['date'] = datetime.datetime.fromtimestamp(os.path.getmtime(
                fname)).strftime("%d-%b_%H_%M_%S")
            result['l_total_reward'] = data['l_total_reward'].copy()   
            result['mean_reward'] = np.mean(data['l_total_reward'])
            result['std_reward'] = np.std(data['l_total_reward'])
            result['last portfolio'] = []
            if ('PPO' in fname) or ('A2C' in fname):
                if 'l_info' in data:
                    result['last portfolio'].append(data['l_info'][len(data['l_info'])-1]['portfolio'])
            elif 'DQN' in fname:
                if 'info' in data:
                    result['last portfolio'].append(data['info'][len(data['info'])-1][-1]['portfolio'])
            else:
                if 'l_info' in data:
                    result['last portfolio'].append(data['l_info'][len(data['l_info'])-1][-1]['portfolio'])
            results.append(result)    

# Ruta del archivo CSV
fname_csv = './results/resultados_4 modelos.csv'

# Obtener las claves de los diccionarios
keys = results[0].keys()

# Abrir el archivo CSV en modo escritura
with open(fname_csv, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys)

    # Escribir las claves como encabezados en la primera fila del archivo CSV
    writer.writeheader()

    # Escribir los valores de cada diccionario en filas sucesivas del archivo CSV
    for dic in results:
        writer.writerow(dic)


ETF_names = ['ETF-CONSUMER', 'ETF-CONSUMER-BASIS', 'ETF-ENERGY', 'ETF-FINANCIAL', 'ETF-HEALTH-CARE', 'ETF-INDUSTRIAL',
             'ETF-MATERIALS', 'ETF-REAL-STATE', 'ETF-TECHNOLOGY', 'ETF-UTILITIES']

suffixes = ['_dqn_v2_400-09-Jun.pickle', '_lstm_v2_14-Jun.pickle', '-09-Jun_v1_pickle.pickle','_v3_LR_0.0005.pickle', ]

modelos = ['DQN_V2', 'DQN-LSTM_V2', 'A2C_V1', 'PPO_LR_0.0005']

for j, periodo in enumerate(periodos):
    fig, ax = plt.subplots()

    n = 10
    x = np.arange(n)
    width = 1/5


    for i, modelo in enumerate(modelos):
        plt.bar(x + (i-1.5) * width, results[4*j+i]['last portfolio'][0], width=width, label = modelo)

    plt.xticks(x, ETF_names, rotation='vertical')
    plt.ylabel("Posición")
    plt.title("Portfolio final " + periodo)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./figures/comparativa_portfolio_final_'+periodo+'.png', format='png')    
    plt.show()
