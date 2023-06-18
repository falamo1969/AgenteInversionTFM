import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import datetime


modelo =  'LSTM'
raiz = './results/' + modelo + '/'
# Obtener la lista de archivos en el directorio
files = os.listdir(raiz)

# Ordenados por fecha 
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(raiz, x)))   
#files = ['LSTM_2018-2020_lstm_v3_11-Jun_23_48_18.pickle'] 

results = []
pd.DataFrame(columns=['Fichero', 'l_total_reward', 'mean_reward', 'std_reward', 'last portfolio'])

for fname in files:
    if os.path.isfile(os.path.join(raiz, fname)):
        data = pickle.load(open(raiz + fname,'rb'))
        result = {}
        result['Fichero'] = fname
        result ['date'] = datetime.datetime.fromtimestamp(os.path.getmtime(
            os.path.join(raiz, fname))).strftime("%d-%b_%H_%M_%S")
        result['l_total_reward'] = data['l_total_reward'].copy()   
        result['mean_reward'] = np.mean(data['l_total_reward'])
        result['std_reward'] = np.std(data['l_total_reward'])
        result['last portfolio'] = []
        if ('PPO' in fname) or ('A2C' in fname):
            if 'l_info' in data:
                for i in range(len(data['l_info'])):
                    result['last portfolio'].append(data['l_info'][i]['portfolio'])
        elif 'DQN' in fname:
            if 'info' in data:
                for i in range(len(data['info'])):
                    result['last portfolio'] = data['info'][:][i][-1]['portfolio'].copy()
        else:
            if 'l_info' in data:
                for i in range(len(data['l_info'])):
                    result['last portfolio'] = data['l_info'][:][i][-1]['portfolio'].copy()
        results.append(result)    

# Ruta del archivo CSV
fname_csv = './results/resultados_' + modelo +'v4.csv'

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
