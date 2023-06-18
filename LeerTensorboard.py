import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import csv


#experiment_id = "./logs/1M/A2C_1M/A2C_4/events.out.tfevents.1684196776.LAPTOP-Fernis.8052.1"

# PPO
#{'images': [], 'audio': [], 'histograms': [], 
# 'scalars': ['time/fps', 'train/approx_kl', 'train/clip_fraction', 'train/clip_range', 'train/entropy_loss', 
# 'train/explained_variance', 'train/learning_rate', 'train/loss', 'train/policy_gradient_loss', 
# 'train/value_loss', 'rollout/ep_len_mean', 'rollout/ep_rew_mean'], 
# 'distributions': [], 'tensors': [], 'graph': False, 'meta_graph': False, 'run_metadata': []}


# A2C
#{'images': [], 'audio': [], 'histograms': [], 
# 'scalars': ['time/fps', 'train/entropy_loss', 'train/explained_variance', 'train/learning_rate', 'train/policy_loss', 
# 'train/value_loss', 'rollout/ep_len_mean', 'rollout/ep_rew_mean'], 
# 'distributions': [], 'tensors': [], 'graph': False, 'meta_graph': False, 'run_metadata': []}

def LeeLogFile(f_name, scalar='rollout/ep_rew_mean'):
    acc = EventAccumulator(f_name)
    acc.Reload()
    return pd.DataFrame(acc.Scalars(scalar))


def LeeLogPath(path, scalar='rollout/ep_rew_mean'):
    # Obtener la lista de archivos en el directorio
    files = os.listdir(path)

    # Ordenados por fecha 
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)))    
    pd_total = pd.DataFrame()
    i = 0
    # Iterar sobre la lista de archivos
    for file in files:
        # Comprobar si es un archivo
        i +=1
        if os.path.isfile(os.path.join(path, file)):
            # Leer el archivo
            pd_new = LeeLogFile(os.path.join(path, file), scalar=scalar)
            pd_total = pd.concat([pd_total, pd_new], ignore_index=True)
            #print (f'file {i} filename{file}')
    pd_total.reset_index(inplace=True)    
    return pd_total

def plot_log_train(features, features_fich, versiones, suffix, periodo, raiz, modelo, save = True):
    for feature, feature_fich in zip(features, features_fich):
        datos = LeeLogPath(raiz+versiones[0]+'/'+suffix, feature)
        logs = datos[['step', 'value']].copy()
        logs = logs.rename(columns={'value': versiones[0]})
        for logpath, columna in zip(versiones[1:], versiones[1:]):
            datos = LeeLogPath(raiz+logpath+'/'+suffix, feature)
            logs = pd.merge(logs, datos[['step', 'value']], on='step', how='outer')
            logs = logs.rename(columns={'value': columna})

        plt.figure(figsize=(8, 6))
        for version in versiones:
            plt.plot(logs['step'], logs[version], label=version)

        plt.legend()
        plt.title(modelo + ': ' + feature + ' '+ periodo)
        plt.xlabel('number of steps')
        if save:
            plt.savefig('./figures/' +modelo+ '/' + modelo +'_' + periodo +'_'+ feature_fich + '.png')
        else:
            plt.show() 
        plt.close()

def genera_csv_summary_train(modelo, directorios, periodos, suffixes, features, eps, raiz):
    ruta_csv = raiz + modelo +'_summary.csv'
    with open(ruta_csv, 'w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(["Version","Periodo", "feature", "Media", "Std"])
            
            # Escribir la lista en una fila del CSV
        for feature in features:
            for dir in directorios:
                for i, suffix in enumerate(suffixes):
                    datos = LeeLogPath(raiz+dir+'/'+suffix, feature)
                    logs = datos[['step', 'value']].copy()
                    media = logs['value'].tail(eps).mean()
                    desv = logs['value'].tail(eps).std()
                    print(f"Version {dir} Periodo {periodos[i]} {feature} Media: {media} Desviación {desv}")
                    writer.writerow([dir, periodos[i], feature, media, desv])
                writer.writerow([])
                writer.writerow([])
        writer.writerow([])


def plot_logs_modelo(modelo, raiz, periodos):
    if modelo == 'A2C':
        suffixes = ['A2C_1', 'A2C_2', 'A2C_3', 'A2C_4']
        versiones = ['A2C_1M', 'A2C_3M', 'A2C_LR_0.0005', 'A2C_LR_0.0010']
    elif modelo == 'PPO':
        suffixes = ['PPO_1', 'PPO_2', 'PPO_3', 'PPO_4']
        versiones = ['PPO_1M', 'PPO_3M', 'PPO_LR_0.0005', 'PPO_LR_0.0001']

    features = ['rollout/ep_rew_mean', 'train/value_loss']
    features_fich = ['mean_reward', ' value_loss']

    for periodo, suffix in zip(periodos, suffixes):
        plot_log_train(features, features_fich, versiones, suffix, periodo, raiz, modelo, True)
        

eps = 50
modelo = 'PPO'
raiz = './logs/'
periodos =['2008-2009', '2012-2014', '2018-2020', '2020-2022']

if modelo == 'A2C':
    suffixes = ['A2C_1', 'A2C_3', 'A2C_4', 'A2C_2']
    versiones = ['A2C_1M', 'A2C_3M', 'A2C_LR_0.0005', 'A2C_LR_0.0010']
elif modelo == 'PPO':
    suffixes = ['PPO_1', 'PPO_3', 'PPO_4', 'PPO_2']
    versiones = ['PPO_1M', 'PPO_3M', 'PPO_LR_0.0005', 'PPO_LR_0.0001']

features = ['train/value_loss', 'rollout/ep_rew_mean']
features_fich = ['mean_reward', ' value_loss']

genera_csv_summary_train(modelo, versiones, periodos, suffixes, features, eps, raiz)
