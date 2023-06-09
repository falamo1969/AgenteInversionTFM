import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os


#experiment_id = "./logs/1M/A2C_1M/A2C_4/events.out.tfevents.1684196776.LAPTOP-Fernis.8052.1"

def LeeLogFile(f_name, scalar='rollout/ep_rew_mean'):
    acc = EventAccumulator(f_name)
    acc.Reload()
    return pd.DataFrame(acc.Scalars(scalar))


def LeeLogPath(path):
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
            pd_new = LeeLogFile(os.path.join(path, file))
            pd_total = pd.concat([pd_total, pd_new], ignore_index=True)
            print (f'file {i} filename{file}')
    pd_total.reset_index(inplace=True)    
    return pd_total

#path='./logs/10M/A2C_17/'
path='./logs/3M/PPO_3M/PPO_31/'
#path= './logs/A2C_lr_0.0005_2/'
kk = LeeLogPath(path)


x = kk['step']
y = kk['value']
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Recompensa')
plt.legend()
plt.show() 
plt.close()

