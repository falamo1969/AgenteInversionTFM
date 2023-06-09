# Comparación de tests

import pickle
import matplotlib.pyplot as plt

def plot_result(test_name_l, result, titulo):
    n_plots = len(test_name_l) 
    fig, axs = plt.subplots(n_plots, figsize=(12, 8))
    fig.suptitle(titulo)
    for n_ax, f_name in enumerate(test_name_l):
        v1_results = pickle.load(open('./results/A2C_'+ f_name + '_pickle.pickle', 'rb'))
        v2_results = pickle.load(open('./results/A2C_'+ f_name + '_v2_pickle.pickle', 'rb'))
        v3_results = pickle.load(open('./results/A2C_'+ f_name + '-09-Jun_v1_pickle.pickle', 'rb'))
        v4_results = pickle.load(open('./results/PPO_'+ f_name + '_pickle.pickle', 'rb'))
        v5_results = pickle.load(open('./results/PPO_'+ f_name + '_v2_pickle.pickle', 'rb'))
        v6_results = pickle.load(open('./results/PPO_'+ f_name + '-09-Jun_v1_pickle.pickle', 'rb'))
        x = [i for i in range(len(v1_results[result]))]
        axs[n_ax].plot(x, v1_results[result], label='A2C_v1')
        axs[n_ax].plot(x, v2_results[result], label='A2C_v2')
        axs[n_ax].plot(x, v3_results[result], label='A2C_09-Jun_v1')
        axs[n_ax].plot(x, v4_results[result], label='PPO_v1')
        axs[n_ax].plot(x, v5_results[result], label='PPO_v2')
        axs[n_ax].plot(x, v6_results[result], label='PPO_09-Jun_v1')
        axs[n_ax].set_xlabel('Test Episodes')
        axs[n_ax].set_ylabel('Reward (EUR)')
        axs[n_ax].set_title('Test-period ' + f_name)
        axs[n_ax].legend()
        
    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()    
        
        
    
    
    


test_name_l = ['2008-2009', '2020-2022', '2012-2014', '2018-2020']
resultados  = 'l_total_reward'
plot_result(test_name_l, resultados, "Comparativa A2C y PPO \n episodios entrenamiento")




