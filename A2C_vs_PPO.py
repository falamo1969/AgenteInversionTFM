# Comparación de tests

import pickle
import matplotlib.pyplot as plt

def plot_result(name_l, result, titulo):
    n_plots = len(name_l) 
    fig, axs = plt.subplots(n_plots, figsize=(12, 8))
    fig.suptitle(titulo)
    for n_ax, f_name in enumerate(test_name_l):
        A2C_results = pickle.load(open('./results/A2C_'+ f_name +'_pickle.pickle', 'rb'))
        PPO_results = pickle.load(open('./results/PPO_'+ f_name +'_pickle.pickle', 'rb'))
        x = [i for i in range(len(A2C_results[result]))]
        axs[n_ax].plot(x, A2C_results[result], label='A2C')
        axs[n_ax].plot(x, PPO_results[result], label='PPO')
        axs[n_ax].set_xlabel('Test Episodes')
        axs[n_ax].set_ylabel('Reward (EUR)')
        axs[n_ax].set_title('Test-period ' + f_name)
        axs[n_ax].legend()
        
    for ax in fig.get_axes():
        ax.label_outer()
    plt.show()    
        
        
    
    
    


test_name_l = ['2008-2009', '2020-2022', '2012-2014', '2018-2020']
resultados  = ['l_total_reward', 'l_exe_time']
for result in resultados:
    plot_result(test_name_l, result, "Total reward")




