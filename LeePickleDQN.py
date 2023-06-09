import pickle
import matplotlib.pyplot as plt
import numpy as np


#train_pickle = {'train_rewards': self.train_rewards,
#                'mean_train_rewards': self.mean_train_rewards,
#                'loss': self.train_losses,
#                'epsilon':self.l_epsilons}


def plot_train_DQN(fnames, feature, labels=[], save='', path='./logs/DQN/'):
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

def plot_test(fnames, feature):
    plt.figure(figsize=(8, 6))
    for f_name in fnames:
        experiment = pickle.load(open('./results/' + f_name + '.pickle','rb'))
        x = [i for i in range(len(experiment[feature]))]
        y = experiment[feature]
        print('\n')
        print(y)
        print(f"Mean {np.mean(y)}, varianza {np.std(y)}")
        plt.plot(x, y, label=f_name)
    plt.title(feature)
    plt.legend()
    plt.tight_layout()
    plt.show() 
    plt.close()


files_DQN_train_200 = ['agentDQN_2008-2009_dqn_v1_200_EPS', 'agentDQN_2012-2014_dqn_v1_200_EPS',
                   'agentDQN_2018-2020_dqn_v1_200_EPS','agentDQN_2020-2022_dqn_v1_200_EPS']

files_DQN_train_400 = ['agentDQN_2008-2009_dqn_v2_400_400_EPS', 'agentDQN_2012-2014_dqn_v2_400_400_EPS',
                   'agentDQN_2018-2020_dqn_v2_400_400_EPS','agentDQN_2020-2022_dqn_v2_400_400_EPS']

files_DQN_test_200 = ['DQN_2008-2009_dqn_v1', 'DQN_2012-2014_dqn_v1',
                   'DQN_2018-2020_dqn_v1','DQN_2020-2022_dqn_v1']

files_DQN_test_400 = ['DQN_2008-2009_dqn_v2_400', 'DQN_2012-2014_dqn_v2_400',
                   'DQN_2018-2020_dqn_v2_400','DQN_2020-2022_dqn_v2_400']

files_LSTM_train_v1 = ['agentDQN-LSTM_2008-2009_lstm_v1_400_EPS', 'agentDQN-LSTM_2012-2014_lstm_v1_400_EPS',
                        'agentDQN-LSTM_2018-2020_lstm_v1_400_EPS','agentDQN-LSTM_2020-2022_lstm_v1_400_EPS']

files_LSTM_train_v2 = ['agentDQN-LSTM_2008-2009_lstm_v2_300_EPS', 'agentDQN-LSTM_2012-2014_lstm_v2_300_EPS',
                        'agentDQN-LSTM_2018-2020_lstm_v2_300_EPS','agentDQN-LSTM_2020-2022_lstm_v2_300_EPS']

files_LSTM_test = ['LSTM_2008-2009_lstm_v1', 'LSTM_2012-2014_lstm_v1',
                   'LSTM_2018-2020_lstm_v1','LSTM_2020-2022_lstm_v1']

files_LSTM_test_400 = ['LSTM_2008-2009_lstm_v2', 'LSTM_2012-2014_lstm_v2',
                   'LSTM_2018-2020_lstm_v2','LSTM_2020-2022_lstm_v2']

#plot_test(files_LSTM_test + files_LSTM_test_400, 'l_total_reward')
#feature = 'mean_train_rewards'
#plot_train_DQN(files_LSTM_train_v2, feature, ['2008-2009', '2012-2014', '2018-2020', '2020-2022'], 
#               save='./figures/fig_LSTM_'+feature+'_300.png', path='./logs/LSTM/')



# LEER LOS PORTFOLIOS DE LOS TEST

files_results = ['DQN_2008-2009_dqn_v2_400-09-Jun.pickle', 'DQN_2020-2022_dqn_v2_400-09-Jun.pickle', 
                 'DQN_2012-2014_dqn_v2_400-09-Jun.pickle', 'DQN_2018-2020_dqn_v2_400-09-Jun.pickle']

info0 = pickle.load(open('./results/' + files_results[0] ,'rb'))
info1 = pickle.load(open('./results/' + files_results[1] ,'rb'))
info2 = pickle.load(open('./results/' + files_results[2] ,'rb'))
info3 = pickle.load(open('./results/' + files_results[3] ,'rb'))
print('kk')

