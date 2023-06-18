from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pickle
import time

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values =self.fc3(x)
        return q_values
    
class AgentDQN:
    def __init__(self, env, state_size, action_size, lr=0.001, epsilon=0.1):
        self.env= env
        self.state_size = state_size
        self.action_size = action_size[0]
        self.n_assets = action_size[1]
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = []
        self.next_state = []
        self.epsilon = epsilon
        self.total_reward=0
        self.LOSS_RANGE = 1E6
        self.loss_outsider = 0
        # Inicializa las redes y estadísticas    
        self._initialize()

    def flatten_state(self, state):
        state_flat = np.zeros(0)
        if isinstance(state, dict):
            for (_, substate) in state.items():
                state_flat = np.concatenate((state_flat, np.array(substate)))
        else:
            state_flat = state
        return state_flat    

    def _initialize(self):
        # Inicialización de las redes
        self.main_network = [QNetwork(self.state_size, self.action_size).to(self.device) for _ in range(self.n_assets)]
        self.target_network= deepcopy(self.main_network)
        params =  []
        for i in range(self.n_assets):
            params += list(self.main_network[i].parameters())
            
        self.optimizer = optim.Adam(params=params, lr = self.lr)

        # Variables estadísticas
        self.l_epsilons = []
        self.train_rewards = []
        self.train_losses = []
        self.update_loss = []
        self.mean_train_rewards = []

    def get_action(self):
        action = []
        for i in range(self.n_assets):
            with torch.no_grad():
                state = torch.FloatTensor(self.state).to(self.device)
                q_values = self.main_network[i](state)
                action.append(torch.argmax(q_values))
        return action
    
    def update_main_network(self, action, reward, done):
        state = torch.FloatTensor(self.state).to(self.device)
        if not done:
            next_state = torch.FloatTensor(self.next_state).to(self.device)
        loss = torch.empty((), dtype=torch.float)
        self.optimizer.zero_grad()
        for i in range(self.n_assets):
            q_values = self.main_network[i](state)
            q_value = q_values[action[i]]
              
            with torch.no_grad():
                if done:
                    next_q_value = torch.tensor(0.0, dtype=torch.float, device=self.device)
                else:
                    next_q_values = self.target_network[i](next_state)
                    next_q_value = torch.max(next_q_values)

                target_q_value = reward + self.gamma * next_q_value
                
            loss += nn.functional.smooth_l1_loss(q_value, target_q_value)
        loss.backward()
        self.optimizer.step()

        # Guardamos los valores de pérdida quitando outsiders
        if (loss < self.LOSS_RANGE) & (loss >-self.LOSS_RANGE):
            if self.device != 'cuda':
                self.update_loss.append(loss.detach().cpu().numpy())
            else:
                self.update_loss.append(loss.detach().numpy())
        else:
            self.loss_outsider +=1          

    def step(self, mode='train'):
        if (mode == 'explore') or ((mode=='train') & (random.uniform(0, 1) < self.epsilon)):
            action = [random.randint(0, self.action_size -1) for _ in range(self.n_assets)] # acción aleatoria
        else:
            action = self.get_action() # acción a partir del valor de Q (elección de la acción con mejor Q)
        
        next_state, reward, done, info = self.env.step(action)
        self.next_state =  self.flatten_state(next_state) #Recordar preprocesar los estados
        self.total_reward += reward

        return action, reward, done, info

    def train(self, gamma=0.99, n_episodes=1000, dnn_update_frequency=7, dnn_sync_frequency=30, 
                    min_epsilon = 0.01, epsilon=0.1, eps_decay=0.99, nblock=10, verbose=0):
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.gamma = gamma

        episode = 0
        for episode in range(1, n_episodes+1):
            t_start = time.time()
            # Inicialización del entorno
            next_state = self.env.reset()
            self.next_state = self.flatten_state(next_state)
            self.total_reward = 0
            step = 1
            done = False
            self.loss_outsider = 0

            while done == False:
                self.state = deepcopy(self.next_state)
                action, reward, done, info = self.step()

                if (step % dnn_update_frequency) == 0:
                  self.update_main_network(action, reward, done)

                if (step % dnn_sync_frequency) == 0:
                    for i in range(self.n_assets):
                        self.target_network[i].load_state_dict(self.main_network[i].state_dict())
                step += 1
            # Update del último estado (done)
            self.update_main_network(action, reward, done)
            #print(f"Numero de outsiders loss {self.loss_outsider}")

        #####       ALMACENAR EL SEGUIMIENTO DE INFORMACIÓN  LOSS, REWARD, ETC    #######

            self.l_epsilons.append(self.epsilon) # Añado a la lista de epsilons
            self.train_rewards.append(self.total_reward) # Añado a la lista de recompensas
            self.train_losses.append(np.mean(self.update_loss)) # Añado a la lista de pérdidas
            self.mean_train_rewards.append(np.mean(self.train_rewards[-nblock:]))
            self.update_loss = []

            # Actualizo epsilón
            self.epsilon = max(min_epsilon, self.epsilon * self.eps_decay)
            if verbose == 1:
                t_train = time.time()
                print ("Episode {}\tTotal Reward {:.2f}\t Time {:.2f} minutes".format(episode, self.total_reward, (time.time() - t_start)/60))

    def save_stats(self, fname):
        train_pickle = {'train_rewards': self.train_rewards,
                        'mean_train_rewards': self.mean_train_rewards,
                        'loss': self.train_losses,
                        'epsilon':self.l_epsilons}

        pickle.dump(train_pickle,open(fname, 'wb'))

    def save_model(self, fname):
        for i in range(self.n_assets):
            torch.save(self.main_network[i].state_dict(), fname+"_"+str(i)+".pth")

    def load_model(self, fname):
        for i in range(self.n_assets):
            self.main_network[i].load_state_dict(torch.load(fname+"_"+str(i)+".pth"))
            self.main_network[i].eval()

    def test(self):
        # Inicialización del entorno
        info_step = []
        next_state = self.env.reset()
        self.next_state = self.flatten_state(next_state)
        self.total_reward = 0
        eps_reward = 0
        done = False

        while not done:     
            self.state = self.next_state.copy()
            _, reward, done, info = self.step('test')
            eps_reward += reward
            info_step.append(deepcopy(info))

        return eps_reward, info_step