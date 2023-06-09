from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pickle
import time

from AgentDQN_LSTM import AgentDQNLSTM


class QLSTMV2Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(QLSTMV2Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.LSTM(128, hidden_size=64, num_layers =1, batch_first =  True)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.unsqueeze(x, 0)
        x, _= self.fc2(x)
#        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x.flatten())
        return q_values

class AgentDQNLSTM_V2(AgentDQNLSTM):
    def __init__(self, env, state_size, action_size, lr=0.001, epsilon=0.1):
        super(AgentDQNLSTM_V2, self).__init__(env, state_size, action_size, lr, epsilon)

    def _initialize(self):
        # Inicialización de las redes
        self.main_network = [QLSTMV2Network(self.state_size, self.action_size).to(self.device) for _ in range(self.n_assets)]
        self.target_network= deepcopy(self.main_network)
        self.optimizer = [optim.Adam(self.main_network[i].parameters(), lr = self.lr) for i in range(self.n_assets)]

        # Variables estadísticas
        self.l_epsilons = []
        self.train_rewards = []
        self.train_losses = []
        self.update_loss = []
        self.mean_train_rewards = []

    def update_main_network(self, action, reward, done):
        state = torch.FloatTensor(self.state).to(self.device)
        if not done:
            next_state = torch.FloatTensor(self.next_state).to(self.device)

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
                
            loss = nn.functional.smooth_l1_loss(q_value, target_q_value)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            # Guardamos los valores de pérdida
            if self.device == 'cuda':
                self.update_loss.append(loss.detach().cpu().numpy())
            else:
                self.update_loss.append(loss.detach().numpy())            
