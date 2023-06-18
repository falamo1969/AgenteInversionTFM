from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pickle
import time
from AgentDQN_v3 import AgentDQN

class QLSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QLSTMNetwork, self).__init__()
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
    
class AgentDQNLSTM(AgentDQN):
    def __init__(self, env, state_size, action_size, lr=0.001, epsilon=0.1):
        super(AgentDQNLSTM, self).__init__(env, state_size, action_size, lr=lr, epsilon=epsilon)
    
    def _initialize(self):
        # Variables estadísticas
        self.l_epsilons = []
        self.train_rewards = []
        self.train_losses = []
        self.update_loss = []
        self.mean_train_rewards = []
        
        # Inicialización de las redes
        self.main_network = [QLSTMNetwork(self.state_size, self.action_size).to(self.device) for _ in range(self.n_assets)]
        self.target_network= deepcopy(self.main_network)
        params =  []
        for i in range(self.n_assets):
            params += list(self.main_network[i].parameters())        
        self.optimizer = optim.Adam(params=params, lr = self.lr)


