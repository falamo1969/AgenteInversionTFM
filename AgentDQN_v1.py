from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
import torch.optim as optim
import random


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
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size[0]
        self.n_assets = action_size[1]
        self.gamma = gamma
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = [QNetwork(self.state_size, self.action_size).to(self.device) for _ in range(self.n_assets)]
        self.optimizer = [optim.Adam(self.q_network[i].parameters(), lr = self.lr) for i in range(self.n_assets)]
        
    def get_action(self, state, epsilon=0.1):
        action = []
        if random.uniform(0, 1) < epsilon:
            action = [random.randint(0, self.action_size -1) for _ in range(self.n_assets)]
        else:
            for i in range(self.n_assets):
                with torch.no_grad():
                    state = torch.FloatTensor(state).to(self.device)
                    q_values = self.q_network[i](state)
                    action.append(torch.argmax(q_values))
        return action
    
    def update_q_network(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        for i in range(self.n_assets):
            q_values = self.q_network[i](state)
            q_value = q_values[action[i]]
            
            with torch.no_grad():
                next_q_values = self.q_network[i](next_state)
                next_q_value = torch.max(next_q_values)
                target_q_value = reward + self.gamma * next_q_value * (1 - done)
                
            loss = nn.functional.smooth_l1_loss(q_value, target_q_value)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()