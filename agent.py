import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_sizze, random_seed).to(device
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_act)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_critic, weight_decay=weight_decay)
        
        self.noise = OUNoise(action_size, random_seed)

    def learn(self, exp, gamma):
       ''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state))'''
        actions_next = self.actor_target(next_states)
        Q_trgt_next = self.critic_target(next_states)

