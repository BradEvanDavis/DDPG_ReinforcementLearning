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
       ''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        actor_target(state) -> action
        critic_target(state, action) -> q-value
        '''
        states, actions, rewards, next_states, dones = experiences
        
        #update critic -------------------------------------------

        #get predicted next state actions and Q values from targets
        actions_next = self.actor_target(next_states)
        Q_trgts_next = self.critic_target(next_states)
        #compute Q targets for current state (y_i)
        Q_trgts = rewards + (gamma * Q_trgts_next * (1- dones))
        #critic loss
        Q_expected = self.critic_loca(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_trgts)
        #minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor--------------------------------------------

        #actor loss 
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        #minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update trgt network-------------------------------------
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

