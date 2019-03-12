import numpy as np
import random
import copy
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from collections import deque
import torch.optim as optim
import torch.nn as nn

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128     # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACT = 0.001           # learning rate of the actor 
LR_CRITIC = 0.001        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
LEARN_NUM = 10
LEARN_EVERY = 20
EPSILON = 0.75
EPSILON_DECAY = 0.00001

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed=44):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON

        self.actor_local = nn.DataParallel(Actor(state_size, action_size, random_seed)).cuda()
        self.actor_target = nn.DataParallel(Actor(state_size, action_size, random_seed)).cuda() 
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACT)
        
        self.critic_local = nn.DataParallel(Critic(state_size, action_size, random_seed)).cuda()
        self.critic_target = nn.DataParallel(Critic(state_size, action_size, random_seed)).cuda()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        #generate noise
        self.noise = OUNoise(action_size, random_seed)

        #replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, timestep):
        #save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        #learn
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY==0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        #reutrn action based on current policy
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, GAMMA):
        ''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        actor_target(state) -> action
        critic_target(state, action) -> q-value
        '''
        states, actions, rewards, next_states, dones = experiences
        
        #update critic -------------------------------------------

        #get predicted next state actions and Q values from targets
        actions_next = self.actor_target(next_states)
        Q_trgts_next = self.critic_target(next_states, actions_next)
        #compute Q targets for current state (y_i)
        Q_trgts = rewards + (GAMMA * Q_trgts_next * (1- dones))
        #critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_trgts)
        #minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        #update actor--------------------------------------------

        #actor loss 
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        #minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        #update trgt network-------------------------------------

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        #noise updates--------------------------------------------
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            TAU: interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

class OUNoise:
    '''Ornstein-Uhlenbeck process'''

    def __init__(self, size, seed=44, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        '''reset noise to mean'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''update internal state and return as noise'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma *np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    '''fixed buffer to store experiences'''
    def __init__(self, action_size, BUFFER_SIZE, BATCH_SIZE, seed=44):
        ''' BUFFER_SIZE = maximum size of buffer
            BATCH_SIZE = size of each training batch'''
        self.action_size = action_size
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.BATCH_SIZE = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''add experience to memory'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''randomly sample a batch of experience from memory'''
        experiences =  random.sample(self.memory, k=self.BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        '''current size of internal memory'''
        return len(self.memory)