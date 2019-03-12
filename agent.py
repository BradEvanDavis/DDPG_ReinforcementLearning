import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim


buffer_size = int(1e5)  # replay buffer size
batch_size = 128        # minibatch size
gamma = 0.99            # discount factor
tau = 1e-3              # for soft update of target parameters
LR_act = 1e-4           # learning rate of the actor 
LR_critic = 1e-3        # learning rate of the critic
weight_decay = 0        # L2 weight decay

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)   
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_act)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_critic, weight_decay=weight_decay)
        
        #generate noise
        self.noise = OUNoise(action_size, random_seed)

        #replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def step(self, state, action, reward, next_sate, done):
        #save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        #learn
        if len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample()
            self.learn(experiences, gamma)
    
    def act(self, state, add_noise=True):
        #reutrn action based on current policy
        state = torch.from_numpy(state).half().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


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

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau: interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

class OUNoise:
    '''Ornstein-Uhlenbeck process'''

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
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
        x = self.stated
        dx = self.theta * (self.mu - x) + self.sigma *np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    '''fixed buffer to store experiences'''
    def __init__(self, action_size, buffer_size, batch_size, seed):
        ''' buffer_size = maximum size of buffer
            batch_size = size of each training batch'''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed =  random.seed(seed)

    def add(self, state, action, reward, next_State, done):
        '''add experience to memory'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''randomly sample a batch of experience from memory'''
        experiences =  random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).half().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).half().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).half().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).half().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).half().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        '''current size of internal memory'''
        return len(self.memory)