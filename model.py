import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=44, fcAct_units1=512, fcAct_units2=256, fcAct_units3=128, fcAct_units4=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fcAct_units1)
        #self.BatchNorm1 = nn.BatchNorm2d(fcAct_units1)
        self.fc2 = nn.Linear(fcAct_units1, fcAct_units2)
        #self.BatchNorm2 = nn.BatchNorm2d(fcAct_units2)
        self.fc3 = nn.Linear(fcAct_units2, fcAct_units3)
        #self.BatchNorm3 = nn.BatchNorm2d(fcAct_units3)
        self.fc4 = nn.Linear(fcAct_units3, fcAct_units4)
        self.fc5 = nn.Linear(fcAct_units4, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state):
        """Build an actor (policy) network that maps states -> actions."""
        out =  F.leaky_relu(self.fc1(state))
        #out = self.BatchNorm1(out)
        out =  F.leaky_relu(self.fc2(out))
        #out = self.BatchNorm2(out)
        out =  F.leaky_relu(self.fc3(out))
        #out = self.BatchNorm3(out)
        out =  F.leaky_relu(self.fc4(out))
        return torch.tanh(self.fc5(out))
            

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed=44, fc1_units=512, fc2_units=256, fc3_units=128, fc4_units=64):
        super(Critic, self).__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        self.seed = torch.manual_seed(seed)
        self.fc_1 = nn.Linear(state_size, fc1_units)
        #self.BatchNorm_1 = nn.BatchNorm2d(fc1_units)
        self.fc_2 = nn.Linear(fc1_units + action_size, fc2_units)
        #self.BatchNorm_2 = nn.BatchNorm2d(fc2_units)
        self.fc_3 = nn.Linear(fc2_units, fc3_units)
        #self.BatchNorm_3 = nn.BatchNorm2d(fc3_units)
        self.fc_4 = nn.Linear(fc3_units, fc4_units)
        self.fc_5 = nn.Linear(fc4_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(*hidden_init(self.fc_3))
        self.fc_4.weight.data.uniform_(*hidden_init(self.fc_4))
        self.fc_5.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs =  F.leaky_relu(self.fc_1(state))
        out = torch.cat((xs, action), dim=1)
        #out = self.BatchNorm_1(out)
        out =  F.leaky_relu(self.fc_2(out))
        #out = self.BatchNorm_2(out)
        out =  F.leaky_relu(self.fc_3(out))
        #out = self.BatchNorm_3(out)
        out =  F.leaky_relu(self.fc_4(out))
        return  self.fc_5(out)