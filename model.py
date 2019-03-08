import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Moduile):
    def __init__(self, state_size, action_size, seed, action_sizes, fcAct_units1, fcAct_units2, fcAct_units3, fcAct_units4):
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
        self.seed = torch.manual(seed)
        self.fc1 = FC_layer(state_size, fcAct_units1)
        self.fc2 = FC_layer(fcAct_units1, fcAct_units2, bias=True)
        self.fc3 = FC_layer(fcAct_units2, fcAct_units3, bias=True)
        self.fc4 = FC_layer(fcAct_units3, fcAct_units4, bias=True)
        self.fc5 = FC_layer(fcAct_units4, action_sizes, bias=False, Batchnorm=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state):
        """Build an actor (policy) network that maps states -> actions."""
        out =  F.selu(self.fc1(state))
        out =  F.selu(self.fc2(out))
        out =  F.selu(self.fc3(out))
        out =  F.selu(self.fc4(out))
        out =  F.tanh(self.fc5(out))
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""
def __init__(self, state_size, action_size, seed, fc1_units, fc2_units, fc3_units, fc4_units):
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
        self.fc1 = FC_layer(state_size, fc1_units)
        self.fc2 = FC_layer(fc1_units, fc2_units, bias=True)
        self.fc3 = FC_layer(fc2_units, fc3_units, bias=True)
        self.fc4 = FC_layer(fc3_units, fc4_units, bias=True)
        self.fc5 = FC_layer(fc4_units, 1, bias=False, Batchnorm=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self,state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs =  F.selu(self.fc1(state))
        out = torch.cat((xs, action), dim=1)
        out =  F.selu(self.fc2(x))
        out =  F.selu(self.fc3(out))
        out =  F.selu(self.fc4(out))
        out =  self.fc5(out)
        return out