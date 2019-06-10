import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.general_helpers import init_weights

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def init_vpg_nets(L_RATE, USE_CUDA, NUM_ACTIONS=4):
    """
    Out: Model (or dictionay) as well as optimizer
    """
    num_inputs = 10*20*6
    num_outputs = 4
    hidden_size = 128
    agent = {"policy": vpg_policy_net(num_inputs, num_outputs, hidden_size),
             "value": vpg_value_net(num_inputs, hidden_size)}

    if USE_CUDA:
        agents["value"] = agent["value"].cuda()
        agents["policy"] = agent["policy"].cuda()

    optimizers = {"value": optim.Adam(params=agent["value"].parameters(), lr=L_RATE),
                  "policy": optim.Adam(params=agent["policy"].parameters(), lr=L_RATE)}

    return agent, optimizers


class vpg_policy_net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(vpg_policy_net, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        probs = self.actor(x)
        policy  = Categorical(probs)
        return policy


class vpg_value_net(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(vpg_value_net, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        return value
