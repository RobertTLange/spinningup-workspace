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


def init_a2c(L_RATE, USE_CUDA, NUM_ACTIONS=4):
    """
    Out: Model (or dictionay) as well as optimizer
    """
    num_inputs = 10*20*6
    num_outputs = 4
    hidden_size = 128

    agent = ActorCritic(num_inputs, num_outputs, hidden_size)

    if USE_CUDA: agent = agent.cuda()

    optimizer = optim.Adam(params=agent.parameters(), lr=L_RATE)

    return agent, optimizer


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        init_weights(self.critic)
        init_weights(self.actor)

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value
