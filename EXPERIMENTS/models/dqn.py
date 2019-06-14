import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from utils.general_helpers import init_weights

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def init_dqn(model, L_RATE, USE_CUDA,
             INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS,
             load_checkpoint_path=None):
    """
    Out: Model (or dictionay) as well as optimizer
    """
    agents = {"current": model(INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS),
              "target": model(INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS)}

    if USE_CUDA:
        agents["current"] = agents["current"].cuda()
        agents["target"] = agents["target"].cuda()

    if load_checkpoint_path is not None:
        checkpoint = torch.load(load_checkpoint_path,
                                map_location='cpu')
        agents["current"].load_state_dict(checkpoint)
        agents["target"].load_state_dict(checkpoint)


    # Initialize optimizer object - single agent
    optimizers = optim.Adam(params=agents["current"].parameters(), lr=L_RATE)
    return agents, optimizers


class MLP_DQN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
        super(MLP_DQN, self).__init__()

        self.action_space_size = NUM_ACTIONS

        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_space_size)
        )

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space_size)
        return action


class MLP_DuelingDQN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
        super(MLP_DuelingDQN, self).__init__()
        # Implements a Dueling DQN agent based on MLP
        self.action_space_size = NUM_ACTIONS

        self.feature = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_SIZE),
            nn.ReLU()
        )
        self.feature.apply(init_weights)

        self.advantage = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_space_size)
        )
        self.advantage.apply(init_weights)

        self.value = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        self.value.apply(init_weights)

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0),
                             volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space_size)
        return action
