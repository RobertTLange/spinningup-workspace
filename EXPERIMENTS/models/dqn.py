import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def init_dqn(model, L_RATE, USE_CUDA, NUM_ACTIONS=4,
               load_checkpoint_path=None):
    """
    Out: Model (or dictionay) as well as optimizer
    """
    agents = {"current": model(NUM_ACTIONS),
              "target": model(NUM_ACTIONS)}

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
    def __init__(self, NUM_ACTIONS=4):
        super(MLP_DQN, self).__init__()

        num_inputs = 10*20*6
        self.action_space_size = NUM_ACTIONS

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_size)
        )

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
    def __init__(self, NUM_ACTIONS=4):
        super(MLP_DuelingDQN, self).__init__()
        # Implements a Dueling DQN agent based on MLP
        num_inputs = 10*20*6
        self.action_space_size = NUM_ACTIONS

        hidden_units = 128
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, hidden_units),
            nn.ReLU()
        )
        self.feature.apply(init_weights)

        self.advantage = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, self.action_space_size)
        )
        self.advantage.apply(init_weights)

        self.value = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
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


class CNN_DDQN(nn.Module):
    def __init__(self, params):
        super(CNN_DDQN, self).__init__()

        self.input_shape = params.observation_shape
        self.num_actions = params.action_space_size

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=3),
            nn.ReLU(),
            )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0),
                             volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
