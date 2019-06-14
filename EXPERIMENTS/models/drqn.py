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


class MLP_DRQN(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS):
        super(MLP_DRQN, self).__init__()
        self.action_space_size = NUM_ACTIONS
        self.hidden_size = HIDDEN_SIZE
        self.init_hidden()

        self.linear_in = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_SIZE),
                                       nn.ReLU())
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_out = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)

    def forward(self, x):
        out1 = self.linear_in(x)
        out2, self.hidden_state = self.gru(out1.unsqueeze(0),
                                           self.hidden_state)
        out3 = self.linear_out(out2.squeeze(0))
        return out3

    def init_hidden(self, batch_size=1):
        self.hidden_state = torch.zeros(1, batch_size, self.hidden_size)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.action_space_size)
        return action
