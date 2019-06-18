import argparse
import math
import random
import pandas as pd
import numpy as np
from collections import deque
import gym

import torch
import torch.autograd as autograd


def compute_episode_loss(agents, optimizer, replay_buffer,
                         TRAIN_BATCH_SIZE, GAMMA, Variable, TRAIN_DOUBLE):
    sampled_episodes = replay_buffer.sample_episodes(TRAIN_BATCH_SIZE)

    loss_total = 0
    for ep in sampled_episodes:
        ep_id, step, obs, acts, reward, next_obs, done = ep

        # Flatten the visual fields into vectors for MLP
        obs = [ob.flatten() for ob in obs]
        next_obs = [next_ob.flatten() for next_ob in next_obs]
        obs = Variable(torch.FloatTensor(np.float32(obs)))
        next_obs = Variable(torch.FloatTensor(np.float32(next_obs)))
        action = Variable(torch.LongTensor(acts))
        done = Variable(torch.FloatTensor(done))

        steps_in_ep = len(step)

        # Select either global aggregated reward if float or agent-specific if dict
        if type(reward[0]) == np.float64 or type(reward[0]) == int:
            reward = Variable(torch.FloatTensor(reward))

        agents["current"].init_hidden()
        agents["target"].init_hidden()
        for t in range(steps_in_ep):
            q_values = agents["current"](obs[t].unsqueeze(0))
            q_value = q_values.gather(1, action[t].unsqueeze(0).unsqueeze(1)).squeeze(1)

            if TRAIN_DOUBLE:
                next_q_values = agents["current"](next_obs[t].unsqueeze(0))
                next_q_state_values = agents["target"](obs[t].unsqueeze(0))
                next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            else:
                next_q_values = agents["target"](next_obs[t].unsqueeze(0))
                next_q_value = next_q_values.max(1)[0]

            expected_q_value = reward[t] + GAMMA* next_q_value * (1 - done[t])

            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            # Perform optimization step for agent
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(agents["current"].parameters(), 0.5)
            optimizer.step()
            loss_total += loss.item()

    return loss_total
