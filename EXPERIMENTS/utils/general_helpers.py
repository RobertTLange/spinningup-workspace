import torch
import torch.nn as nn
import torch.autograd as autograd

import gym
import gridworld

import math
import random
import numpy as np
import pandas as pd
from collections import deque

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def init_weights(m):
    # Xavier initialization weights in network
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)

    def push(self, ep_id, step, state, action,
             reward, next_state, done):
        self.buffer.append((ep_id, step, state, action, reward, next_state, done))

    def sample(self, batch_size):
        ep_id, step, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(eps_id, epsilon_start, epsilon_final, epsilon_decay):
    eps = (epsilon_final + (epsilon_start - epsilon_final)
           * math.exp(-1. * eps_id / epsilon_decay))
    return eps


def update_target(current_model, target_model):
    # Transfer parameters from current model to target model
    target_model.load_state_dict(current_model.state_dict())


def get_logging_stats(opt_counter, agent, GAMMA,
                      NUM_ROLLOUTS, MAX_STEPS, AGENT):
    steps = []
    rew = []

    for i in range(NUM_ROLLOUTS):
        step_temp, reward_temp, buffer = rollout_episode(agent, GAMMA,
                                                         MAX_STEPS, AGENT)
        steps.append(step_temp)
        rew.append(reward_temp)

    steps = np.array(steps)
    rew = np.array(rew)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median",
                                         "rew_10th_p", "rew_90th_p"])

    steps_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                        "steps_median",
                                        "steps_10th_p", "steps_90th_p"])

    reward_stats.loc[0] = [opt_counter, rew.mean(), rew.std(), np.median(rew),
                           np.percentile(rew, 10), np.percentile(rew, 90)]

    steps_stats.loc[0] = [opt_counter, steps.mean(), steps.std(), np.median(steps),
                         np.percentile(steps, 10), np.percentile(steps, 90)]

    return reward_stats, steps_stats


def rollout_episode(agent, GAMMA, MAX_STEPS, AGENT):
    env = gym.make("dense-v0")
    # Rollout the policy for a single episode - greedy!
    replay_buffer = ReplayBuffer(capacity=5000)

    obs = env.reset()
    episode_rew = 0
    steps = 0

    while steps < MAX_STEPS:
        if AGENT == "Vanilla-PG":
            obs = Variable(torch.FloatTensor(obs.flatten()).unsqueeze(0),
                           volatile=True)
            policy_v = agent["policy"].forward(obs)
            action = policy_v.sample()
        elif AGENT == "A2C":
            obs = Variable(torch.FloatTensor(obs.flatten()).unsqueeze(0),
                           volatile=True)
            policy_v, value = agent(obs)
            action = policy_v.sample()
        else:
            action = agent["current"].act(obs.flatten(), epsilon=0.05)
        next_obs, reward, done, _ = env.step(action)
        steps += 1

        replay_buffer.push(0, steps, obs, action,
                           reward, next_obs, done)

        obs = next_obs

        episode_rew += GAMMA**(steps - 1) * reward
        if done:
            break
    return steps, episode_rew, replay_buffer.buffer


def run_multiple_times(args, run_fct):

    df_across_runs = []
    print("START RUNNING VPG AGENT LEARNING FOR {} TIMES".format(args.RUN_TIMES))
    for t in range(args.RUN_TIMES):
        start_t = time.time()
        df_temp = run_fct(args)
        df_across_runs.append(df_temp)
        total_t = time.time() - start_t
        print("Done training {}/{} runs after {:.2f} Secs".format(t+1,
                                                                  args.RUN_TIMES,
                                                                  total_t))

    df_concat = pd.concat(df_across_runs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_means.to_csv("results/" + str(args.RUN_TIMES) + "_RUNS_VPG.csv")
    return df_means
