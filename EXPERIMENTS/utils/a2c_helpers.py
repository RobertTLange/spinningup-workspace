import argparse
import torch
import numpy as np

import gym
import gridworld
import torch.autograd as autograd

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def command_line_a2c():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_upd', '--ROLLOUT_EVERY', action="store",
                        default=20, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-n_runs', '--RUN_TIMES', action="store", default=1, type=int,
                        help='# Times to run agent learning')
    parser.add_argument('-n_upd', '--NUM_UPDATES', action="store",
                        default=5000, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-n_roll', '--NUM_ROLLOUTS', action="store",
                        default=5, type=int,
                        help='# rollouts for tracking learning progrees')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=200, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-num_steps', '--NUM_STEPS', action="store",
                        default=5, type=int,
                        help='Max # of steps before A2C update')
    parser.add_argument('-v', '--VERBOSE', action="store_true", default=False,
                        help='Get training progress printed out')
    parser.add_argument('-print', '--PRINT_EVERY', action="store",
                        default=20, type=int,
                        help='#Episodes after which to print.')

    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')


    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=32, type=int, help='# images in training batch')
    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    return parser.parse_args()


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train_a2c_steps(env, obs, agent, optimizer, NUM_STEPS, GAMMA):
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(NUM_STEPS):
        obs_T = Variable(torch.FloatTensor(obs.flatten()).unsqueeze(0),
                         volatile=True)
        policy_v, value = agent(obs_T)

        action = policy_v.sample()
        next_obs, rew, done, _ = env.step(action.cpu().numpy())

        log_prob = policy_v.log_prob(action)
        entropy += policy_v.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(rew)
        masks.append(1 - done)

        obs = next_obs

        if done:
            obs = env.reset()
            break

    next_obs = Variable(torch.FloatTensor(next_obs.flatten()).unsqueeze(0),
                        volatile=True)
    _, next_value = agent(next_obs)
    returns = compute_returns(next_value, rewards, masks, GAMMA)

    log_probs, returns, values = torch.cat(log_probs),  torch.cat(returns).detach(), torch.cat(values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return actor_loss, critic_loss, env, obs
