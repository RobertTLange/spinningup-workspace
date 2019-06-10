import argparse
import torch
import numpy as np

import gym
import gridworld
import torch.autograd as autograd

# Set device config variables
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


def command_line_vpg():
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


def train_one_batch(agent, optimizers, TRAIN_BATCH_SIZE):
    # Initialize placeholder lists
    batch_obs, batch_acts = [], []
    batch_advantage, batch_log_probs, batch_values = [], [], []
    batch_rets, batch_lens = [], []
    ep_rews = []

    # reset episode-specific variables
    env = gym.make("dense-v0")
    obs = env.reset()
    done = False
    steps = 0

    while True:
        obs_T = Variable(torch.FloatTensor(obs.flatten()).unsqueeze(0),
                         volatile=True)
        policy_v = agent["policy"].forward(obs_T)
        value = agent["value"].forward(obs_T)

        action = policy_v.sample()
        next_obs, rew, done, _  = env.step(action.cpu().numpy())
        steps += 1

        batch_obs.append(obs.copy())
        batch_acts.append(action)
        ep_rews.append(rew)

        log_prob = policy_v.log_prob(action)
        batch_log_probs.append(log_prob)
        batch_values.append(value)

        # Go to next episode if current one terminated or update obs
        if done:
            batch_rets.append(sum(ep_rews))
            batch_lens.append(len(ep_rews))

            # Different formulation: Full Reward: [ep_ret] * ep_len
            batch_rew_to_go = reward_to_go(ep_rews)
            batch_advantage += list(batch_rew_to_go - batch_values)
            # reset episode-specific variables
            obs, done, ep_rews, steps, batch_values = env.reset(), False, [], 0, []
            # end experience loop if enough data gathered
            if len(batch_obs) > TRAIN_BATCH_SIZE:
                break
        else:
            obs = next_obs

    batch_log_probs = torch.cat(batch_log_probs)
    batch_advantage = torch.cat(batch_advantage)

    actor_loss = -(batch_log_probs * batch_advantage.detach()).mean()
    critic_loss = batch_advantage.pow(2).mean()

    # Perform Policy Gradient Step
    optimizers["policy"].zero_grad()
    actor_loss.backward()
    optimizers["policy"].step()

    # Update value network estimate
    optimizers["value"].zero_grad()
    critic_loss.backward()
    optimizers["value"].step()
    return actor_loss.item(), critic_loss.item() , batch_rets, batch_lens


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs
