import gym
import time
import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd

from models.a2c import init_a2c
from utils.a2c_helpers import command_line_a2c, train_a2c_steps
from utils.general_helpers import get_logging_stats, run_multiple_times


def run_a2c_learning(args):
    log_template = "Step {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Actor L {:.1f} | Critic L {:.1f}"

    # Set the GPU device on which to run the agent
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.cuda.set_device(args.device_id)
        print("USING CUDA DEVICE {}".format(args.device_id))
    else:
        print("USING CPU")
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
    start = time.time()

    # Extract variables for arguments
    TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
    GAMMA, L_RATE = args.GAMMA, args.L_RATE

    NUM_UPDATES = args.NUM_UPDATES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    NUM_STEPS = args.NUM_STEPS
    MAX_STEPS = args.MAX_STEPS
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    PRINT_EVERY = args.PRINT_EVERY
    VERBOSE = args.VERBOSE

    AGENT = "A2C"

    agent, optimizer = init_a2c(L_RATE, USE_CUDA, NUM_ACTIONS=4)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    ep_id = 0

    env = gym.make("dense-v0")
    obs = env.reset()

    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < NUM_UPDATES:
        a_loss, c_loss, env, obs = train_a2c_steps(env, obs,
                                                   agent,
                                                   optimizer,
                                                   NUM_STEPS,
                                                   GAMMA)

        opt_counter += 1
        # On-Policy Rollout for Performance evaluation
        if (opt_counter+1) % ROLLOUT_EVERY == 0:
            r_stats, s_stats = get_logging_stats(opt_counter, agent,
                                                 GAMMA, NUM_ROLLOUTS,
                                                 MAX_STEPS, AGENT)
            reward_stats = pd.concat([reward_stats, r_stats], axis=0)
            step_stats = pd.concat([step_stats, s_stats], axis=0)

        if VERBOSE and (opt_counter+1) % PRINT_EVERY == 0:
            stop = time.time()
            print(log_template.format(opt_counter+1, stop-start,
                                      r_stats.loc[0, "rew_median"],
                                      r_stats.loc[0, "rew_mean"],
                                      a_loss, c_loss))
            start = time.time()

    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/VPG_{}.csv".format(NUM_UPDATES))
    return df_to_save


if __name__ == "__main__":
    args = command_line_a2c()

    if args.RUN_TIMES == 1:
        print("START RUNNING VPG AGENT LEARNING FOR 1 TIME")
        run_a2c_learning(args)
    else:
        run_multiple_times(args, run_a2c_learning)
