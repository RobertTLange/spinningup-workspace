import gym
import time
import numpy as np
import pandas as pd
import gridworld

import torch
import torch.autograd as autograd

from models.dqn import init_dqn
from models.drqn import MLP_DRQN
from utils.dqn_helpers import command_line_dqn
from utils.drqn_helpers import compute_episode_loss
from utils.general_helpers import ReplayBuffer, update_target, epsilon_by_episode, get_logging_stats

def run_drqn_learning(args):
    log_template = "Step {:>2} | T {:.1f} | Median R {:.1f} | Mean R {:.1f} | Median S {:.1f} | Mean S {:.1f}"

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
    EPS_START, EPS_STOP, EPS_DECAY = args.EPS_START, args.EPS_STOP, args.EPS_DECAY
    GAMMA, L_RATE = args.GAMMA, args.L_RATE

    INPUT_DIM = args.INPUT_DIM
    HIDDEN_SIZE = args.HIDDEN_SIZE
    NUM_ACTIONS = args.NUM_ACTIONS

    NUM_UPDATES = args.NUM_UPDATES
    NUM_ROLLOUTS = args.NUM_ROLLOUTS
    MAX_STEPS = 200
    ROLLOUT_EVERY = args.ROLLOUT_EVERY
    UPDATE_EVERY = args.UPDATE_EVERY
    PRINT_EVERY = args.PRINT_EVERY
    VERBOSE = args.VERBOSE
    AGENT = args.AGENT

    if AGENT == "DOUBLE": TRAIN_DOUBLE = True
    else: TRAIN_DOUBLE = False

    agents, optimizer = init_dqn(MLP_DRQN, L_RATE, USE_CUDA,
                                 INPUT_DIM, HIDDEN_SIZE, NUM_ACTIONS)

    replay_buffer = ReplayBuffer(capacity=5000)

    reward_stats = pd.DataFrame(columns=["opt_counter", "rew_mean", "rew_sd",
                                         "rew_median", "rew_10th_p", "rew_90th_p"])

    step_stats = pd.DataFrame(columns=["opt_counter", "steps_mean", "steps_sd",
                                       "steps_median", "steps_10th_p", "steps_90th_p"])

    # Initialize optimization update counter and environment
    opt_counter = 0
    ep_id = 0
    env = gym.make("dense-v0")

    # RUN TRAINING LOOP OVER EPISODES
    while opt_counter < NUM_UPDATES:
        epsilon = epsilon_by_episode(opt_counter + 1, EPS_START,
                                     EPS_STOP, EPS_DECAY)

        obs = env.reset()

        steps = 0
        # Initialize Hidden State
        agents["current"].init_hidden()
        agents["target"].init_hidden()

        while steps < MAX_STEPS:
            action = agents["current"].act(obs.flatten(), epsilon)
            next_obs, rew, done, _  = env.step(action)
            steps += 1
            # Push transition to ER Buffer
            replay_buffer.push(ep_id, steps, obs, action,
                               rew, next_obs, done)

            # Break episode if terminated & perform update step
            if done: break
            else: obs = next_obs

        ep_id += 1
        if replay_buffer.num_episodes() > TRAIN_BATCH_SIZE:
            opt_counter += 1
            loss = compute_episode_loss(agents, optimizer,
                                        replay_buffer,
                                        TRAIN_BATCH_SIZE, GAMMA,
                                        Variable, TRAIN_DOUBLE)

        # On-Policy Rollout for Performance evaluation
        if (opt_counter+1) % ROLLOUT_EVERY == 0:
            r_stats, s_stats = get_logging_stats(opt_counter, agents,
                                                 GAMMA, NUM_ROLLOUTS,
                                                 MAX_STEPS, AGENT)
            reward_stats = pd.concat([reward_stats, r_stats], axis=0)
            step_stats = pd.concat([step_stats, s_stats], axis=0)

        if (opt_counter+1) % UPDATE_EVERY == 0:
            update_target(agents["current"], agents["target"])

        if VERBOSE and (opt_counter+1) % PRINT_EVERY == 0:
            stop = time.time()
            print(log_template.format(opt_counter+1, stop-start,
                                      r_stats.loc[0, "rew_median"],
                                      r_stats.loc[0, "rew_mean"],
                                      s_stats.loc[0, "steps_median"],
                                      s_stats.loc[0, "steps_mean"]))
            start = time.time()

    # Finally save all results!
    torch.save(agents["current"].state_dict(), "agents/" + str(NUM_UPDATES) + "_" + AGENT_FNAME)
    # Save the logging dataframe
    df_to_save = pd.concat([reward_stats, step_stats], axis=1)
    df_to_save = df_to_save.loc[:,~df_to_save.columns.duplicated()]
    df_to_save = df_to_save.reset_index()
    df_to_save.to_csv("results/"  + args.AGENT + "_" + STATS_FNAME)
    return df_to_save


if __name__ == "__main__":
    args = command_line_dqn()

    if args.RUN_TIMES == 1:
        print("START RUNNING {} AGENT LEARNING FOR 1 TIME".format(args.AGENT))
        run_drqn_learning(args)
    else:
        run_multiple_times(args, run_drqn_learning)
