# Title: Reinforcement Learning with Unsupervised Auxiliary Tasks

# Author: Jaderberg et al. (2016, DeepMind)

#### General Content:
Introduce a set of auxiliary tasks that can provide main task-relevant gradient signals. Sharing of relevant representations allows for learning in even sparse reward environments. Core idea of empowerment: prediction and control of features of environment. Introduce 3 general aux tasks: value function replay, reward prediction, pixel control which they incorporate into training an A3C agent characterized by CNN embedding and LSTM layer.

#### Key take-away for own work:
Very first ideas in the direction of self-supervised learning in RL. Environment experiences have a lot of info in them. Question of how to define/extract task union/dependencies! Empowerment via prediction.

#### Keypoints:
* Env produces wide possible training signals. Define pseudo-rewards that overlap with task.
    * Especially helpful when the reward signals are sparse
    * Need for representation sharing between these pseudo-tasks and the actual optimized behavior
    * Most general objectives: Predict and control sensorimotor stream - flexible control of future
    * Important: There is no additional supervision signal provided!

* Base Agent: A3C LSTM with shared torso and two output heads (value & policy) - On-policy

* Here 3 main auxiliary tasks - storage of relevant transitions in small buffer:

1. Value Function Replay:
    - Additional training of the critic to promote faster value iteration!
    - Often times done in AC methods since actor critically depends on correct signal in loss!
    - Resampling of **recent** historical sequences - form of additional prioritization
    - Also: Randomly vary temporal position of truncation window - different k-step returns/amounts of bootstrapping

2. Aux Reward - Reward Prediction:
    - Given recent three 3 frames predict the next reward for unobserved timestep!
    - Uses CNN embedding layer and stack three channels to then feed into simple FFW architecture
    - Need to provide balanced amount of rewards/non-rewards: Remove/introduce bias
    - Skewed sampling by splitting buffer into two parts!
    - Reward predictor is not used for anything except for providing an additional gradient flowing in
    - Predict if reward positive/negative or zero: Multiclass cross-entropy loss

3. Aux Control - Pixel Control:
    - Train aux policies to max pixel change in intensity over different regions of the image
    - Use CNN & LSTM from agent together with a separate deconv network
    - Learns to "control" the environment
    - Define a RL objective - TD MSBE error: Need to do in off-policy fashion
    - Predict best action to change pixels of nxn non-overlapping grids of the input frame
    - Head of aux network is of size N_act x n x n - dueling architecture!

* Related work:
    - Temporal abstractions: here are no pseudo-reward specific policies learned!
    - Horde: identify value functions for multitude of distinct pseudo-rewards - not used for representation learning
    - UVFA: factored representation of continuous set of optimal value functions. Combines features of state with embedding of pseudo-reward function
    - SR: factors continuous set of expected value fcts for a fixed policy
    - Learning models of the environment: Fails in complex environments

* Total loss aggregates the different components & in practice the gradients are computed from different data (on/off-policy, more/less recent)

* Experiments CNN-LSTM a3C with 20 step returns
    * Better final performance and faster for ATARI and different Labyrinth tasks
    * Input reconstruction loss hurts. Hypothesis - focus on irrelevant features for the future!
    * Ablation study - pixel control gives biggest performance boost!

* Interesting approach of sampling the learning rate/entropy cost from a log-uniform! Used RMSprop optimizer

#### Questions/Critical Observations:

* Weird feature control - max activation of neurons in network itself - where does the signal come from and isnt that too non-stationary with learning?
* How to weight different loss components in practice - multi-task problem see focal loss ideas
