# Title: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

# Author: Schrittwieser et al. (2019; DeepMind)

#### General Content:
Next stage in the alphago search-based PI algo evolution. Get rid of transition dynamic hardcoding and learn by doing MCTS in latent/abstract space using a learned model. Learning the model is done by predicting key relevant quantities (reward, value/n-step, policy) for planning. Extend the general setting to single-agent games and non-zero intermediate rewards.

#### Key take-away for own work:
* Planning becomes a lot easier after reducing the state space/having learned meaningful representations of the world. -> Move to abstract MDP world and work in low-dim space

#### Keypoints:

* Problem of current MB work: cant scale to visually rich domains such as ATARI
* Problem of current MF work: far from SOTA in domains requiring lookahead

* Key idea: Predict aspects of future that are directly relevant to planning.
    * trafo observation into hidden state - update iteratively using recurrent process of hypothetical actions
    * At each step predict policy, value function and immediate reward
    * Train end-to-end. No constraint for hidden state to capture everything to reconstruct orignal observation - reduce memory footprint
    * "agent can invent, internally, the rules or dynamics that lead to most accurate planning"

* Problem of compounding error based on tripartite separation: Representation, Model, Planning
    * Planning at pixel-level is not computationally tractable - but: majority of model capacity is focused on potentially irrelevant detail
    * Alternative approach: End-to-end prediction of the value function. Construct abstract MDP in which planning is (approx) equivalent to real env. Have to ensure value equivalence.
    * Value equivalence: starting from the same real state, the cumulative reward of a trajectory through the abstract MDP matches the cum reward of a trajectory in the real world
    * Related lit: Predictron, TreeQN, Value Iteration Networks, Value Prediction Networks

* Planning in MuZero:
    * 3 Modules: Representation/state encoding h, dynamics g, prediction f
    * Dynamics: Recurrent process that computes at each step k, an immediate reward and an internal state - mirrors structure of MDP - but: no semantics of the world attached. Here assumed to be deterministic.
    * Prediction: Computed based on internal state - value and policy functions
    * Representation: Encodes past obs o_1,...,o_t
        * Conv + ResNet architecture as in AlphaZero (but only 16 blocks)

* Acting in MuZero:
    * Apply any MDP planning algo to internal rewards/state space induced by dynamics function - e.g. MCTS with UCB max
    * Perform MCTS at each time step using planning. Sample next action from search policy which is proportional to the visit count for each action from root node
    * Generate transitions and store all of the in buffer at the end of the episode

* Training in MuZero:
    * Sample trajectory from buffer. Use input sequence to generate encoded initial state. Unroll model for k-steps (Mostly 5). Compare to underlying truth and construct error to propagate gradients
    * 4 part loss: Prediction immediate reward, value, policy, L2 regularizer

* Results & Ablations
    * Use more simulations for board games (800=0.1 sec search) than for ATARI (50), 256 channels
    * Less compute per node in search tree - amortization in search tree/caching
    * Sample efficiency: reanalyzes of old trajectories - provide fresh targets! - A lot better performance on 200 Mio ATARI frames
    * Scalability of planning: Different thinking times - MuZero matched performance of full model even with long thinking times!
    * Scalability across ATARI: 100 simulations in search leads to plateau. But only one simulation/just policy network beats SOTA
    * Comparison with only model-free objective function (DQN style) and without any search. Same performance but slower learning. Conjecture: search-based policy improvement step provides a stringer learning signal than high bias, high variance targets of Q-learning.

* More details from appendix:
    * Comparison with AlphaZero: actions available not masked (only at root) - network learns which moves are very rare and never occur! Termination does not receive further treatment - just treat as terminal state
    * Scepasvari - convergence of MCTS to optimal policy and minimax value function in zero-sum games
    * Search - 3 stages: Selection, Expansion, Backup
        - Tracked stats: Count N, Mean value Q, Policy P, Reward R, State transition S
        - Selection: Max UCB - same constants as in AlphaGo, keep large lookup table and lookup next state state and reward in table!
        - Expansion: Final t compute reward and state based on dynamics function and store. Compute policy and value & add final node. Search algo makes at most one call to dynamics function and prediction per simulation.
        - Backup: Update of statistics along the trajectory - provide generalization and change the pUCR rule for case with non-zero sum reward games - keep a max min track
    * Training use prioritization for visitation counts

#### Questions/Critical Observations:

* Potential gains for making dynamics probabilistic - Bayesian notion?!
