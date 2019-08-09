# Title: Towards Characterizing Divergence in Deep Q Learning

# Author: Achiam et al (2019) (openAI)

#### General Content:
Authors analyze the Q-Learning updates with a linear approximation and their contractive behavior. Based on that they are able to split the update into the different contributions of the deadly triad. They argue that most of current techniques tackle the bootstrapping and off-policy pitfalls. Their proposed algorithm (PreQN), on the other hand, targets within-batch generalization error by preconditioning with the matrix of inner products of Q gradients. It achieves SOTA performance on continuous control tasks without any use of target nets.

#### Key take-away for own work:
Revelation: MSBE gradient update reduces to tabular Q-Learning update when thinking of parameters as entries in the table. Authors put a stronger emphasis on the function approximation aspect and overgeneralization.

#### Keypoints:

* DDPG = DQN for continuous space! Actor allows to take max since greedy max is impractical.

* Motivation: Many more or less heuristic solutions to the deadly triad problem but no real characterization of the divergence. Here:
    * Derive leading order approximation to the DQL update operator - Disentanglement of contributions of the triad!
    * Relationship between Neural Tangent Kernel and the Q approximator
    * Describe conditions under which operator is a contraction under the sup norm

* Linearization of update allows to decompose into 3 parts:
    1. Function approx K with off diag elements Characterizing generalization
    2. Diagonal matrix D with rho, the distribution from the replay buffer
    3. The bootstrap term

* Authors then successively introduce the terms to the update to gain intuition about the convergence behavior/requirements necessary for convergence.

* Intuition results:
    1. The close the update operator is to a simple Q-Learning update the more stable the learning.
    2. Off-policy learning update is a contraction if all state-action pairs are visited and learning rate is in specific range. Missing data is a key problem - relation to van Hasselt observation of divergence happening early on!
    3. Linear fct. approx (linear in the features which might arise from nonlinear mapping!) case - can show that under certain conditions on K update operator is a contraction - small off-diag elements compared to on-diag! CANT OVERGENERALIZE! Result also holds for a sequence of convergent Bellman operators

* Failure Mode Insights (may not be orthogonal - can produce loops!):
    1. Linearization breaks - large learning rate, second-order terms large
    2. Overshooting - Leaning rate in range where operator sometimes expansion
    3. Overgeneralization - K large off-diag elements
    4. Extrapolation Error - Overgeneralization

* PreQN algorithm: Make updates non-expansive!
    * Computationally expensive!
    * Precondition the TD-errors in minibatch gradient updates, using inverse of matrix of inner products of Q gradients.
    * Show that under restricted conditions the update is equivalent to a natural gradient Q update
    * Calculate preconditioning matrix via LS on minibatch!

* Experiment insights
    * Use ratio of K entries as a metric and compare different inits and activations of networks - evaluate for a set of 1000 rails-random policy state action pairs
    * sin nets perform best without target net and preconditioning on continous control tasks
    * Future - Insights into architecture design!!

#### Questions/Critical Observations:

* Write down/summarize the derivation/1st order taylor!
* Read up on Neural Tangent Kernel
* How does this relate to the Lottery Ticket Hypothesis?!
