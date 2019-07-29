# Title:  A Distributional Perspective on RL

# Author: Bellemare et al (2017) (DeepMind)

#### General Content:
Consider the full value distribution and not only the standard mean formulation of the RL problem. Previously only done in order to implement risk aversion/sensitivity. Provide theoretical results for the distributional instability in the control case of the classical operator - no exponential convergence guarantee! Highlights importance of learning from a nonstationary policy! Instead introduce the distributional Bellman operator to learn an approximate value distribution. Evaluation on ATARI - beats current SOTAs.


#### Key take-away for own work:
Fight against habits. Very novel idea that goes beyond and above the general RL doctrine! Think of bootstrapping as a form of complementing a limited reward signal in order to obtain something that is similar to a target in supervised learning.

#### Keypoints:

* Show that Bellman op over value distribution is contraction in max form of Wasserstein metric. Not in KL - important!
* Benefits of distributional Bellman op:
    1. Preserves multimodality in value distributions - believe that this leads to more stable learning.
    2. Approximation of full distribution mitigates effects of learning from nonstat policy!
    3. Reduced chattering coming from fct approx and instability
    4. State aliasing - coupling representation learning with policy learning
    5. Richer set of prediction - accuracy coupled with performance
    6. Inductive bias via bounding of reward
    7. Easy computation of KL between two categorical distributions

* 3 sources of randomness: Reward function, transitions, Next state value distribution - assumption that these sources are independent

* Practical approximation of the full distribution
    * Divide distribution into bins/atoms (equally spaced) - categorical distribution
    * Get probs of bins via softmax from output values of parametrized network
    * Bellman updated distribution and pre-updated distribution almost always have disjoint support - need to project back! Reduces the sample update to multiclass classification
    * Bounding of value as a form of regularization
    * Categorical distribution updates verz similar to SARSA!
    * Optimization of KL loss between Bellman projected categorical distribution (with flexible - optimizable parameters) and original one (with fixed parameters)! Do not want to change too much in distributional space vs MSBE in mean value!

* Select 51 atoms for ATARI based on hyperparam optimisation

* Intuition for why it might work: Distributional updates separate low-value events from high-value events rather than averaging into an unrealisable expectation!

* Values are often times seen to be Gaussian in ATARI games! Authors argue about this being due to discretization of diffusion process induced by discount factor

* Additional computational cost are overcome by extreme faster speed!

#### Questions/Critical Observations:
* Go through math and derive Q-Learning contraction mapping proof!
* Ideas for how to use this formulation of distribution/uncertainy for exploration!
* Is there a principled way for choosing the number of atoms
* ATARI version fail to execute actions in 25% of cases!
* Read up on conservative policy iteration - Kakade Langford 2002
