# Title: Deep Reinforcement Learning and the Deadly Triad

# Author: van Hasselt et al (2018) (DeepMind)

#### General Content:
Authors study the problem of instability in Deep Q Learning with the help of massive compute and different ablation/hyperparameter configurations.

#### Key take-away for own work:
Well written! There is not a lot of theory possible at the moment - and this work tries to shine light onto the observation that DQNs appear to be fairly stable. Key insight: Temperature in prioritization controls amount of off-policiness.

#### Keypoints:

* Deadly triad = Non-linear function approximator, Bootstrapping, Off-Policy learning - when all come together - learning can diverge with unbounded value estimates

* Off-policy learning = learn about multiple policies in parallel
* ER buffer: Samples transitions from mixture of previous policies

* Problem of value function approximators - risk of inappropriate changes of other states than the one involved in transition - apparently not problem in on-policy case

* Observation: Deadly triad is not a binary thing but a continuum
    * multi-step returns - reduces bootstrapping impact
    * function approx. capability - influences generalisation and aliasing
    * heavier prioritization = more off-policy update

* Simplified example of two states with different features but same value - need additional parameter to not diverge
    * Off-policy updates lead to divergence
    * Convergence can be achieved with increased capacity

* Validated Hypothesis and Experimental Evidence - 3 runs all 57 ATARI games:
    1. Divergence does not happen frequently with function approx
    2. Less divergence when bootstrapping on separate networks - bootstrap target cannot be inadvertently updated immediately if separate network is used.
    3. Correction of overestimation bias by decoupling action selection from action evaluation in bootstrap target - Double Q-Learning.
    4. Smaller nets are more stable
    5. Increasing prioritization increases instability

* Tryout inverse double q learning: Still same network used for evaluation in bootstrap but separate network for action selection - does not benefit from stabilization in value estimation of target network
    * Only addresses overestimation problem but does not utilize the stability of using a separate target network

* Intuition multi-step returns reduce amount of bootstrapping: One step return// imediate bootstrap - contraction is proportional to gamma/discount factor. Multi-step updates imply gamma^multi-step contraction - we bootstrap less
    * Only holds for TD operator in tabular/linear case - how does this generalize to non-linear case

* Divergence measured by largest absolute value estimate. Observation: Many values rise quickly to unrealistic magnitudes but then shrink again to plausible magnitude

* Target Q and double Q both reduce issues of inappropriate generalization. Overestimation might exist but ranking can still be in tact which leads to the ultimate greedified policy.

* More stable update rules allow for more capacity in network architecture - leads to better performance but generally larger networks are less stable!

* Too strong prioritisation - correlates both with unreasonably high value estimates as well as reduced performance.

#### Questions/Critical Observations:
