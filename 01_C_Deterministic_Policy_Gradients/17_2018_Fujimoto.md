# Title: Addressing Function Approximation Error in Actor-Critic Methods

# Author: Fujimoto et al (2018) (McGill University)

#### General Content:
Authors extend on DDPG algorithm to combat high variance as well as overestimation build up. The main improvements include introducing a set of critics (twins) instead of simply using a target network, clipping (taking min across them) as well as updating the policy network at a slower delayed time-scale. Finally, they introduce a regularization technique (target value smoothing) which constructs the target with noise around the selected next action in the target network. Derive SOTA on MuJoCo as well as perform ablation studies showing that a combination of all proposed tricks is needed to achieve significant improvements!


#### Key take-away for own work:
Again not many novel ideas here. Just a set of stabilizing contributions as well as many experiments. Estimate of true value of state by averaging cumulate rewards over 1000 episodes following current policy, starting from states sampled from replay buffer. Average these value estimates over 10000 states. Compare these to simple value estimates for the states. Use 10000 steps of pure random exploration on MuJoCo tasks.

#### Keypoints:

* Show that overestimation bias due to noisy value estimates (original Thrun and Schwartz, 1993 paper) is also existing in AC methods. Leads to accumulation of error which is even stronger when using function approximators.

* Can't simply use Double DQN - slow changing policy in AC leads to current and target values which are too similar.
    * Proposed solution: Use a pair of independently trained critics.

* This trick reduces the bias but still large variance can lead to overestimations in local regions of state space.
    * Proposed solution: clipped double-q-learning - value estimate suffering from overest bias can be used as approximate upper bound to true value estimate.
    * Favors underestimation. But these are not propagated!
    * Take minimum over both critic value estimates. Use single actor optimized with respect to one of the critics
    * Additional benefit: Minimum operator provides higher value to states with lower variance est error - form of exploration?!

* Have to address coupling of value and policy by delaying policy update until value estimate convergence. In experiments 2 updates of critic, 1 update of actor net
    * Idea: The more time between policy updates - the more accurate the value estimate used to perform ultimate update -> higher quality policy updates -> two time-scales

* Further variance reduction by using target value smoothing with sampled noise added to continuous action
     * Relationship to SARSA On-policy update
     * Fitting of value of small area around target action. Intuition: similar actions = similar value - explicit regularization

* Problem of negative feedback loop: Inaccurate value estimates lead to poor policy updates which in turn lead to a suboptimal critic update


#### Questions/Critical Observations:
* Why not use more than two critics and have more than one upper bound, so that we can choose the tightest?
* Analyze interplay of gamma and decay/target net updates
