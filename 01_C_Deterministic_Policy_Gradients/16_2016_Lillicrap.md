# Title: Continuous Control with Deep Reinforcement Learning

# Author: Lillicrap et al (2016) (DeepMind)

#### General Content:
Authors scale up original work by Silver et al (2014) on DPG with the help of DQN techniques (fct approximator, buffer target net, batch norm, etc.). Previously naive AC methods with approximators were instable! Key contribution: Simplicity! Standard AC architecture + learning algo. Demonstrate scalability in ATARI, MuJoCo as well as standard continuous control environments.

#### Key take-away for own work:


#### Keypoints:

* Why not simply discretize continous action space? - Runs into curse of dimensionality Also throws away info about structure of the action space which can be exploited/used in fct approx to generalize from.

* Deterministic policy allows to get rid of inner expectation over action space in Bellman equation. Allows for off-policy learning from all transitions.

* DPG algortihm:
    * Critic: Updates from simple Bellman equation with Q-learning
    * Actor: Update following chain rule to expected return
    * No convergence guarantee with non-linear fct approximators. NFQCA - same updating rule with nets and minibatch updates

* Here: DQN inspired adaptations to DPG which allow to learn in large state and action spaces
    * Experience replay buffer for sample efficiency - storing off-policy transitions
    * Soft target updates: Dont update target Q/policy weights only every C iters but do slow Polyak averaging. Authors note that this moves unstable action-value function learning closer to case of supervised learning. May slow down learning but stabilizes at the same time!
    * Batch norm: Different units of measurements for different dimensions of state vector. In DL used to minimize covariance shift during training as well as making gradient closer to natural gradient due to approx identity matrix - curvature. Used at input as well as all individual layers
    * Exploration: Add noise to policy. Usually done with Ohnstein-Uhlenbeck process since it provides temporally-corr exploration - good for physical control problems which requrie inertia.

* Techniques described for scaling DPG also apply to stochastic policies when using the reparametrization trick!

* Guided policy search - 3 phases:
    1. Use full state obs to create locally-linear approx to dynamics around nominal trajectories
    2. Use optimal control to find locally-linear optimal policy along trajectories
    3. Use supervised learning to train complex, non-linear policy to reproduce state-to-action mapping of optimized trajectories

* PILCO: GP to learn non-param probabilistic model of dynamics
    * Use model to calculate analytical PG
    * Problem: High computational demand for inversion of Cov/Kernel makes it impractical for high-dim problems


#### Questions/Critical Observations:

* I have heard that DDPG is very hard to tune in terms of hyperparams. Check!
