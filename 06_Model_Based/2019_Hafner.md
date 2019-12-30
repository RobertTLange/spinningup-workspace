# Title: Dream to Control: Learning Behaviors by Latent Imagination

# Author: Hafner et al. (2019; Brain)

#### General Content:

Introduce Dreamer algo that is able to tackle long-horizon tasks from pure pixel inputs. Learns behaviors by propagating "analytical" gradients of learned state values through imagined trajectories of the world model.

#### Key take-away for own work:


#### Keypoints:

* Abstract MDP - predict forward in compact state spaces - small memory footprint
* Problem: Considering fixed short-term horizon rewards can lead to shortsighted behaviors
* Previous work: Derivative-free optimization for robust training - rather that analytic gradients offered by neural net dynamics.
* Here: AC algo that accounts for rewards beyond imagination horizon with efficient neural net dynamics
    * Predict values and actions in learned latent space
    * Values optimize Bellman consistency for imagined rewards
    * Policy maximizes values by propagating their analytic gradients back through the dynamics

* Ingredients to systems that learn in imagination: dynamics, behavior learning and env interaction.
    * Behavior learned by predicting hypothetical trajectories in compact latent space
    1. Learn latent dynamics model from dataset of past experiences to predict future rewards from action and past obs - different formulations possible
    2. Learn action + value models from predicted latent trajectories = Value - Bellman Consistency, Policy - Propagate analytical gradients
    3. Execute the learned model in real world to grow the dataset

* Latent dynamics - 3 components = Mimic non-linear Kalman Filter State Space Model
    1. Representation model: Encode observations and actions - into abstract state representation
    2. Transition model: Predict future model states without seeing corresponding observations
    3. Reward model: Predict rewards given model states

* Efficient stochastic gradient propagation of multi-step returns through neural network predictions using reparametrization
    * Imagination = defines a fully observed MDP
    * Imagined trajectories - fixed horizon H. Dreamer uses AC to learn behaviors that consider rewards beyond horizon
    * Action model estimates expected imagined rewards - outputs tanh-transformed Gaussian sufficient stats
    * Reparametrization sampling - views sampled actions as deterministically dependent on NN output - allows to backprop analytical gradients through sampling operation
    * Expectations are estimated under imagined trajectories
    * Action & value model are trained "cooperatively" as in policy iteration

* Learning objective:
    * Action model - predict actions that result in state trajectories with high value estimates for all trajectory states.
    * Value model - regress the value estimates - Bellman consistency
    * Straight-through gradients for discrete action - gradient propagation

* Different approaches to representation learning:
    1. Reward prediciton: Simply learn to predict future rewards - might not be enough with sparse rewards - use aux rewards/tasks from predicting state changes, etc.
    2. Reconstruction: Variational lower bound optimization - reconstruct obs and reward
    3. Contrative estimation: Do reconstruction in lower dimensional state space - again KL objective - efficient limitation of info extraction

* D4PG = DDPG + Distr collection + Distr Q-Leanring + Multi-Step return + Prioritized Replay

* Results
    * Value model makes Dreamer more robust to horizon and performs well even with short horizons
    * pixel reconstruction better than constrastive estimation

#### Questions/Critical Observations:
* Still unclear what analytical gradients actually mean - still batch estimates - but no longer stochastic?! Still use reparametrization trick though.
* What are straight-through gradients - Bengio et al 13
