# Title: Noisy Networks for Exploration

# Author: Meire Fortunato et al (2018) (DeepMind)

#### General Content:
Introduce special fully-connected layers that add parametrized noise to the output. The parameters of the noise can be learned and thereby allow for efficient and potentially targeted exploration. They thereby do not separate the generalisation from the exploration problem in DRL. In order to stay computationally efficient they introduce to different noise sampling processes. (1) Independent Gaussian noise - draw for each fc param independent at each forward pass (used in A3C variant). (2) Factorised Gaussian noise - one independent noise per output and input per layer (used in DQN variants due to single thread comoutational restraint). Show improved performance and that linearly descreasing exploration noise might not be the best strategy for all games.


#### Key take-away for own work:
Everything can be learned! You as a designer choose how flexible your architecture should/has to be and the resulting computational load.


#### Keypoints:

* Optimism in the face of uncertainty - limited to small state-action spaces; Intrinsic motivation - separation generalization and exploration

* Key idea: Random perturbation-based exploration is not efficient. Instead learn from the environment a state-and-state-of-learning dependent noise process.

* Single change to weight vector = induces consistent and potentially complex state-dep change in policy over multiple time steps! Not only single where you perturb

* Variance of injected noise = energy of perturbation

* Non-Bayesian: Do not maintain explicit distribution over weights during training! SImply inject and automatically tune intensity

* Randomised value function PoV: dont require duplicates of full network
    * Computational complexity usually still dominate by activation multiplications and not by the sampling process

* Noise is additive to parameters and has a learned parameter that is multiplied dot wise - leads to double expectation in the loss function.

* Easy gradient derivation with respect to noise as well as normal parameters. MC estimate over batch and another over the sampled noise possible!

* Two types of Gaussian noise: Activations vs Weights
    * Independent = uses independent Gaussian noise entry per weight - pq+q Gaussians per layer (p inputs to layer, q output)
    * Factorised = uses an independent noise per each output and another per each input (per layer!)- reduces compute time due to random number gen - p+q Gaussians per layer (p inputs to layer, q output)

* Implementation:
    * Sample set of new noise only after every step of the optimisation - between opt updates the agents acts according to fixed set of weights!
    * No eps-greedy/entropy reg based exploration any more
    * Double DQN - 3 sets of noise: 1. Online net, 2. Eval Target net, 3. Select Online net
    * A3C - Adding Noise weights and sampling such corresponds to choosing a different current policy - naturally favours exploration - optimisation every n-steps!

#### Questions/Critical Observations:

* Relationship to supervised learning dropout/gaussian noise ideas of robustness and Bayesian DL/GP formulations?
* Can the network actually learn state dependent exploration or simply amplification/decrease of noise?
* Is there a relationship between n in n-step returns and the variance of the sampled noise!?
