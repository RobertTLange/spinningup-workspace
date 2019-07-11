# Title: Soft Actor-Critic: Off-Policy Maximum Entropy DRL with a Stochastic Actor

# Author: Haarnoja et al (2018) (UC Berkeley)

#### General Content:
Introduce first off-policy soft AC algorithm based on maximum entropy objective (max reward while behaving as randomly as possible). This allows for more robustness compared to classic "brittle" DDPG. Furthermore, they provide a convergence proof for soft policy iteration in the tabular case. SAC provides a scalable approximation using function approximators. Finally, the authors show SOTA performance on multiple MuJoCo tasks as well as insights regarding deterministic evaluation, reward scaling as well as Polyak averaging parameter.

#### Key take-away for own work:
Most theoretical guarantees can only be established for the tabular case, i.e. soft policy iteration. There are no guarantees for the non-linear function approx case. Enforcing randomness/entropy allows/nudges the policy to fit multiple modes and therefore leads to more robust performance if the env is noisy!

Also general advice regarding experiments: Easier tasks make easier to initially tune hyperparameters to sensible range. Harder tasks can then refine to even more narrow bins.

#### Keypoints:

* General motivation: Off-policy methods are more sample efficient but tend to exhibit unstable learning dynamics. On-policy is more stable but less sample efficient. DDPG instable due to interplay of deterministic actor and Q-fct/critic.
    * Max entropy methods on the other hand are robust towards model/estimation errors + improve exploration + behavioral diversity
    * Previously used entropy as form of regularizer but not as a specific part of the objective

* Compared to DDPG: Use stochastic actor and max its entropy. Results in more stable and scalable algo!
    * Relationship to Stochastic value gradients (SVG(0) - Heess et al 2015). But does not use a separate network as in AC
    * Compared to previous Haarnoja 2017 work: They motivate actor net as approximate sampler rather that actor in ac framework. Depends on how well sampler approximates posterior. Here: prove convergence regardless of parametrization/policy class
    * Prior work: Directly solve for optimal q-function from which optimal policy is recovered. Here: Instead evaluate q-fct of current policy and update policy through off-policy gradient update

* Proofs assume gamma=1 for easier derivation. Problem: Only rewards and not state transitions are discounted.
    * Based on modified bellman backup operator.
    * Soft policy iteration: Alternate between soft policy eval and soft policy improvement - provably converges to optimal max entropy policy among feasible policies.

* Objective includes additional entropy term weighted by alpha which determines relative entropy importance.

* Main benefits of entropy objective:
    1. Policy is incentivized to explore more widely
    2. Policy can capture multiple modes of near-optimal behavior
    3. Improved exploration behavior!

* Scalable version: 3 networks - Value net, soft Q net, policy network
    * In practice one can recover V from soft Q and policy net. But experiments showed that separate network helps in stabilizing
    * Entropy objective kicks in for value net
    * Keep target network for V and update using Polyak averaging
    * Policy objective is given by KL divergence term which can be simplified with help of reparametrization trick. Extends unbiased DDPG-like PG to any stochastic policy
    * Keep separate network and do Double-Q to combat upward bias

* PPO requires large BS to stabilize on high-dim tasks!

* Reward scaling = role of temperature of energy-based optimal policy - controls stochasticity!

* Polyak tau: Large value leads to instabilities while small tau can slow down learning. If using discrete number of updates between target net update observed that more gradient updates might be useful.

#### Questions/Critical Observations:

* Read original soft q-learning paper by Haarnoja et al 2017 - Establishes relationship between PG and Q-learning when taking probabilistic/entropy reg perspective.
