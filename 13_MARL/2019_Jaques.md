# Title: Social Influence as Intrinsic Motivation for MA-DRL

# Author: Jaques et al. (2019, MIT)

#### General Content:
Define intrinsic reward as causal influence on other agents' actions and link this to a social form of empowerment. Show an equivalence to high mutual info between actions. This may lead to better/meaningful communication/coordination. Combine the setup with Theory of Mind to compute rewards in a decentralized manner. Goal hereby is to overcome the classical CT-DE paradigm. First form of social inductive bias ?!

#### Key take-away for own work:
Extremely well-written paper! Super clear structure and build-up.

#### Keypoints:

* Reward = influence on other agents policy execution - via counterfactual reasoning - Pearl causality perspective.
* Empirical observation: Social influence leads to better coordination & communication
* Show that rewards can be computed in a decentralized fashion by making agents learn form of ToM.

* Intrinsic motivation = pseudo-rewards.
    * Previously for MARL: Hand-crafted (specific to env) with centralisation assumption
    * Focus on empowerment and curiosity
    * Here: Key idea - actions that lead to relatively higher change in other agents' behavior are considered to be influential and are rewarded. - Form of social inductive bias!
    * Social influence = social form of empowerment

* Experiments + Results::
    * SSD (sequential social dilemma) - Leibo et al 2017 - Prisoner's dilemma style
    * E1: Basic - Train RNN policies from pixels - higher coordination/collective rewards
        * Implicit communication through behavior
    * E2: Comm - Explicit communication channel - influential communication is beneficial to receiver
    * E3: Other Agents Modelling - Independent training with Model of Other Agents (MOA)

* MOA: Use to predict how simulated counterfactual actions will affect other agents - compute own intrinsic influence reward
* Architecture: CNN+LSTM+Linear trained using A3C

* Basic Social Influence: Decomposition of emitted reward into env/extrinsic + intrinsic/social
    * Counterfactual idea: p(a_j|a_k, s_j) replace a_k by tilde_a_k - compute distribution: How would js action change if I had acted differently in this situation
    * Marginalizing lead to p(a_j|s_j) - no consideration of agent k
    * Discrepancy between marginal and conditional = measure of social influence - c = KL divergence! Degree to which j changes its planned action distribution because of ks action
    * General problem: Policy gradients variance increases in number of agents - social influence can reduce by introducing explicit dependencies across actions of each agent
    * Two assumptions:
        1. Centralized training for computing c
        2. Assume unidirectional influence
    * Training: use a curriculum learning approach - gradually increase weight of social influence reward. - Need to first learn a little greedy behavior?

* Influential Communication: AC with 4 heads (2 actors/critics) with discrete communication
    * Only propagate influence gradients through the message heads - form of aux task
    * Main benefit: Agents can choose to ignore non-cooperative communication - free
    * Hypothesis: Influential communication must provide useful info to the listener!
    * Analysis of the learned communication using Bogin et al. 2018:
        - Speaker consistency: how consistently speaker emits a particular symbol
        - Instantaneous coordination:
            - symbol/action: MI influencers symbol and listeners next action
            - action/action: MI influencers action and listeners next action
        - Average over trajectory steos and compute max between two agents to determine coordination
    * Influence is sparse in time - selective listening only when beneficial - learned meaningful communication

* Modelling other agents: Previous - required knowledge of prob of another agents action given a counterfactual - solved by centralization
    * MOA = second set of fully conn + LSTM layers connected to agents conv layer (split streams) - used to predict all other agents next actions, given previous actions and egocentric state - give itself rewards
    * Unsupervised aux task - learn better embedding layer

#### Questions/Critical Observations:
* How is mutual info equivalence related to notion of transfer entropy (which has temporal lags!)
