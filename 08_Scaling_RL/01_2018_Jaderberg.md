# Title: Human-Level performance in first-person multiplayer games with population-based DRL

# Author: Jaderberg et al. 2018

#### General Content:

Introduce the population-based training setup of Jaderberg in the context of a Quake III multi-agent game. Contributions (all combined with massively parallel and async training):

* overcome severe sparsity by evolving (PBT learned hyperparams) an internal reward signal (based on game points not game win).
* introduce multi-timescale LSTM architecture which allows for hierarchical representations. Can be formulated in terms of Bayesian doctrine. Slow timescale sets prior to fast timescale. Use as a regularizer in training.
* use external working memory module as in alex graves work

#### Key take-away for own work:

* KL-regularized RL seems to be everywhere in large-scale projects (also in alphastar). Have to look more into RL as inference literature
* UNREAL loss components provide great support & simply more signal to train on

#### Keypoints:

* Key problem: non-stationarity from "concurrent adaptation of other learning agents". Still here: decentralized, no model access, no human policy priors, no communication

* Capture-the-flag game setup:
    * Team with greatest number of flag captures within 5 minutes wins
    * In later evaluation: make sure to adjust capabilities to human level
        * Make tagging reaction times and tagging accuracy fair

* Algorithm details:
    * Each agent individually maximises prob of team winning (but too sparse!)
    * Need to learn dense internal reward from game points - use parametrisation that is optimised via PBT
    * Dont use self-play: unstable and does not support concurrent training - crucial for scalability
    * Instead train in parallel and introduce diversity - first form of league!
    * Use stochastic matchmaking scheme to ensure similar strength of players
    * Different levels of optimization:
        * Inner: max exp future discounted internal rewards - RL/SGD
        * Outer: meta game: max wrt to internal rew and hyperparams - Evolutionary/PBT
    * PBT optimized hyperparameters include learning rate, KL weighting, internal timescale

* Policy: Multiscale RNN with external memory
    * Actions generated conditional on stochastic latent variable
    * Slow timescale: Prior, Fast timescale: Posterior
    * Variational objective: trades off max exp reward & consistency between timescales of inference - relationship to hierarchical temporal representations

* Training details:
    * Train 30 agents in parallel for PBT
    * Classic A3C loss + KL regularization towards prior
    * IMPALA: actor-learner structure + V-trace off-policy correction
    * RMSProp optimiser with log uniform sampled initial learning rate
    * Use UNREAL loss components (pixel control, reward prediction aux losses)

* Results details:
    * Algo policy is robust maps, no players, teammate choice (also human!)
    * Extract ground-truth state & find that network representation are strong at decoding - evidence that algo learns about key quantities
    * Visualization using tsne and similarity-preserving topological embeddings
    * Analyse sequential acquisition of knowledge - "emergence of concepts" - concept cells


#### Questions/Critical Observations:

* Open challenges:
    * Maintain multi-agent diversity - later the league!
    * PBT: Greedy meta-optimization
    * RL variance/sample efficiency
