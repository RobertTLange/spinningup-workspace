# Title: Grandmaster level in Starcraft II using MARL

# Author: Vinyals et al. (2019; DeepMind)

#### General Content:
Introduce MA-RL architecture + learning paradigm that achieves great performance of SCII

#### Key take-away for own work:

* Very similar to CTF-FTW setup with IMPALA-based Actor-Learner + special KL regularization
    * Instead of using dense internal rewards learned via PBT - here: human latent prior
    * Also no two-timescale RNN but instead autoregressive policy to tame the large action space

#### Keypoints:

* Main problem: Game theoretic challenge of cyclic, non-transitice strategies + combinatorial action space (What where when - 10^26 actions each step) + 10 minute long credit assignment

* Constraints for Human comparison: limit reaction times as well as rate of actions via Monitoring layer (Still processing at fast rate?)

* Training for 44 days on TPUs; 900 distinct players were created in league

* Learning Algo:
    * Condition policy on latent z statistic that summarizes strategy sampled from human data + Self-attention + LSTM + A3C + Autoregressive policy + Recurrent pointer network
    * The League alternative to standard self-play - structured approach to ensure diversity
    * Max win rate + KL regularization when human statistic given (similar to FTW with two timescale RNNs)
    * Structures exploration problem by keeping initial behavior close to human prior
    * Advantage AC + Replay Buffer = off-Policy
        * But need only little correction since in large action spaces current and previous policies are highly unlikely to match over many steps - otherwise correction to large!
        * Use combination of techniques to learn effectively despite mismatch: Clipped importance sampling (V-trace) + UPGO self-imitation algo = move policy towards trajectories with better-than-avg-reward
        * Variance reduction by conditioning critic on both players observations
    * Human prior regularization leads to wide variety of relevant modes of play and their exploration

* Fictitious Self-Play (FSP): Avoid cycles by computing best response against uniform misture of all previous policies - with mixture convergence to nash equilibrium in two-player 0-sum games
    * Extend to best response with non-uniform mixture - The league - agent-specific mixtures - 3 distinct types of agents - different selection mechanisms:
        1. Main agents = prioritized FSP: Adapt mixture probs proportionately to win rate of each opponent against the agent - rapid learning as in traditional self-play
        2. Main exploiters = Identity potential exploits in main agents - encourage main agents to address weaknesses
        3. League exploiters = Similar PFSP but not targeted by main exploiter agents - find systematic weaknesses of entire league
    * Main exploiters & league exploiters are periodically reinitialized to encourage diversity and rapid discovery of specialist strategies

* Evaluation:
    * Only play against opponents once: strength under approx stationary conditions
    * Round-robin tournament within the league shows that main agents grew increasingly robust: Nash eq over all payers at each point in time assigns small probs to players from previous iterations - learning does not cycle/regress

* Details from appendix
    * Autoregressiv action sampling: condition on the ouputs of the LSTM and the observation encoders
    * Main reason for supervised learning/imitation learning: Maintain diverse exploration
    * Summary statistics z = first 20 constructed building and units + cumulative statistics (units, buildings, effect & upgrades) during game
    * Set z to 0 in 10% of cases
    * Adam optimizer + L2 regularization + gamma=1 no discount
    * Pseudo rewards to follow z: Measure edit distance between sampled and executed build orders, hamming distance between sampled and executed cum stats - must be differentiable!
    * V-trace: corrections truncate trace early - assume indep action type, delay and other arguments - clipped importance ratio
    * UPGO; update the policy from partial trajectories with better-tahn-exp returns by bootstrapping when behaviour policy takes worse-than-avg action
    * Populate league by regularly saving params from agents
    * Different weightings for PFSP: Hard vs Var
    * Main Agents: 35% SP, 50% PFSP all past players in league, 15% PFSP matches against forgotten main players, past main exploiters
    * League Exploiters: PFSP - identify global blind spots in league, add weights to league if they defeat all players in league in more than 70% of games or after timeout; 25% chance of reset to supervised parameters
    * Main Exploiters: Play against main agents. Half of time + if prob of winning lower than 20%, exploiters use f_var PFSP weighting over players created by main agent


#### Questions/Critical Observations:

* Relationship to Policy Space Response Oracle Framework
