# Title: A general RL algorithm that masters chess, shogi and Go through self-play

# Author: Silver et al (2019) (DeepMind)

#### General Content:
Extend AlphaGo Zero to all two-player complete info games. Algorithm learns entirely from self-play ("tabula rasa") and without human games/knowledge (except for game rules) - learning from "first-principles".  


#### Key take-away for own work:
Evolution through self-play, search and learning + immense amounts of computational power = FUTURE (Rich Sutton blog post).


#### Keypoints:

* AlphaZero: Replace handcrafted features with DRL and MCTS.
    * DRL: Changed two networks (value and policy) into single architecture with two heads. Takes board position as input and returns move probability vector as well as scalar value estimate. Uses both to guide the search.
    * MCTS: Series of simulated games of self-play that traverse a tree from root until leaf is reached. Moves are selected by trading of low visit count, high move probability and high value according to network. Search returns vector representing prob distr over moves given root node.
    * Train network via self-play: MCTS search from current position at t, select move according to pi_t. At end of game, scoring -1, 0, 1. Update params to minimize error between predicted outcome and game outcome and to max similarity of policy vector and search probs

* Changes compared to AlphaGo Zero:
    1. No binary outcome as in Go anymore. Instead estimate the expected outcome and not max prob. of winning.
    2. Rules of go invariant to rotation and reflection - allowed for augmentation and MC evaluation could be averaged over different biases. Chess and shogi do not allow for such augmentation.
    3. Before best player of all previous iterations was measured against new player. AlphaZero maintains single net and updates it continually rather than waiting for iteration to complete. Self-play games always generated with latest params
    4. Game state + action is encoded in spatial planes based on basic rules of game
    5. Still use conv architecture even though rules of games might not be translationally invariant.
    6. No hyperparam tuning - use BO results from AlphaGo Zero for all games

* Alpha Zero searches less positions per second compared to stockfish and Elmo - Hypothesis: Compensation by using deep net to focus much more selectively on most promising variations.

* Previous work alpha-beta search: AlphaZero MCTS averages over position evaluations within a subtree rather than computing the minimax eval of that subtree. Speculate that approx errors induced by NNs tend to cancel out when evaluating a large subtree
    * Soft minimax averaged over many rollouts provides form of MCTS of alpha-beta search? - Never hard max or min since convergence never guaranteed due to non-stationary self-play learning

#### Questions/Critical Observations:

* Gary Marcus critique: MCTS is a huge human made inductive bias and not a first principle

* How much compute time/Floaps goes into 5000 1st gen TPUs and self-play + 16 2nd gen TPU training?
