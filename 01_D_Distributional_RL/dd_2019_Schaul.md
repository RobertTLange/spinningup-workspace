# Title: Ray Interference: A source of plateaus in DRL

# Author: Schaul et al (2019) (DeepMind)

#### General Content:
Ray interference is a phenomenon observed in (multi-objective) DRL when learning dynamics travel through a sequence of plateaus. This is caused by a coupling of learning and data generation arising due to on-policy rollouts. This constrains the agent to learn one thing at a time when parallelization is better. The authors derive a relationship to dynamical systems and show a relationship to saddle points,

#### Key take-away for own work:


#### Keypoints:

* Coupling: Incompetent agent will not generate good training data

* Here: Focus on case where objective has multiple components summed up
    * Multiple tasks/contexts
    * Multiple starting points
    * Multiple opponents
    * Domains with bifurcating dynamics

* Knowledge sharing/representations between components can lead to skull reuse/generalization but in general different components do not coordinate and compete for resources - resulting in interference

* Ray interference = learning system suffers from plateaus if it has negative interference between components of objective and coupled performance and learning progress
    * Reason: Negative interference creates winner-take-all regions
    * Progress in one component forces learning to stall/regress for the other
    * Due to: NN sharing of representations + improving behavior policy generates the future training data

* Minimal contextual multi-arm bandit example
    * Sharing of parameters across context
    * Interference formally defined as cosine sim between two components gradients
    * positive = transfer, negative = interference
    * Plateau = learning curve switches from concave to convex
    * Basins of attractions & Winner take all dynamics - here: very high chance of initialization in one of them
    * Supervised learning can overcome! still interference but no saddle!
    * Plateaus become more severe as number of arms/contexts grows
    * Plateaus become more prolonged after each stage!
        1. low initial performance dramatically affects the chance that the dynamics go through flat plateau
        2. previous tasks can dominate update such that they move params in direction to revocer their performance. null-space becomes smaller with each task

* RL generalization (no example)
    * Simple MA - multiple rooms, levels/opponents
    * Additive sum of objectives as a first-order approx
    * HRL: Split trajectories near reward events
    * Different resource computations: memorize, represent states, options to refine, where to improve value accuracy
    * Potential positive transfer - UNREAL
    * Off-policy learning can uncouple!

* Beyond rl
    * Talk about memorization - Ganguli linear nets svd work
    * Task dominance in multi-task learning

* PBT can shield against effect - hides plateaus through reliance on other members of the population

#### Questions/Critical Observations:
