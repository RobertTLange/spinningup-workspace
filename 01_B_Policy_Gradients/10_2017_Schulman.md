# Title: Proximal Policy Optimization Algorithms

# Author: Schulman et al (OpenAI)

#### General Content:
Introduce a first order algorithm (no Fisher info matrix needed) that emulates TRPO monotonic improvement result. Intend to make PPO easier than TRPO and with clipped surrogate objective and/or adaptive KL penality coefficient.


#### Key take-away for own work:
Optimal clipping constant is around 0.2. Have to read up on ACER. Can easily combine all PG/AC methods with trust region and surrogate clipping objective.

#### Keypoints:

* Two novel additions:
    * Clipped surrogate objective (PPO): Min + Clipping restricts update to lie within a specific proportion of the original policy. This replaces the constraint and incorporates it within the objective
    * Adaptive KL Penalty Coefficient: Increase/decrease adhock based on whether policy change (measured by KL) was smaller or bigger than target. Essentially ensures steady progress and prohibts too strong updates to policy.

* Easy to incorporate into Actor-Critic style architectures: Must use loss fct that combines policy surrogate and value fct error term - additionally include entropy term for exploration
    * A3C introduced k-Step returns - need advantage estimator that looks beyond k
    * Run with parallel workers that collect the trajectories
    * A2C do synchronous: and average gradients across the workers

#### Questions/Critical Observations:
* Authors argue that it is challenging to pick a single beta across different tasks - but is it really easier to bick an epsilon?
* Not really clear to me what PPO really means. In Deepmind Locomotion paper they essentially treat the adaptive KL TRPO version as PPO. Is this correct?
