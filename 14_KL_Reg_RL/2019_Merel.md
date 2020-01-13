# Title: Neural Probabilistic Motor Primitives for Humanoid Control

# Author: Merel et al. (2019)

#### General Content:
Propose novel architecture of a inverse model with latent-var bottleneck. Train it entirely offline to compress many expert policies - results in motor primitive embedding space. Afterwards, the system is capable of doing one-shot imitation of whole-body behaviors. Behavior appears naturalistic.

#### Key take-away for own work:


#### Keypoints:

* Few-shot humanoid control given motion capture data
    * Focus on case with large number of experts performing single skills well - wish to transfer into shared policy while maintaining generalization capabilities.
    * NPMP: Learn a "dense" embedding space of the individual expert policies.
    * Network is capable of reusing, combining & sequencing distilled behaviors
    * Goal: Avoid closed-loop - transfer from supervised learning

* Dont do behavioral cloning - may require many expert rollouts for robustness
    * Instead: Matching of local noise-feedback properties along single representative expert reference trajectory
    * Call this approach: Linear Feedback Policy Cloning

* Related Literature:
    * Distillation vs imitation = main difference in goal - distillation wants to compress!
    * DAGGER, Ross et al. (2011): Online Imitation Learning
    * Mordatch et al. (2015) - used linear-feedback policies to train NNs

* CMU MOCAP data - yields time-indexed NNs robust to moderate noise
    * Cut 6 second snippets into smaller parts - 2 secs - total 2707 expert policies

* Aim: Need **representations** that not only encode behavioral modes but also allow for effective indexing of behaviors at recall
    * Train autoregressive latent-var model of state-conditional action sequence
    * Condition on small look-ahead sequence - nominal/reference trajectory
    * Maintains general structure of inference model - produce action given state and target
    * Latent state is forwarded through time by an AR(1) process
    * End-to-end training of entire architecture using ELBO + Autoencoder loss
    * Effectively resembles an information bottleneck between future trajectory and action given past latent state

* **Student Training**: conceptualize experts as nonlinear feedback controllers around nominal trajectory - think of states visited as tube around reference
    * behavioral cloning objective may be too sample inefficient to optimize
    * Two alternative options:
        1. Noisy expert trajectories + logging of optimal action of expert - DART
        2. Action-State Jacobian logging + optimal action - single nominal trajectory
            - use jacobian to construct linear feedback controller
            - emprically this is more sample efficient
            - boils down to a form of pertubation objective/denoising autoencoder

* Experiments:
    - open loop control fails even with small noise in single-skill transfer. LFPC on the other performs well/as good as methods trained on many thousands of trajectories.
    - Compression of 2707 experts - important to compare actual behavior - decoder has learned to stabilize noisy encodings!
    - Important: Regularization and large embedding spaces!
    - Failures do not arise due to encoder - decoder might be undertrained on borderline cases
    - Investigate transfer of pretrained NPMP module

#### Questions/Critical Observations:
