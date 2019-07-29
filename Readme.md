# Deep Reinforcement Learning Workbook
## Author: Robert Tjarko Lange | 2019

In this repository I document my self-study of Deep Reinforcement Learning. More specifically, I collect reading notes as well as reproduction attempts. The chronology of this repository is based on the amazing ["Spinning Up in DRL"](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) tutorial by OpenAI which in my mind is the best resource on SOTA DRL as of today.

Here are all papers and the corresponding notes that I got to read so far:

# 1. Model-Free RL: (a) Deep Q-Learning

* [x] [Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.](01_A_Deep_Q_Learning/01_2013_Mnih.md)
* [x] [Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone, 2015.](01_A_Deep_Q_Learning/02_2015_Hausknecht.md)
* [x] [Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015.](01_A_Deep_Q_Learning/03_2015_Hasselt.md)
* [x] [Prioritized Experience Replay, Schaul et al, 2016.](01_A_Deep_Q_Learning/04_2016_Schaul.md)
* [x] [Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2016.](01_A_Deep_Q_Learning/05_2016_Wang.md)
* [x] [Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al, 2017.](01_A_Deep_Q_Learning/06_2017_Hessel.md)

**Supplementing Papers**

* [x] [Noisy Networks for Exploration, Fortunato et al, 2018.](01_A_Deep_Q_Learning/xx_2018_Fortunato.md)
* [x] [A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play, Silver et al 2019.](01_A_Deep_Q_Learning/yy_2019_Silver.md)

# 1. Model-Free RL: (b) Policy Gradients

* [x] [Asynchronous Methods for Deep Reinforcement Learning, Mnih et al, 2016.](01_B_Policy_Gradients/07_2016_Mnih.md)
* [x] [Trust Region Policy Optimization, Schulman et al, 2015.](01_B_Policy_Gradients/08_2015_Schulman.md)
* [x] [High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al, 2016. Algorithm: GAE.](01_B_Policy_Gradients/09_2015_Schulman.md)
* [x] [Proximal Policy Optimization Algorithms, Schulman et al, 2017.](01_B_Policy_Gradients/10_2017_Schulman.md)
* [x] [Emergence of Locomotion Behaviours in Rich Environments, Heess et al, 2017.](01_B_Policy_Gradients/11_2017_Heess.md)
* [x] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation, Wu et al, 2017.](01_B_Policy_Gradients/12_2017_Wu.md)
* [x] [Sample Efficient Actor-Critic with Experience Replay, Wang et al, 2016.](01_B_Policy_Gradients/13_2017_Wang.md)
* [x] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018. Algorithm: SAC.](01_B_Policy_Gradients/14_2018_Haarnoja.md)

# 1. Model-Free RL: (c) Deterministic Policy Gradients
* [x] [Deterministic Policy Gradient Algorithms, Silver et al, 2014.](01_C_Deterministic_Policy_Gradients/15_2014_Silver.md)
* [x] [Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.](01_C_Deterministic_Policy_Gradients/16_2016_Lillicrap.md)
* [x] [Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018.](01_C_Deterministic_Policy_Gradients/17_2018_Fujimoto.md)

# 1. Model-Free RL: (d) Distributional RL
* [x] [A Distributional Perspective on Reinforcement Learning, Bellemare et al, 2017.](01_D_Distributional_RL/18_2017_Bellemare.md)
* [ ] Distributional Reinforcement Learning with Quantile Regression, Dabney et al, 2017.
* [ ] Implicit Quantile Networks for Distributional Reinforcement Learning, Dabney et al, 2018.
* [ ] Dopamine: A Research Framework for Deep Reinforcement Learning, Anonymous, 2018.

# 1. Model-Free RL: (e) Policy Gradients with Action-Dependent Baselines
* [ ] Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic, Gu et al, 2016. Algorithm: Q-Prop.
* [ ] Action-depedent Control Variates for Policy Optimization via Stein’s Identity, Liu et al, 2017. Algorithm: Stein Control Variates.
* [ ] The Mirage of Action-Dependent Baselines in Reinforcement Learning, Tucker et al, 2018. Contribution: interestingly, critiques and reevaluates claims from earlier papers (including Q-Prop and stein control variates) and finds important methodological errors in them.

# 1. Model-Free RL: (f) Path-Consistency Learning
* [ ] Bridging the Gap Between Value and Policy Based Reinforcement Learning, Nachum et al, 2017.
* [ ] Trust-PCL: An Off-Policy Trust Region Method for Continuous Control, Nachum et al, 2017.

# 1. Model-Free RL: (g) Other Directions for Combining Policy-Learning and Q-Learning
* [ ] Combining Policy Gradient and Q-learning, O’Donoghue et al, 2016.
* [ ] The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning, Gruslys et al, 2017.
* [ ] Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning, Gu et al, 2017.
* [ ] Equivalence Between Policy Gradients and Soft Q-Learning, Schulman et al, 2017.

# 1. Model-Free RL: (h) Evolutionary Algorithms
* [ ] Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Salimans et al, 2017.
