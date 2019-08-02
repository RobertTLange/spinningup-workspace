# Deep Reinforcement Learning Workbook
## Author: Robert Tjarko Lange | 2019

In this repository I document my self-study of Deep Reinforcement Learning. More specifically, I collect reading notes as well as reproduction attempts. The chronology of this repository is based on the amazing ["Spinning Up in DRL"](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) tutorial by OpenAI which in my mind is the best resource on SOTA DRL as of today.

Here are all papers and the corresponding notes that I got to read so far:

| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
13/05/19 #1 - :fire: | Playing Atari with Deep Reinforcement Learning, Mnih et al. | 2013 | Deep Q-Learning | DQN | [Click](https://arxiv.org/pdf/1312.5602.pdf)) | [Click](01_A_Deep_Q_Learning/01_2013_Mnih.md) |
| 01/12/18 #2 - :fire: | Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone | 2015 | Deep Q-Learning |DRQN | [Click](https://arxiv.org/abs/1507.06527) | [Click](01_A_Deep_Q_Learning/02_2015_Hausknecht.md) |
16/05/19 #3 - :fire: | Deep Reinforcement Learning with Double Q-learning, van Hasselt et al.| 2015| Deep Q-Learning | DDQN | [Click](https://arxiv.org/abs/1509.06461) | [Click](01_A_Deep_Q_Learning/03_2015_Hasselt.md) |
| 17/05/19 #4 - :fire:| Prioritized Experience Replay, Schaul et al. | 2016 | Deep Q-Learning | PER | [Click](https://arxiv.org/pdf/1511.05952.pdf) | [Click](01_A_Deep_Q_Learning/04_2016_Schaul.md) |
| 15/05/19 #5 - :fire: | Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. | 2016 | Deep Q-Learning | Dueling DQN | [Click](https://arxiv.org/pdf/1511.06581) | [Click](01_A_Deep_Q_Learning/05_2016_Wang.md) |
| 17/05/19 #6 - :fire:| Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al. | 2017 | Deep Q-Learning |Rainbow| [Click](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17204/16680) | [Click](01_A_Deep_Q_Learning/06_2017_Hessel.md) |
| 24/05/19 #7 - :fire:| Noisy Networks for Exploration, Fortunato et al. | 2018 | Deep Q-Learning | Noisy Nets|[Click](https://arxiv.org/pdf/1706.10295) | [Click](01_A_Deep_Q_Learning/xx_2018_Fortunato.md) |
| 25/05/19 #8 - :fire:| A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play, Silver et al. |2019| Deep Q-Learning | AlphaZero | [Click](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/alphazero-science.pdf) | [Click](01_A_Deep_Q_Learning/yy_2019_Silver.md) |
|25/05/019 #9 - :key: | Asynchronous Methods for Deep Reinforcement Learning, Mnih et al. |  2016 | Policy Gradients | A3C | [Click]() | [Click](01_B_Policy_Gradients/07_2016_Mnih.md) |
|29/05/019 #10 - :key: | Trust Region Policy Optimization, Schulman et al.| 2015 | Policy Gradients | TRPO | [Click](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf) | [Click](01_B_Policy_Gradients/08_2015_Schulman.md) |
|11/06/019 #11 - :key: | High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. | 2016 | Policy Gradients | GAE | [Click](https://arxiv.org/pdf/1506.02438) | [Click](01_B_Policy_Gradients/09_2015_Schulman.md) |
|18/06/019 #12 - :key: |Proximal Policy Optimization Algorithms, Schulman et al. | 2017 | Policy Gradients | PPO | [Click](https://arxiv.org/pdf/1707.06347) | [Click](01_B_Policy_Gradients/10_2017_Schulman.md) |
| 20/06/19 #13 - :key: | Emergence of Locomotion Behaviours in Rich Environments, Heess et al. | 2017 | Policy Gradients | - | [Click](https://arxiv.org/pdf/1707.02286) | [Click](01_B_Policy_Gradients/11_2017_Heess.md) |
| 20/06/19 #14 - :key: | Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation, Wu et al. | 2017 | Policy Gradients | ACKTR | [Click](http://papers.nips.cc/paper/7112-scalable-trust-region-method-for-deep-reinforcement-learning-using-kronecker-factored-approximation.pdf) | [Click](01_B_Policy_Gradients/12_2017_Wu.md) |
| 09/07/19 #15 - :key: | Sample Efficient Actor-Critic with Experience Replay, Wang et al. | 2016 | Policy Gradients | ACER | [Click](https://arxiv.org/pdf/1611.01224) | [Click](01_B_Policy_Gradients/13_2017_Wang.md) |
| 11/07/19 #16 - :key:| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al. | 2018 | Policy Gradients | SAC | [Click](https://arxiv.org/pdf/1801.01290) | [Click](01_B_Policy_Gradients/14_2018_Haarnoja.md) |
| 09/07/19 #17 - :monkey:|  Deterministic Policy Gradient Algorithms, Silver et al. | 2014 | Deterministic PG | DPG | [Click]() | [Click](01_C_Deterministic_Policy_Gradients/15_2014_Silver.md) |
| 10/07/19 #18 - :monkey:| Continuous Control With Deep Reinforcement Learning, Lillicrap et al. | 2015 | Deterministic PG | DDPG | [Click](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf) | [Click](01_C_Deterministic_Policy_Gradients/16_2016_Lillicrap.md) |
| 12/07/19 #19 - :monkey:| Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al. | 2018 | Deterministic PG | TD3 | [Click](https://arxiv.org/pdf/1802.09477.pdf),) | [Click](01_C_Deterministic_Policy_Gradients/17_2018_Fujimoto.md) |
| 29/07/19 #20 - :moon:| A Distributional Perspective on Reinforcement Learning, Bellemare et al.| 2017 | Distributional RL | C51 | [Click](https://arxiv.org/pdf/1707.06887) | [Click](01_D_Distributional_RL/18_2017_Bellemare.md) |
| 31/07/19 #21 - :moon:| Distributional Reinforcement Learning with Quantile Regression, Dabney et al. | 2017 | Distributional RL | QR-DQN| [Click](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17184/16590) | [Click](01_D_Distributional_RL/19_2017_Dabney.md) |
| 01/08/19 #22 - :moon:| Implicit Quantile Networks for Distributional Reinforcement Learning, Dabney et al. | 2018 | Distributional RL | IQN | [Click](https://arxiv.org/pdf/1806.06923) | [Click](01_D_Distributional_RL/20_2018_Dabney.md) |
| 02/08/19 #23 - :moon:| Deep Reinforcement Learning and the Deadly Triad, van Hasselt et al. | 2018 | Deep Q-Learning | - | [Click](https://arxiv.org/abs/1812.02648) | [Click](01_D_Distributional_RL/aa_2018_Hasselt.md) |

# 1. Model-Free RL: (d) Distributional RL
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
