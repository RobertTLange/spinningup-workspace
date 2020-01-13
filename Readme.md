# Deep Reinforcement Learning Workbook
## Author: Robert Tjarko Lange | 2019

In this repository I document my self-study of Deep Reinforcement Learning. More specifically, I collect reading notes as well as reproduction attempts. The chronology of this repository is based on the amazing ["Spinning Up in DRL"](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) tutorial by OpenAI which in my mind is the best resource on SOTA DRL as of today.

Here are all papers and the corresponding notes that I got to read so far:

# 1. Deep Q-Learning
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
13/05/19 #1 - :fire: | Playing Atari with Deep Reinforcement Learning, Mnih et al. | 2013 | Deep Q-Learning | DQN | [Click](https://arxiv.org/pdf/1312.5602.pdf) | [Click](01_A_Deep_Q_Learning/01_2013_Mnih.md) |
| 01/12/18 #2 - :fire: | Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone | 2015 | Deep Q-Learning |DRQN | [Click](https://arxiv.org/abs/1507.06527) | [Click](01_A_Deep_Q_Learning/02_2015_Hausknecht.md) |
16/05/19 #3 - :fire: | Deep Reinforcement Learning with Double Q-learning, van Hasselt et al.| 2015| Deep Q-Learning | DDQN | [Click](https://arxiv.org/abs/1509.06461) | [Click](01_A_Deep_Q_Learning/03_2015_Hasselt.md) |
| 17/05/19 #4 - :fire:| Prioritized Experience Replay, Schaul et al. | 2016 | Deep Q-Learning | PER | [Click](https://arxiv.org/pdf/1511.05952.pdf) | [Click](01_A_Deep_Q_Learning/04_2016_Schaul.md) |
| 15/05/19 #5 - :fire: | Dueling Network Architectures for Deep Reinforcement Learning, Wang et al. | 2016 | Deep Q-Learning | Dueling DQN | [Click](https://arxiv.org/pdf/1511.06581) | [Click](01_A_Deep_Q_Learning/05_2016_Wang.md) |
| 17/05/19 #6 - :fire:| Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al. | 2017 | Deep Q-Learning |Rainbow| [Click](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17204/16680) | [Click](01_A_Deep_Q_Learning/06_2017_Hessel.md) |
| 24/05/19 #7 - :fire:| Noisy Networks for Exploration, Fortunato et al. | 2018 | Deep Q-Learning | Noisy Nets|[Click](https://arxiv.org/pdf/1706.10295) | [Click](01_A_Deep_Q_Learning/xx_2018_Fortunato.md) |
| 25/05/19 #8 - :fire:| A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play, Silver et al. |2019| Deep Q-Learning | AlphaZero | [Click](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/alphazero-science.pdf) | [Click](01_A_Deep_Q_Learning/yy_2019_Silver.md) |
| 25/12/19 - :fire:| Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model, Schrittwieser et al. |2019| Deep Q-Learning | MuZero | [Click](https://arxiv.org/pdf/1911.08265.pdf) | [Click](01_A_Deep_Q_Learning/zz_2019_Schrittwieser.md) |
| 29/07/19 #20 - :moon:| A Distributional Perspective on Reinforcement Learning, Bellemare et al.| 2017 | Distributional RL | C51 | [Click](https://arxiv.org/pdf/1707.06887) | [Click](01_D_Distributional_RL/18_2017_Bellemare.md) |
| 31/07/19 #21 - :moon:| Distributional Reinforcement Learning with Quantile Regression, Dabney et al. | 2017 | Distributional RL | QR-DQN| [Click](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17184/16590) | [Click](01_D_Distributional_RL/19_2017_Dabney.md) |
| 01/08/19 #22 - :moon:| Implicit Quantile Networks for Distributional Reinforcement Learning, Dabney et al. | 2018 | Distributional RL | IQN | [Click](https://arxiv.org/pdf/1806.06923) | [Click](01_D_Distributional_RL/20_2018_Dabney.md) |
| 02/08/19 #23 - :moon:| Deep Reinforcement Learning and the Deadly Triad, van Hasselt et al. | 2018 | Deep Q-Learning | - | [Click](https://arxiv.org/abs/1812.02648) | [Click](01_D_Distributional_RL/aa_2018_Hasselt.md) |
| 09/08/19 #24 - :moon:| Towards Characterizing Divergence in Deep Q Learning, Achiam et al. | 2019 | Deep Q-Learning | PreQN | [Click](https://arxiv.org/abs/1903.08894) | [Click](01_D_Distributional_RL/bb_2019_Achiam.md) |
| 10/08/19 #25 - :moon:| Non-Delusional Q-Learning and Value Iteration, Lu et al. | 2019 | Deep Q-Learning | PCVI/PCQL | [Click](https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration.pdf) | [Click](01_D_Distributional_RL/cc_2018_Lu.md) |
| 15/08/19 #26 - :moon:| Ray Interference: A source of plateaus in DRL, Schaul et al. | 2019 | Deep Q-Learning | - | [Click](https://arxiv.org/abs/1904.11455) | [Click](01_D_Distributional_RL/dd_2019_Schaul.md) |

# 2. Policy Gradient Methods
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
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
| 12/07/19 #19 - :monkey:| Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al. | 2018 | Deterministic PG | TD3 | [Click](https://arxiv.org/pdf/1802.09477.pdf) | [Click](01_C_Deterministic_Policy_Gradients/17_2018_Fujimoto.md) |


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

# 2. Exploration

| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | VIME: Variational Information Maximizing Exploration, Houthooft et al, 2016. Algorithm: VIME. |   |  |  | [Click]() | [Click]() |
| # :question: | Unifying Count-Based Exploration and Intrinsic Motivation, Bellemare et al, 2016. Algorithm: CTS-based Pseudocounts. |   |  |  | [Click]() | [Click]() |
| # :question: | Count-Based Exploration with Neural Density Models, Ostrovski et al, 2017. Algorithm: PixelCNN-based Pseudocounts. |   |  |  | [Click]() | [Click]() |
| # :question: |Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning, Tang et al, 2016. Algorithm: Hash-based Counts. |   |  |  | [Click]() | [Click]() |
| # :question: | EX2: Exploration with Exemplar Models for Deep Reinforcement Learning, Fu et al, 2017. Algorithm: EX2. |   |  |  | [Click]() | [Click]() |
| # :question: | Curiosity-driven Exploration by Self-supervised Prediction, Pathak et al, 2017. Algorithm: Intrinsic Curiosity Module (ICM). |   |  |  | [Click]() | [Click]() |
| # :question: | Large-Scale Study of Curiosity-Driven Learning, Burda et al, 2018. Contribution: Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments. |   |  |  | [Click]() | [Click]() |
| # :question: | Exploration by Random Network Distillation, Burda et al, 2018. Algorithm: RND. |   |  |  | [Click]() | [Click]() |
| # :question: | Variational Intrinsic Control, Gregor et al, 2016. Algorithm: VIC. |   |  |  | [Click]() | [Click]() |
| # :question: | Diversity is All You Need: Learning Skills without a Reward Function, Eysenbach et al, 2018. Algorithm: DIAYN. |   |  |  | [Click]() | [Click]() |
| # :question: | Variational Option Discovery Algorithms, Achiam et al, 2018. Algorithm: VALOR. |   |  |  | [Click]() | [Click]() |

# 3. Transfer and Multitask RL
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Progressive Neural Networks, Rusu et al, 2016. Algorithm: Progressive Networks. |   |  |  | [Click]() | [Click]() |
| # :question: | Universal Value Function Approximators, Schaul et al, 2015. Algorithm: UVFA. |   |  |  | [Click]() | [Click]() |
| 04/11/19 #3 - :smile: | Reinforcement Learning with Unsupervised Auxiliary Tasks, Jaderberg et al | 2016 | Auxiliary | UNREAL | [Click](https://arxiv.org/abs/1611.05397) | [Click](03_Transfer_and_Multitask_RL/03_2016_Jaderberg.md) |
| # :question: | The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, Cabi et al, 2017. Algorithm: IU Agent. |   |  |  | [Click]() | [Click]() |
| # :question: | PathNet: Evolution Channels Gradient Descent in Super Neural Networks, Fernando et al, 2017. Algorithm: PathNet. |   |  |  | [Click]() | [Click]() |
| # :question: | Mutual Alignment Transfer Learning, Wulfmeier et al, 2017. Algorithm: MATL. |   |  |  | [Click]() | [Click]() |
| # :question: | Learning an Embedding Space for Transferable Robot Skills, Hausman et al, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Hindsight Experience Replay, Andrychowicz et al, 2017. Algorithm: Hindsight Experience Replay (HER). |   |  |  | [Click]() | [Click]() |

# 4. Hierarchy
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Strategic Attentive Writer for Learning Macro-Actions, Vezhnevets et al, 2016. Algorithm: STRAW. |   |  |  | [Click]() | [Click]() |
| # :question: | FeUdal Networks for Hierarchical Reinforcement Learning, Vezhnevets et al, 2017. Algorithm: Feudal Networks |   |  |  | [Click]() | [Click]() |
| # :question: | Data-Efficient Hierarchical Reinforcement Learning, Nachum et al, 2018. Algorithm: HIRO. |   |  |  | [Click]() | [Click]() |

# 5. Memory
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Model-Free Episodic Control, Blundell et al, 2016. Algorithm: MFEC. |   |  |  | [Click]() | [Click]() |
| # :question: | Neural Episodic Control, Pritzel et al, 2017. Algorithm: NEC. |   |  |  | [Click]() | [Click]() |
| # :question: | Neural Map: Structured Memory for Deep Reinforcement Learning, Parisotto and Salakhutdinov, 2017. Algorithm: Neural Map. |   |  |  | [Click]() | [Click]() |
| # :question: | Unsupervised Predictive Memory in a Goal-Directed Agent, Wayne et al, 2018. Algorithm: MERLIN. |   |  |  | [Click]() | [Click]() |
| # :question: | Relational Recurrent Neural Networks, Santoro et al, 2018. Algorithm: RMC. |   |  |  | [Click]() | [Click]() |

# 6. Model-Based RL
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| 30/12/19 # :smile: |  Dream to Control: Learning Behaviors by Latent Imagination, Hafner et al. |  2019 | Model Learning | Dreamer | [Click](https://arxiv.org/pdf/1912.01603.pdf) | [Click](06_Model_Based/2019_Hafner.md) |
| # :question: | Imagination-Augmented Agents for Deep Reinforcement Learning, Weber et al, 2017. Algorithm: I2A. |   |  |  | [Click]() | [Click]() |
| # :question: | Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning, Nagabandi et al, 2017. Algorithm: MBMF. |   |  |  | [Click]() | [Click]() |
| # :question: | Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning, Feinberg et al, 2018. Algorithm: MVE. |   |  |  | [Click]() | [Click]() |
| # :question: | Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion, Buckman et al, 2018. Algorithm: STEVE. |   |  |  | [Click]() | [Click]() |
| # :question: | Model-Ensemble Trust-Region Policy Optimization, Kurutach et al, 2018. Algorithm: ME-TRPO. |   |  |  | [Click]() | [Click]() |
| # :question: | Model-Based Reinforcement Learning via Meta-Policy Optimization, Clavera et al, 2018. Algorithm: MB-MPO. |   |  |  | [Click]() | [Click]() |
| # :question: | Recurrent World Models Facilitate Policy Evolution, Ha and Schmidhuber, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, Silver et al, 2017. Algorithm: AlphaZero. |   |  |  | [Click]() | [Click]() |
| # :question: | Thinking Fast and Slow with Deep Learning and Tree Search, Anthony et al, 2017. Algorithm: ExIt. |   |  |  | [Click]() | [Click]() |

# 7. Meta-RL
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| 14/11/19 - #1 :smile: | RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning, Duan et al, 2016. Algorithm: RL^2. |   |  |  | [Click]() | [Click]() |
| 06/01/20 - #2 :smile: | Meta-Learners' Learning Dynamics are unlike learners', Rabinowitz | 2019 | Learning Dynamics | - | [Click](https://arxiv.org/abs/1905.01320) | [Click](07_Meta_RL/2019_Rabinowitz.md) |
| # :question: | Learning to Reinforcement Learn, Wang et al, 2016. |   |  |  | [Click]() | [Click]() |
| # :question: | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, Finn et al, 2017. Algorithm: MAML. |   |  |  | [Click]() | [Click]() |
| # :question: | A Simple Neural Attentive Meta-Learner, Mishra et al, 2018. Algorithm: SNAIL. |   |  |  | [Click]() | [Click]() |

# 8. Scaling RL
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :smile: - 23/12/19 | Human-Level performance in first-person multiplayer games with population-based DRL, Jaderberg et al. | 2018 | Multi-Agent | PBT | [Click](https://arxiv.org/abs/1807.01281) | [Click](08_Scaling_RL/01_2018_Jaderberg.md) |
| # :smile: - 24/12/19 | Grandmaster level in Starcraft II using MARL, Vinyals et al. | 2019 | Multi-Agent | League | [Click](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning) | [Click](08_Scaling_RL/02_2019_Vinyals.md) |
| # :smile: - 22/12/19 | Emergent Tool Use From Multi-Agent Autocurricula by Baker et al. | Self-Play | 2019 | CT-DE |  [Paper](https://arxiv.org/abs/1909.07528) | [Notes](08_Scaling_RL/2019_Baker.md)
| # :question: | Accelerated Methods for Deep Reinforcement Learning, Stooke and Abbeel, 2018. Contribution: Systematic analysis of parallelization in deep RL across algorithms. |   |  |  | [Click]() | [Click]() |
| # :question: | IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures, Espeholt et al, 2018. Algorithm: IMPALA. |   |  |  | [Click]() | [Click]() |
| # :question: | Distributed Prioritized Experience Replay, Horgan et al, 2018. Algorithm: Ape-X. |   |  |  | [Click]() | [Click]() |
| # :question: | Recurrent Experience Replay in Distributed Reinforcement Learning, Anonymous, 2018. Algorithm: R2D2. |   |  |  | [Click]() | [Click]() |
| # :question: | RLlib: Abstractions for Distributed Reinforcement Learning, Liang et al, 2017. Contribution: A scalable library of RL algorithm implementations. Documentation link. |   |  |  | [Click]() | [Click]() |

# 9. RL in the Real World
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| #1 :smile: | Solving Rubik's Cubes with a Robotic Hand | 2019 | Robotics | ADR | [Click](https://arxiv.org/abs/1910.07113) | [Click](09_Real_RL/01_2019_Akkaya.pdf) |
| # :question: | Benchmarking Reinforcement Learning Algorithms on Real-World Robots, Mahmood et al, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Learning Dexterous In-Hand Manipulation, OpenAI, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation, Kalashnikov et al, 2018. Algorithm: QT-Opt. |   |  |  | [Click]() | [Click]() |
| # :question: | Horizon: Facebook’s Open Source Applied Reinforcement Learning Platform, Gauci et al, 2018. |   |  |  | [Click]() | [Click]() |

# 10. Safety
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Concrete Problems in AI Safety, Amodei et al, 2016. Contribution: establishes a taxonomy of safety problems, serving as an important jumping-off point for future research. We need to solve these! |   |  |  | [Click]() | [Click]() |
| # :question: | Deep Reinforcement Learning From Human Preferences, Christiano et al, 2017. Algorithm: LFP.|   |  |  | [Click]() | [Click]() |
| # :question: | Constrained Policy Optimization, Achiam et al, 2017. Algorithm: CPO. |   |  |  | [Click]() | [Click]() |
| # :question: | Safe Exploration in Continuous Action Spaces, Dalal et al, 2018. Algorithm: DDPG+Safety Layer. |   |  |  | [Click]() | [Click]() |
| # :question: | Trial without Error: Towards Safe Reinforcement Learning via Human Intervention, Saunders et al, 2017. Algorithm: HIRL. |   |  |  | [Click]() | [Click]() |
| # :question: | Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning, Eysenbach et al, 2017. Algorithm: Leave No Trace. |   |  |  | [Click]() | [Click]() |

# 11. Imitation Learning and Inverse Reinforcement Learning
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| 05/11/19 #1 - :smile: | Distilling Policy Distillation, Czarnecki et al. | 2019 | Distillation | ExpEntropyReg | [Click](https://arxiv.org/abs/1902.02186) | [Click](11_IRL/01_2019_Czarnecki.md) |
| # :question: | Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy, Ziebart 2010. Contributions: Crisp formulation of maximum entropy IRL. |   |  |  | [Click]() | [Click]() |
| # :question: | Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization, Finn et al, 2016. Algorithm: GCL. |   |  |  | [Click]() | [Click]() |
| # :question: | Generative Adversarial Imitation Learning, Ho and Ermon, 2016. Algorithm: GAIL. |   |  |  | [Click]() | [Click]() |
| # :question: | DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, Peng et al, 2018. Algorithm: DeepMimic. |   |  |  | [Click]() | [Click]() |
| # :question: | Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow, Peng et al, 2018. Algorithm: VAIL.|   |  |  | [Click]() | [Click]() |
| # :question: | One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL, Le Paine et al, 2018. Algorithm: MetaMimic.|   |  |  | [Click]() | [Click]() |

# 12. Reproducibility, Analysis, and Critique
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Benchmarking Deep Reinforcement Learning for Continuous Control, Duan et al, 2016. Contribution: rllab. |   |  |  | [Click]() | [Click]() |
| # :question: | Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control, Islam et al, 2017. |   |  |  | [Click]() | [Click]() |
| # :question: | Deep Reinforcement Learning that Matters, Henderson et al, 2017.
| # :question: | Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods, Henderson et al, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?, Ilyas et al, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Simple Random Search Provides a Competitive Approach to Reinforcement Learning, Mania et al, 2018. |   |  |  | [Click]() | [Click]() |
| # :question: | Benchmarking Model-Based Reinforcement Learning, Wang et al, 2019. |   |  |  | [Click]() | [Click]() |

# 13. Multi-Agent RL
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :smile: | Social Influence as Intrinsic Motivation for MA-DRL, Jaques et al. | 2019 | Social Influence | MOA | [Click](https://arxiv.org/abs/1810.08647) | [Click](13_MARL/2019_Jaques.md) |

# 14. KL-Regularized RL

| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :smile: | Information Asymmetry in KL-Regularized RL, Galashov et al. | 2019 | Learned Priors | - | [Click](https://arxiv.org/abs/1905.01240) | [Click](14_KL_reg_RL/2019_Galashov.md) |
| # :smile: | Neural Probabilistic Motor Primitives, Merel et al. | 2019 | Few-Shot Transfer | NPMP | [Click](https://arxiv.org/abs/1811.11711) | [Click](14_KL_reg_RL/2019_Merel.md) |

# 15. Bonus: Classic Papers in RL Theory or Review
| Read / Notes  | Title  & Author  | Year  | Category | Algorithm | Paper  |  Notes |
| ------ |:-------------:|  :-----:| :-----:|  :-----:| :-----:|:-----:|
| # :question: | Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000. Contributions: Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes. |   |  |  | [Click]() | [Click]() |
| # :question: | An Analysis of Temporal-Difference Learning with Function Approximation, Tsitsiklis and Van Roy, 1997. Contributions: Variety of convergence results and counter-examples for value-learning methods in RL. |   |  |  | [Click]() | [Click]() |
| # :question: | Reinforcement Learning of Motor Skills with Policy Gradients, Peters and Schaal, 2008. Contributions: Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods. |   |  |  | [Click]() | [Click]() |
| # :question: | Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford, 2002. Contributions: Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms. |   |  |  | [Click]() | [Click]() |
| # :question: | A Natural Policy Gradient, Kakade, 2002. Contributions: Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL. |   |  |  | [Click]() | [Click]() |
| # :question: | Algorithms for Reinforcement Learning, Szepesvari, 2009. Contributions: Unbeatable reference on RL before deep RL, containing foundations and theoretical background. |   |  |  | [Click]() | [Click]() |
