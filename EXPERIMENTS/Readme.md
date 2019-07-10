# Deep Reinforcement Learning Algorithm Replications
## Author: Robert Tjarko Lange - 2019

### General comments & Env Setup

In this repository I aim to replicate interesting papers from the list of Spinning Up in DRL from OpenAI. The main purpose is educational and I try to provide as much comments and support as possible. Feel free to contact me (robertlange0@gmail.com) if you have any questions!

#### TODOs:

* [ ] Setup cartpole continuous environment

### Reproducing the individual algorithms

* [x] DQN, Double DQN, Dueling DQN

```
python train_dqn.py -v -agent MLP-DQN
python train_dqn.py -v -agent DOUBLE
python train_dqn.py -v -agent MLP-Dueling-DQN
```

* [x] Vanilla Policy Gradient

```
python train_vpg.py -v
```


* [x] Advantage Actor-Critic

```
python train_a2c.py -v
```
