# Title: Meta-learners' dynamics are unlike learners'

# Author: Neil Rabinowitz (DeepMind; 2019)

#### General Content:

* Applies analytics in the style of Saxxe et al. 2013 and Schaul et al 2019. to analyze the learning dynamics (mostly inner loop) of meta-learners. Unlike "shallow" learners (SGD only one loop - staggering task discovery) it appears that all "task modes" (singular value/fourier frequencies) converge at the same time. Rabinowitz argues that this is akin to Bayes' optimal inference.
* Importance for areas where the actual learning behaviour itself matters (safety, curriculum design, etc.)

#### Key take-away for own work:

* Dont need to have access to linear weight matrix to do Saxxe analysis - simply look at input-ouput mapping (x, y_hat) and compute SVD/FFT on the W_hat resulting from a linear regression! - Indirect only though!

#### Keypoints:

* Meta-learned LSTMs pursue different learning trajectories than SGD-based (not only faster). Here: Study linear regression (Saxxe, 2013), nonlinear regression (Rahaman 2018, Xu 2018) & contextual bandits (Schaul 2019)

* Analysis of ORDER OF TASK STRUCTURE PICKUP!
    * learning algo = domain-general + sample inefficient
    * LSTM meta-learner = domain-specific + sample efficient
    * Here: Give insights into how a learning system operates when it is not trained until convergence - finite resources - time limits!

* How can learning systems build abstraction layers? - outer optimiser can configure inner optimiser to pursue learning trajectories less avaible to a non-nested system

* Think of learning dynamics as a function of priors that a learner brings to the task

* GD = Local searh in hypothesis space - find better hypothesis than the current one. Intermediate points are viewed as obstacles

* Aprit 2017: Inputs with randomised labels are learned later than those with correct
* Achille 2019: Certain structures could only be efficiently leaned early in training - critical periods in biological learning (Hensch 2004)

* Idea that human and machines may show similar learning dynamics at the behavioural level! - especially at the representation level - indication for alignment!

* **Linear Regression Experiment** (Saxxe et al. 2013)
    * 2-layer MLP on Gaussian Lin Reg example - compute effective spectrum & ignore non-diag terms
    * Sigmoidal learning curves with quicker convergence for larger modes
    * Meta: Train LSTM to predict y_t given x_t and y_t-1 (20 time steps)
    * Assess dynamics by fitting a linear approximation to the forward pass again calculate effective spectra with the linear approx W_hat
    * In-distribution results: 95% of singular value cdf induced by sampling tasks - meta-learner learns roughly all singular modes at same rate
    * Out-of-distribution results: Meta-Learner fails to solve task - might be due to saturation of LSTM outside of natural operating regume - targets too large
    * Looks very much like Bayes optimal inference with correct matrix-normal prior over W
    * Results independent of optimiser - same qualitative results

* **Non-Linear Regression Experiment** (Rahaman et al., 2018 & Xu et al., 2018)
    * Low Fourier frequencies in target function are uncovered before higher frequencies
    * Task setup: Sample low-pass fourier spectrum as non-linear trafo & analyze by computing FFT Fourier coefficients on the output + complex inner product for normalised projection
    * Meta: Exactly same analysis as in linear case but with FFT on derived output
    * In-distribution results: Again learns almost all frequencies Simultaneously
    * Out-of-distribution results: Meta-Learner learned an effective prior over Fourier spectra - but suppresses structured outside of support of this distribution - bandpass functions trained on vs stop-band signals

* **Contextual-Bandit Experiment** (Schaul et al., 2019)
    * Interference effect: Improved performance in one context suppresses learning for other contexts - leads to plateus - unique to on-policy learning!
    * Comparison with decoupled system (one per context) - 5 actions and contexts
    * Learner: Linear func approx with softmax policy output
    * Assessment of learning dynamics - expected return as a function of time - double expectation (one over time, one over context) - probability of taking the correct action
    * Result: When learning is coupled across tasks learning takes exponentially longer
    * Meta: Feed in reward from t-1 and not correct y target
    * Bayes optimal inference evidence - no longer plateaus

* **Outer loop learning dynamics**:
    * Measure how inner learning trajectories progressed over the course of outer loop training. Problem: No clear pattern!
        * Nonlinear regression & contextual bandit: staggered learning
        * Linear: in meta-training the inner learning better able to estimate the lesser singular modes than the greater ones - staggered in the opposite direction
    * Cant compare inner and outer loop learning dynamics: One learns a general algorithm while the other learns a solution

#### Questions/Critical Observations:

* Is it possible to make amortized Bayes-optimal inference analogy more explicit?
* How can we make out-of-distribution behavior better?
* Can we take this back and construct a series/decomposition of tasks that is easier to learn?
* Connections to pedagogy: teacher needs to have a rich model of what students are supposed to learn - Shafto et al., 2014; Ho et al. 2018
    * Communication vs Reinforcement signal
