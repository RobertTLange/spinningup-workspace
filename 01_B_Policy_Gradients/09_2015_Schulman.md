# Title: High-dimensional continuous control using Generalized Advantage Estimation

# Author: Schulman et al (2016) (OpenAI, Berkeley)

#### General Content:

Authors introduce general framework to estimate the advantage fct in a exponentially-weighted fashion similar to TD(lambda). They treat the discount factor as a variance reduction parameter at the cost of introducing some bias. Their advantage estimator significantly improves sample efficiency in combination with trust region methods. Show SOTA performance on MuJoCo.


#### Key take-away for own work:

RL problem is not too far away from supervised learning. Here the authors apply intuition from classical bias + variance decomposition and regularization. Also the discount factor has many different interpretations. E.g. as survival rate - think about Economics context.


#### Keypoints:

* Intuition: Variance of grad estimator scales badly with time horizon - effect of action confounded with effects of past and future actions. Here: introduce little bias to significantly reduce variance for sample efficiency.

* View discount not as part of optimization problem - return. But as part of value function estimation - role of variance reduction.
    * Reduce variance induced by future rewards at cost of introducing bias.

* gamma-just advantage estimator: No additional bias when estimating discounted advantage
    * Sufficient condition: decomposition in difference of two terms
        a. estimator of gamma-discounted state-action value
        b. arbitrary baseline function

* Advantage function estimation: k-step estimator.
    * Bias smaller as k to infinity - heavier discount of remaining estimator + baseline does not affect the bias
    * GAE: Exponentially-weighted average of k-step estimators
    * Connection to TD(lambda) - TD is estimator of value while here advantage

* Two corner cases:
    * lambda = 0: high bias, low variance
    * lambda = 1: low bias, high variance
    * role of gamma: determines scale of value function which is independent of lambda

* Reward shaping intuition: lambda as additional discount applied after performing reward shaping transformation
    * Shaping leaves policy gradient unchanged
    * Specific reshaping with value fct + gamma*lambda reward discount gives GAE
    * Derive GAE in terms of response function - advantage across timesteps
    * Interpretation: Reshape rewards with V to shrink temporal extend of response fct and introduce steeper discount to cut off noise from long delays

* Experiments:
    * Use trust region and conjugate gradient approach for both value fct and policy
    * Give details on how they come up with Hessian approx/Fisher info matrix
    * First update the policy and then the value function to avoid additional bias from overfitting of value fct
    * MuJoCo locomotion tasks - Optimal ranges: lambda [0.92, 0.98] and gamma [0.96, 0.99]


#### Questions/Critical Observations:

* Distal reward problem Hull (1943) - read up on!
* Try out scheme of different lambda, gamma per stage of learning? Curriculum discount ;)
