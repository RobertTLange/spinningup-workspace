# Title: Trust Region Policy Optimization

# Author: Schulman et al 2015 (Berley/OpenAI)

#### General Content:
Introduce a specific surrogate objective that guarantees policy improvement (with non-trivial step size!). Afterwards, they take this theoretical result and apply a series of approximations which yields TRPO. Authors provide two alternative variants: (1) Single-path: following a model-free rollout. (2) Vine: Requires agents to transfer back to previous states in env and can only be used in simulation. Show SOTA performance on both continuous control and discrete ATARI.

Intuition: By bounding the size of the policy update, TR methods bound changes in the state distribution and thereby guarantee improvements in the policy.


#### Key take-away for own work:
Need to study more monotonic improvement theory and should read Schulman's thesis. Also implement VPG before going on with reading.

#### Keypoints:

* Objective: Decompose exp return of new policy as exp return of old policy + exp advantage of new policy.    
    * Policy improvement: Every policy update that assures that expected advantage of new policy is nonnegative in every state is guaranteed to increase performance. - leads to policy iteration step with argmax over advantage as new policy.
    * BUT: In practice only access to approximate advantage estimate - there will be states for which exp advantage is negative => Hard to optimize directly!

* Instead local approx by taking expectation over old and not new policy transition probs! = Ignores changes in state visitation density due to changes in the policy.
    * But: Given continuous parametrized policy - then objectives match to first order!
    * Hence: For sufficiently small step sizes improving the exp wrt to the old policy will also improve the objective for the hypothetical exp under the new policy
    * Or: Need to restrict changes in policy to make estimate of exp advantage somewhat better!

* Builds heavily on Kakade & Langford (2002) "approx opt approx RL" work: Propose updating scheme for conservative iteration - show explicit lower bounds on improvement for specific mixtures of policies (here learning rate), discounts and expectation under new policy

* Here - theoretical: Generalize result to all stochastic policies and not only mixture - by replacing expectation by first a total variation divergence bound (proofs based on: 1 - coupling of random variables, 2 - perturbation theory), and then KL bound (due to quadratic bounding of total variation).
    * Still: Assumes that we are able to evaluate advantage exactly!
    * Resulting algo is part of minorization-maximization MM algo - still need to choose penalty coefficient scaling the divergence term & choosing according to theory results in small learning rule

* TRPO for parametrized policies - before general and under assumption of policy eval in all states
    * Instead of small learning rule use hard constraint on the KL divergence (i.e. a trust region constraint)
    * Problem: Constraint imposed that the KL div is bounded at every point in state space. Impractical to solve for large number of constraints. Instead: **Heuristic** - use average KL divergence

* Sample based estimation of objective and constraint:
    * MC samples - approx expectation together with importance sampled value estimates
    * Single path: Standard PG routine - sample individual trajectories
    * Vine: Construct rollout set and perform multiple actions from each state in rollout state. - Common Random Numbers to reduce Q value differences
    * Different variants lead to different importance sampling distributions. Vine leads to better est of advantage at cost of more calls to the simulator.
    * Optimization via conjugate gradient algo + line search - use Fisher info as in natural gradients and analytical computation of Hessian


#### Questions/Critical Observations:

* How does one choose the delta in the hard constraint?
