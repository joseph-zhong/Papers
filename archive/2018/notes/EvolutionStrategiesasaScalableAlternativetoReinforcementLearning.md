[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

- 4 April 2017
- Professor Ali Farhadi/Dieter Fox Group Meeting

Domain:
  Markov Decision Process

$\pi(s, a; \theta)$
  s: state
  a: action
  $\theta$: policy parameter (given states and action, above is probability action to take in that state)
  r: reward: updates policy parameters based on results of "entire rollout"
    - instant reward vs global rewards

Is Evolution Strategies = Biologically Plausible Mathematical Model of Population Dynamics?

Well not really

```
Algorithm 1:
  Input:
    Learning rate $\alpha$
    Noise std. dev $\sigma$
    Initial policy parameters $\theta_0$
  For t = 0, 1, 2 ...
    Sample $\epsilon_1, ... \epsilon_n ~ \mathcal{N}(0, I)$
    Compute returns $F_i = F(\theta_t + \sigma \epsilon_i) for i = 1, ..., n$
    Set $\theta_{t+1} = \theta_t + \alpha \frac{1}{n\sigma} \sum^n_{i=1} F_i \epsilon_i$
```

Key contribution:

- Scalability/Parallelizability
  - Benefit of more threads is higher for this over...
- Better Exploration (Empirically)
- No Backpropagation needed!!

Cons:

- Worse Sample Complexity (3-10x)



