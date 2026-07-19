# Bayesian Hyperparameter Optimization

- 20 July 2017
- Presented by Aaron Jaech
- UW Deep Learning in Practice Summer 2017

## Why Auto Tuning Matters

- Humans are historically horrible at tuning hyperparameters
- DeepMind has shown that "old" Neural Language Models actually were better than state of the art models just via more hyperparameter tuning

### Quick Tips and Tricks

- Don't do grid search, it's popular but it's not correct in practice
  - Random Search is only ~3x slower, indicates very wasteful
- Explore full space 


- ​

### Bayesian Optimization

- Takes ideas from Bayesian Linear Regression
  - Multi-dimensional linear regression with Bayesian inference
- Define Acquisition Function
  - Goal is to explore "beyond what is the currently known best"
  - *Side note*: Perhaps better, or at least common would be to create your own Acquisition Function
- **Expected Improvement**: $EI = \mathbb{E}[max(\gamma - f(\theta), 0)]$
  - If new value is better, rewarded, no loss for worse performance
- ​

