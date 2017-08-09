# Cycle Gans

- 3 Aug. 2017
- UWCSE Deep Learning in Practice, Summer
- Presented by Max Horton

## Cycle Consistency

- Prevent transfering any image turning into an arbitrary different styled image, goal is to achieve the exact same image, but in different style
- Loss becomes Linear Combination of style and similarity

### Formulation

...

### Evaluation

- AMZN MTurk: Flashed Real/Fake
- FCN Score
  - Predict a label for a generated photo
  - Must detect "Car on Road" from generated on photo
    - Compare Reak and Fake
    - ​

### Results

- Significantly better than previous baselines
- Pix2Pix still better on FCN Score

### Ablation

- GAN + Forward suffers from mode collapse, producing identical label maps regardless of input
- CycleGan did poorly at preserving color
  - Identity Mapping Loss 
    - L1 loss between input and mapped image $x- g(x)$ helps

