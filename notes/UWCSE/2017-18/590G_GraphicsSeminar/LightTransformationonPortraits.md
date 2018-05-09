# Portrait Lighting Transfer using a Mass Transport Approach

- CSE590G: Graphics Seminar
- Presented by James Noeckel
- 1 May 2018

Portrait lighting via Mass Transport (ACM Transactions on Graphics 2018): http://www3.cs.stonybrook.edu/~cvl/content/portrait-relighting/prl.html 

## Task: Light Transformation on Portraits

### Optimization

$$
\text{argmin}_{T_{i,j}} \sum_i \sum_j || c_i - c_j ||^2T_{ij}, \\ 
\text{s.t.}\\ 
T_{ij} \geq 0, \\
\sum_j T_{ij} = H_I(c_i),\\
\sum_i T_{ij} = H_R(c_j)
$$

$$
\text{argmin}_\hat{f} \sum_i || c_i - f(c_i)||^2 H_I(c_i), \\ 
\text{s.t.}\\
H_f(I) = H_R
$$

Intuition: We shouldn't ignore geometry of the face during general color transfer, as lighting affects different regions based on position and normal in addition to color
$$
\text{argmin}_\hat{f} \sum_i \hat{H_I}(s_i) (w_c || c_i - \hat{f_c}(s_i) ||^2 \\
+ w_p ||p_i - \hat{f_p}(s_i)||^2 \\
+ w_n||n_i - \hat{f_n}(s_i)||^2)
$$
for $f_c$ is color transformation, $f_p$ is for position, and $f_n$ is normalization (?)

## Algorithm

- Sliced Wasserstein distance: iterative approximate method for mass transport

