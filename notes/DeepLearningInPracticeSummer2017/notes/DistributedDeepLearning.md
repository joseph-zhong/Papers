# Distributed Deep Learning

- 17 Aug. 2017
- UWCSE Deep Learning in Practice Group
- Presented by Daniel Gordon

## Goals

- Minimize Training Time
- Maximize Throughout for large datasets
- Maximize Concurrency for large clusters
- Minimize Data Transfer
- Maximize Batch Size for unstable small batches
- Minimize Latency for deployment

## Maximizing Batch Size

- Core Idea: Multiple cards on a motherboard and split batches on each GPU
  - Taking ideas from [Large Minibatch SGD paper](https://arxiv.org/pdf/1706.02677.pdf) 
  - Mini-batch size: after a minibatch size of 8k, validation error begins to increase 
    - Basic argument: Increasing the minibatches allows you to decrease the number of epochs
- When minibatch size is multiplied by $k$, multiply learning rate by $k$
  - Except on initial epoch: gradually increase learning rate during 5 epochs
- Make sure loss function normalizes w.r.t total batch size
  - Easy and efficient method: Sum all gradients and scale at the end at the synchronize step
- Shuffle Data every epoch across all workers 
  - Important to prevent correlated minibatches
  - Shuffling may be more efficient than randomized indices since sequential read for large datasets is faster than random access
- Batch Normalization becomes more specialized
  - Read the paper for the details
  - Do Batch Norm per GPU, not per mini-batch **(???)**
- Weight Decay, Momentum and Learning Rate
  - Core Idea: Keep loss separate from learning rate
  - Weight Decay should not be scaled by the learning rate
  - Don't scale Momentum by learning rate
    -  Apply changes to momentum and decay after changing learning rate of loss
- Framework Choice
  - Distributed Tensorflow
  - Caffe2

### Results

- Time Per Epoch decreases linearly with increasing mini-batch size
- At smaller scales (~255 GPUs) locking doesn't take away too much time
  - Speedup scales linearly as number of GPUs increases

## Maximizing Throughput and Concurrency

- A3C-Style Approach

  - Data Shards (many) -> Model Replicas (many) -> Central Parameter Server
    - Model Replica computes $\delta w$ pushed to Central Server 
    - Central Server computes $w' = w - n * \delta w$

- Hogwild SGD Algorithm

  - See [Hogwild Paper](https://arxiv.org/abs/1106.5730)

  - ```
    Algorithm Hogwild! update for individual processors
    1: loop
      2: Sample e uniformly at random from E
      3: Read current state x_e and evaluate Ge(x)
      4: for v in e do x_v ← x_v − gamma*b_v^T * G_e(x)
    5: end loop
    ```

## Trade Offs

- Synchronous (Maximizing Batch Size): 
  - More Stable
  - Faster Convergence
- Asynchronous (Maximizing Throughput and Concurrency)
  - Not much theoretical backing for Asynchronous Hogwild SGD
  - Easier to implement correctly (no need to deal with locks or scaling gradients...)
  - Easier to Scale
  - Faster per Sample (more concurrent)

## Minimizing Latency

- Not really a distributed problem
- Really a Fast Hardware solution 
  - See Google's [TPU Paper](https://arxiv.org/pdf/1704.04760.pdf)

## Summary

- PyTorch/Caffe2 for Synchronous
- TensorFlow for Asynchronous