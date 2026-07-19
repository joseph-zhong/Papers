# Recurrent Networks and LSTMs

- 17 Aug. 2017
- UWCSE Deep Learning in Practice Group
- Presented by Connor Schenck

##  Overview

Definition: Network designed to be called or evaluated multiple times

### Normal NN (DAG)

Directed and Acyclic workflow

### Recurrent NN (Non-DAG)

Recurrence in the workflow from Input -> Blackbox -> Output

## Types of RNNs

- Timeseries
- Data Recurrent Networks
  - Equilibrium Neural Networks



## Time Series RNNs 

- Core Idea: Utilize notion of memory across different states in "time-series" 
- Example: Object Detection

### Object Detection with RNNs

State0: Detect an object

State1-N: Pass the previous output "memory" as **additional information**

#### Training RNNs

- Naive method: Must be gradients "upwards" and "leftwards", to backprop both the object detector along with the memory saver
- This causes exploding or vanishing gradient with many layers and timesteps

## LSTM: Long Short-Term Memory Unit

- Unit to combat exploding and vanishing gradient
- Regular NN: Conv x5 -> FC 
- LSTM: Conv x5 -> LSTM -> FC

### Core LSTM Components

![LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

- Four gates which are actually separate layers (Conv or FC, ...)
  - Cell State
  - Forget Gate (output to a `sigmoid`, `0` for forget, `1` to remember)
  - Input Gate (output to a `sigmoid`)
  - Block Input (output to a `tanh`)
  - Output Gate (output to a `sigmoid`)
- What people usually do
  - Add Recurrent State (another cell state) to **concat** to the input
  - Add Peepholes (Additional Recurrent States) to **concat** to different gates

### Gated Recurrent Units (GRU)

![GRU](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png)

- Simpler than LSTMs
  - Update Gate: `1-sigmoid` to add to memory
  - Reset Gate: ...

### Example in Research: Detecting and Tracking Water

- "Can we track how much water is in a bowl given that we can watch the water poured into it?"
- Ground Truth is from Water Simulator
  - Real-life works as well with a infrared camera with hot water

#### Neural Network Architecture

- LSTM uses Fully Convolutional 1x1 Kernels
  - Could use Fully Connected usually instead


- Takes 3 Recurrent States
  - 3x3 Conv x5 -> (20 1x1 kernels LSTM) -> 1x1 Conv -> Deconv -> Output
  - 5x5 Conv x3 -> ...

### Analysis

- CNN: Blob of Water Detection
- MultiFrame-CNN: Slightly Sharper
- LSTM-CNN: Very Sharp water detection
  - Intuitively, we know that water interacts given gravity and it moves in a certain way

### RE3 - Tracking Arbitrary Objects 

- Recent work by Daniel Gordon
  - Paper: https://arxiv.org/pdf/1705.06368.pdf



