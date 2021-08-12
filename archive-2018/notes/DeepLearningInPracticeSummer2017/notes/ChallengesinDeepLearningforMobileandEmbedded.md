# Challenges in Deep Learning for Mobile and Embedded

- 6 July 2017
- UW CSE Deep Learning in Practice Group Meeting
- Presenter: Carlo C. del Mundo 
- carlo@xnor.ai

## Demos

- iPhone running on CPU with realtime obj. detection
- Raspberry Pi Zero running "Person recognizer"

## Deployment

### Machines for Deep Learning

- NVIDIA Jetson TX2
  - GPU:
    - 256 CUDA Cores
    - 8GB @59.7GB/s
  - CPU: 
  - ARM Quadcore
- Apple iPhone 
  - RAM: 2gb
  - One of the best ARM processors
- Raspberry Pi Zero ($5)

### Considerations in a Platform

- Multicore? Parallel is better
- Vector units? Vector instructions?
  - SIMD Mode: Multiple adds in parallel
- Mobile GPU available?
  - iPhone has 400GFlops

#### Special Instructions

- `dp4a`: Four-way byte dot product-accumulate
- Quantized floating point makes linear difference

## Which Deep Learning Frameworks?

- The key is the frameworks are for training
  - Built on cuDNN
  - Frameworks are built on the same backend

### Deep Leraning Building Blocks

- Platform-specific:
  - NVIDIA: cuDNN
  - AMD: MIOpen
  - Apple: Accelerate + CoreML
  ...

### Amdahl's Law

- Overall speedup dependent on two factors:
  - Percent of time the task consumes
  - Factor of speedup 
- DNN spends 70% of time in convolutions and 30% in everything else
  - Delete the convolutions, yet you only have 3x speedup

## Training and Deployment Strategies 

- Significant differences between training and deployment time
- Facebook trains with PyTorch, deploys with Caffe2

### Separation between Inference vs Training

- Deployment must be specialized for inference 
- Optimizations
  - Image Acquisition: Acquire data from sensor (e.g. camera)
  - RGB to Float conversion: `uint8_t` to `float`
  - Data Marshaling: Transpose data to data layout for back-end network
  - Inference: Forward pass
  - Decode: Application specific interpretation of output
  - Draw: Bounding Boxes




