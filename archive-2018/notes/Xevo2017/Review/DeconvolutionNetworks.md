# Deconvolution Networks

## Overview

- Paper published May 2015: https://arxiv.org/abs/1505.04366
- Upsampling VGG16 via Deconvolution/Unpooling layers to produce segmentation masks
  - Expands on deconv idea from FCN by training deconv layers rather than using bilinear interpolation
- Structure:
  - Conv/Pool -> FC -> Deconv/Unpool
- Results: VOC2012
  - DeconvNet MeanIU: 69.6 
  - DeconvNet + CRF MeanIU: 70.5
  - EDeconvNet MeanIU: 71.7
  - EDeconvNet + CRF MeanIU: **72.5**

