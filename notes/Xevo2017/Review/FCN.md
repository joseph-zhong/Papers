# FCN

## Overview 

- Paper: https://arxiv.org/pdf/1605.06211.pdf
- Fully Convolution Networks without Unpooling and upsampling earlier improves accuracy
- Structure: 
  - Convolution/Pool layers -> Conv1x1 -> Deconv 
- Results: VOC2012
  - MeanIU: 62.2

## Structure

```
# VGG16 Layers base CNN.
(1)  Conv(filters=64, kern=3x3, stride=1) 
(2)  Conv(filters=64, kern=3x3, stride=1)
Pool(max)
(3)  Conv(filters=128, kern=3x3, stride=1)
(4)  Conv(filters=128, kern=3x3, stride=1)
Pool(max)
(5)  Conv(filters=256, kern=3x3, stride=1)
(6)  Conv(filters=256, kern=3x3, stride=1)
(7)  Conv(filters=256, kern=3x3, stride=1)
Pool(max)
(8)  Conv(filters=512, kern=3x3, stride=1)
(9)  Conv(filters=512, kern=3x3, stride=1)
(10) Conv(filters=512, kern=3x3, stride=1)
Pool(max)
(11) Conv(filters=512, kern=3x3, stride=1)
(12) Conv(filters=512, kern=3x3, stride=1)
(13) Conv(filters=512, kern=3x3, stride=1)
Pool(max)

# FCN8.
# 1x1 Conv replace the (14-16) Fully Connected layers.
# Deconv follows after the 1x1 Conv. Followed by Hadamard sum.
# REVIEW josephz: Double-check the number of filters.
(14) Conv(filters=numClasses, kern=1x1, stride=1) 
Argmax()

(15) Deconv(shapeDst=Pool4, filters=numClasses, kern=4x4, stride=2)
(16) Conv(filters=numClasses, kern=1x1, stride=1)
(17) Hadamard(operation='sum')

(18) Deconv(shapeDst=Pool3, filters=numClasses, kern=4x4, stride=2)
(19) Conv(filters=numClasses, kern=1x1, stride=1)
(20) Hadamard(operation='sum')

(21) Deconv(shapeDst=inputImg, filters=numClasses, kern=16x16, stride=8)
Argmax()
```