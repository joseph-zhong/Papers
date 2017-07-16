# Detection/Segmentation Notes

- 15 July 2017
- Xevo Summer

## Papers to Review

### General

- [ ] SqueezeNet (Feb. 2016)
- [ ] MobileNet (Jun. 2017)
- [ ] ResNet (Dec. 2015)

### Object Detection

- [ ] SSD (Dec. 2016) 
- [ ] RCNN (Nov. 2013)
- [ ] Fast RCNN (Sept. 2015)
- [ ] Faster RCNN (Jan. 2016)
- [ ] R-FCN (May 2016)

### Semantic Segmentation

- [ ] Deconvolution Networks (May 2015)
- [x] FCN (May 2016)
- [ ] DeepMask (Jun. 2015)
- [ ] SharpMask (Mar. 2016)
- [ ] MaskRCNN (Apr. 2017)
- [ ] BoxSup (May 2015)
- [ ] DeepLab (June 2016)
- [ ] Adelaide (Apr. 2015)
- [ ] Deep Parsing Network (Sept. 2015)

## Reviews

### FCN

- Paper: https://arxiv.org/pdf/1605.06211.pdf
- Fully Convolution Networks without Unpooling and upsampling earlier improves accuracy
- Structure: 
  - Convolution/Pool layers -> Conv1x1 -> Deconv 
- Results: VOC2012
  - MeanIU: 62.2

#### Structure 

- [`FCN.md`](Review/FCN.md)

