# Improving Object Detection

- 20 July 2017
- Presented by Joe Redmon
- UW Deep Learning in Practice Summer 2017

## Prediction

- Historically we have directly predicted the bounding box coordinates
- It is easier to predict offsets from Anchor Boxes
  - Aspect ratios are pre-defined
- Dimension Cluster (From YoloNet)
  - K Means clustering determines which dimension clusters to use as anchor boxes

### Dimension Clusters

- +5% AP: Significant improvement for no more complexity
- Box Generation | # | AVG IOU
- Cluster SSE | 5 | 58.7
- Cluster IOU | 5 | 61.0
- Anchor Boxes | 9 | 60.9

## Multi-scale Training

- +1.5% mAP
- Resize images to train on randomly (every so often batches)
  - Smooth trade-off between FPS and mAP

## What Makes a Network Fast?

- FLOPs is not the only thing which matters
  - Larger networks are better for saturating GPU computation
  - However, efficiency is the goal: smaller, simpler layers better 
    - ResNet vs VGG, DarkNet vs YOLOv1
- Network | Top5 | FLOPs | GPU Speed
- VGG16 | 90.0 | 30.95Bn | 100FPS
- YOLOv1 | 90.8 | 8.52 Bn | 200FPS
- ResNet50 | 92.2 | 7.66 Bn | 90FPS
- DarkNet19 | 91.8 | 5.58 Bn | 200FPS

## Combining Large Datasets

- COCO: 100K Images, 80 Classes
- ImageNet: 1M Images, 1K classes
- Cannot simply concatenate datasets together (clashing classes)
- ImageNet classes come from WordNet
  - YOLO prunes and cleans WordNet into WordTree
    - Essentially a Minimum Spanning Tree of the classes
- COCO has general labels for classes, which can be greatly refined via the ImageNet labels
  - **Now we can jointly train on BOTH COCO and ImageNet**
  - Labels without ImageNet refinements simply have non-refined error
    - Hierarchal Softmax
      - Softmax over syblings in WordTree
      - For layers of single children, better is to merge it with its parent layer and softmax there
    - This actually allows possibilities for **Zero Shot Learning**