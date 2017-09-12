# PASCAL VOC

- Sept. 2017
- Xevo Summer
- Paper: The PASCAL Visual Object Classes (VOC) Challenge

## Overview

- Benchmark in Visual Object Category recognition and detection
- 2005-2012: Accepted as the standard benchmark in object detection

## Tasks

### Classification

For 20 object classes, predict presence/absence of objects in test images

### Detection

For each of the 20 object classes, predict bounding boxes for each object in test images (if any)

### Segmentation
Predict the object class of each pixel in test images, or "background" if the pixel doesn't correspond to any of the 20 object classes

#### Accuracy

A simple evaluation as the "percentage of pixels correctly predicted", can be misleading at times, since predicting a single class every pixel in a test image will result in 100% correctly predicted pixels for that class, despite 0% correct pixels for the other classes, and thus nonuniform class biases may cause misleading scores. While in 2007 this metric was used, in 2008, VOC switched to the segmentation accuracy, using the intersection of correctly predicted and labeled pixels, divided by the union with the incorrect predictions.

Formally, accuracy is calculated as

 $$acc_{seg} = \frac{Positive_{true}}{Positive_{true} + Positive_{false} + Negative_{false}}$$

### Person Layout

Detect people, and provide bounding boxes, and the presence/absence of head/hands/feet and the bounding boxes of those parts in test images

## Evaluation
Evaluation of results on multi-class datasets poses several problems

- The Forced Choice paradigm of asking "which *one* of $m$ classes does the image contain" adopted by Caltech-256 cannot be used for classification
- The distribution of classes is nonuniform, so simple accuracy metrics of *percent correct classifications* cannot be used, especially for detection tasks
- Metrics must be algorithm independent, since many past detection metrics made specific assumptions about models and model parameters

#### Average Precision (AP)

Average Precision summarizes the shape of the precision/recall curve, which semantically represents the ratio of "True Prediction" over "Total Predictions" for a constant set prediction confidence threshold. 

- Precision: Proportion of all examples above the threshold in t`he positive class
- Recall: Proportion of all positive examples ranked above a threshold

Typically, $AP$ is measured as the average Precision across eleven values of Recall: $r  \in \{0, 0.1, ..., 1 \}$

$$AP = \frac{1}{11} \sum_{r \in \{ 0, 0.1, ... 1\}} p_{interp}(r) $$

**Note**: At each recall level $r$, the precision is interpolated by taking the maximum precision measured for the corresponding recall which exceeds or equals $r$

 $$p_{interp}(r) = \max_{\tilde{r}: \tilde{r} \geq r} p(\tilde{r})$$

The intent of interpolation is to smooth the precision/recall curve from small variations 
