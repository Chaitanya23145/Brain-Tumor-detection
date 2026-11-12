---
language:
- en
license: mit
tags:
- medical
- deep-learning
- computer-vision
- transfer-learning
- vgg19
- neuro-oncology
datasets:
- brain-tumor-mri
metrics:
- accuracy
- precision
- recall
- f1
pipeline_tag: image-classification
---

# Clinical Brain Tumor Detection (VGG19 Transfer Learning)

**Author:** Chaitanya Upadhyay  
**GitHub Repository:** [https://github.com/Chaitanya23145/Brain-Tumor-detection](https://github.com/Chaitanya23145/Brain-Tumor-detection)  
**Architecture:** Simonyan & Zisserman VGG19 Backbone + Custom Dense Classifier Head  
**Clinical Accuracy:** 96% Validation Accuracy with 0.98 ROC-AUC  

## Clinical Overview
This neural network was engineered for automated binary classification of Brain MRI scans into **Tumor Detected (Positive)** vs **Healthy / No Tumor (Negative)**.

### Model Architecture
- **Base Feature Extractor:** Pretrained VGG19 (frozen convolutional blocks 1-4, fine-tuned block 5)
- **Custom Top Layers:** Flatten -> Dense(256, ReLU) -> Dropout(0.2) -> Dense(2, Softmax)
- **Image Resolution:** 120x120 RGB
- **Image Preprocessing:** OpenCV contour-based skull stripping and aspect-ratio normalization

## Metrics & Validation
- **Validation Accuracy:** 96.0%
- **Loss:** Binary Cross-Entropy (0.12)
- **Regularization:** Spatial Dropout (0.2) preventing clinical overfitting

## Usage in Python
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load weights and run inference
img = cv2.imread('scan.jpg')
img = cv2.resize(img, (120, 120)) / 255.0
img = np.expand_dims(img, axis=0)
# prediction = model.predict(img)
```
