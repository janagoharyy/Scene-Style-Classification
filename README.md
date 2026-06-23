# Indoor Scene Style Classification (17 Classes)

## Overview
This project focuses on classifying indoor scene images into **17 interior design styles** using deep learning. The goal is to build and compare multiple CNN and transformer-based models to identify the most effective architecture for fine-grained visual classification.
We experimented with several modern vision models and analyzed their performance under the same dataset and training setup.


## Problem Statement
Given an indoor image, the model predicts its corresponding interior design style from 17 possible classes based on visual patterns such as furniture, lighting, textures, and layout.


## Dataset
The dataset consists of labeled indoor scene images across **17 categories**.
Total images: ~12,400

### Classes
asian, boho, coastal, contemporary, craftsman, eclectic, farmhouse, french-country, industrial, mediterranean, minimalist, modern, scandinavian, shabby-chic-style, southwestern, tropical, victorian

### Split
- Training: 80%
- Validation: 20%

### Key Characteristics
- Varied lighting conditions
- Different furniture styles and room layouts
- High intra-class similarity (challenging fine-grained classification task)


## Data Preprocessing
- Image resizing to **224×224**
- Normalization using ImageNet statistics
- Train/validation split (80/20)
- Tensor conversion for PyTorch models

### Data Augmentation
- Random horizontal flip
- Random rotation
- Color jitter
- RandAugment
- Test-time augmentation (TTA)


## Models Used
We evaluated multiple deep learning architectures:

- EVA02-L-14
- DINOv2-L
- Swin Transformer (Swin-B)
- ConvNeXt
- CLIP (ViT-L-14)
- EfficientNet-B3
- DenseNet
- MobileNetV2
- ResNet50
- Vision Transformer (ViT)
- VGG16


## Best Performing Model
### EVA02-L-14

EVA02-L-14 achieved the best performance among all tested models.

- Validation Accuracy: **47.8%**
- Input Size: 224×224
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- Epochs: 4
- Batch Size: 16

### Model Details
- Uses pretrained weights from large-scale vision-language pretraining
- We use only the **visual encoder**
- Custom classification head added:
  - BatchNorm
  - ReLU
  - Dropout
  - Linear layer (768 → 17 classes)

This adaptation allows the model to map high-level visual features into interior design categories.


## Results Summary
This project was developed as part of a competition.

- Final Rank: **6th place out of 32 teams**
- EVA02-L-14: Best performance (48.47%)
- Transformer-based models generally outperformed CNNs
- Dataset proved challenging due to fine-grained class similarity
 The model performed competitively among multiple deep learning approaches submitted by other teams.

## Performance Note
The model achieved ~62% accuracy on the validation set during training, while the final competition submission accuracy was 48.47%.
This performance gap is likely due to dataset distribution shift between validation and test sets, indicating that the test set was more challenging and had slightly different visual characteristics.

## Tools & Libraries
- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- HuggingFace / pretrained vision models


## Key Takeaways
- Transformer-based models are stronger for fine-grained scene classification
- Data augmentation significantly improves generalization
- Class similarity is a major challenge in interior design classification


## Future Work
- Improve accuracy using ensemble models
- Use higher-resolution training (e.g., 384px)
- Apply contrastive learning or metric learning approaches
- Improve dataset balancing and augmentation strategies
