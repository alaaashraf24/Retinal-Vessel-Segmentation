# Retinal Vessel Segmentation

A deep learning-based approach for automatic segmentation of blood vessels in retinal fundus images using an EfficientNetB7-UNET architecture.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)

## Overview

This project implements an advanced deep learning model for retinal vessel segmentation, a critical task in the diagnosis and treatment of various ophthalmological diseases. The system uses a modified U-NET architecture with an EfficientNetB7 encoder to accurately segment blood vessels from retinal fundus images.

Retinal vessel segmentation plays a vital role in the early detection and monitoring of conditions such as diabetic retinopathy, glaucoma, and age-related macular degeneration. Automatic segmentation tools help ophthalmologists make faster and more accurate diagnoses, potentially improving patient outcomes.

## Key Features

- **EfficientNetB7-UNET Architecture**: Combines the powerful feature extraction capabilities of EfficientNetB7 with the precise localization abilities of UNET.
- **High Accuracy**: Achieves state-of-the-art performance on standard retinal vessel segmentation benchmarks.
- **GPU Acceleration**: Optimized for training and inference on NVIDIA GPUs.
- **Pre-trained Models**: Includes pre-trained weights for immediate use.
- **Data Augmentation**: Implements various augmentation techniques to improve model generalization.
- **Evaluation Metrics**: Provides comprehensive metrics including accuracy, precision, recall, F1-score, IoU, and AUC.
- **Visualization Tools**: Includes utilities to visualize the segmentation results and training progress.
  
## Model Architecture

The architecture combines EfficientNetB7 as an encoder (feature extractor) with a UNET-style decoder for precise segmentation:

1. **Encoder**: Pre-trained EfficientNetB7 backbone, which provides powerful feature extraction while being computationally efficient.
2. **Skip Connections**: Connect encoder layers to decoder layers, preserving spatial information.
3. **Decoder**: A series of upsampling blocks that gradually restore the spatial dimensions while decreasing feature channels.
4. **Output Layer**: A 1x1 convolution with sigmoid activation to produce the final segmentation mask.

The model's effectiveness comes from:
- Transfer learning using pre-trained weights
- Skip connections to preserve spatial information
- Attention mechanisms to focus on relevant features
- Custom loss function combining binary cross-entropy and Dice loss

## Dataset

This project uses the [Retina Segmentation Dataset](https://www.kaggle.com/datasets/alaaashraf24/retinasegmentation) available on Kaggle, which includes retinal fundus images and their corresponding vessel segmentation masks.

## Evaluation Metrics

The following metrics are used to evaluate the model:

- **Accuracy**: Overall pixel-wise accuracy
- **Sensitivity (Recall)**: Ability to detect vessel pixels
- **Specificity**: Ability to identify non-vessel pixels
- **F1-Score**: Harmonic mean of precision and recall
- **Area Under ROC Curve (AUC)**: Model's ability to distinguish between vessel and non-vessel pixels
- **Intersection over Union (IoU)**: Overlap between predicted and ground truth masks

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for training)
- TensorFlow 2.x
- OpenCV
- scikit-learn
- scikit-image
- pandas
- numpy
- matplotlib
- albumentations
