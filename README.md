# Campus Vision Model

This repository contains the `Campus Vision Model`, a deep learning project built with PyTorch, to classify buildings on the Mississippi State University campus based on images. This project utilizes a pre-trained ResNet50 model, fine-tuned on custom building data, and was trained with the assistance of RoboFlow for dataset preparation. The training and testing scripts are designed to facilitate ease of training, model evaluation, and predictions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
  - [Key Configurations](#key-configurations)
  - [Training with Mixed Precision](#training-with-mixed-precision)
- [Evaluation](#evaluation)
  - [Validation Process](#validation-process)
- [Inference](#inference)
  - [Predicting for a Single Image](#predicting-for-a-single-image)
  - [Batch Prediction for Folder](#batch-prediction-for-folder)
- [Plots](#plots)
  - [Plot Examples](#plot-examples)
- [Results](#results)

---

## Overview
The Campus Vision Model identifies specific buildings on the Mississippi State University campus. This model was built using PyTorch and leverages transfer learning with the ResNet50 architecture, which was fine-tuned on a custom dataset of building images. The project demonstrates the use of mixed precision for faster training on GPU and early stopping to prevent overfitting.
Our team (Team 2) consisted of the following members:
* Sydney Whitfield
* Mohnish Sao
* Shawn Butler

## Dataset
The dataset consists of labeled images of campus buildings, organized into folders for each building class. The dataset was processed and augmented using RoboFlow, which allowed for resizing, rotation, and color transformations to improve model robustness.

### Class Labels
The classes represent different buildings, with the following label-to-building mapping:

| Class Index | Building Name      |
|-------------|---------------------|
| 0           | Butler Hall        |
| 1           | Carpenter Hall     |
| 2           | Lee Hall           |
| 3           | McCain Hall        |
| 4           | McCool Hall        |
| 5           | Old Main           |
| 6           | Simrall Hall       |
| 7           | Student Union      |
| 8           | Swalm Hall         |
| 9           | Walker Hall        |

## Model Architecture
The model uses a ResNet50 architecture, pre-trained on the ImageNet dataset. The final fully connected layer is replaced with a new layer to match the number of classes in our dataset.

### Training Approach
- **Transformations**: We applied random horizontal flips, rotations, and color jittering to the training set for data augmentation.
- **Mixed Precision**: Enabled for faster training using the `torch.amp` library.
- **Early Stopping**: The training process monitors validation loss, halting training if no improvement is observed for a set number of epochs.
- **Metrics**: The model tracks accuracy, precision, recall, F1-score, and log loss on the validation set.

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Campus-Vision-Model.git
    cd Campus-Vision-Model
    ```

2. **Install dependencies**:
    ```bash
    pip install torch torchvision pillow roboflow matplotlib seaborn scikit-learn
    ```

3. **Download the dataset** (using RoboFlow):
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("debristech").project("campus-vision")
    dataset = project.version(1).download("folder")
    ```

4. **Set up directory structure**:
    - Ensure that the downloaded dataset is structured as:
      ```
      campus-vision-1/
          ├── train/
          └── valid/
      ```

## Training
The `finalcampusvisionmodel.py` script contains the training code. Adjust hyperparameters, such as batch size, learning rate, and the number of epochs, as needed.

### Key Configurations
- **Batch size**: 32
- **Learning rate**: 0.001
- **Image size**: 512x512 pixels
- **Number of epochs**: 300 (with early stopping)

### Training with Mixed Precision
The model uses automatic mixed precision (`torch.amp`) to speed up training on compatible hardware, like NVIDIA A100 GPUs.

To train the model, run:
```bash
python finalcampusvisionmodel.py
```
## Evaluation
After training, the model is evaluated on a validation set with the following metrics:

- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision**: Indicates the accuracy of positive predictions.
- **Recall**: Represents the model's ability to find all positive instances.
- **F1 Score**: A balanced measure between precision and recall.
- **Log Loss**: Measures the performance of the classification model based on probability estimates.

These metrics help in understanding the model's performance and reliability.

## Validation Process
Validation is performed after each epoch. The script also saves the best model based on the lowest validation loss, helping to avoid overfitting and ensuring the model's robustness on unseen data.

## Inference
The `campusvision_testing.py` script provides functions for loading the model and performing predictions. This script can be used to test the model on new data and obtain prediction results.

### Predicting for a Single Image
To predict a single image, use the following code snippet:

```python
from campusvision_testing import predict

predict(model, "path/to/image.jpg", class_to_building)
```

# Batch Prediction for Folder

To predict all images in a folder:

```python
from campusvision_testing import predict_folder
predict_folder(model, "/content/test_images", class_to_building)
```

# Plots

The `TrainPlot.py` script visualizes the training and validation loss and accuracy over epochs.

```bash
python TrainPlot.py
```

# Plot Examples

- **Train vs. Validation Loss**: Useful to understand model convergence and overfitting.

- **Train vs. Validation Accuracy**: Helps visualize the accuracy improvements across epochs.

# Results

After training, the best model achieved the following performance metrics:

- **Best Validation Accuracy**: ~97%
- **Best F1 Score**: ~0.97
- **Best Log Loss**: ~0.11

These metrics indicate that the model performs well in classifying the campus buildings, with high precision and low misclassification.
