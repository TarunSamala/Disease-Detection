# Disease Detection in Leaves and Crops

## Introduction
This project focuses on detecting diseases in leaves and crops using deep learning techniques. It includes a Convolutional Neural Network (CNN) model trained on a dataset containing images of healthy leaves/crops and leaves/crops affected by various diseases. The model can classify a given image of a leaf/crop as either healthy or diseased, aiding farmers in early disease detection and management.

## Cloning the Code
To clone the code repository to your local files, use the following command:
```bash
git clone https://github.com/TarunSamala/Disease-Detection.git
```
## Requirements

Ensure you have the following dependencies installed:

- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- cv2

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Directory Structure 

- `Data Analysis/`: Contains Notebook for Visualization and Prediction
- `Scripts/`: Contains the entire model and python script.
- `PlantVillage/` : Contains dataset of the requrired leaves
- `output/`: Contains the model, However the github upload limit exists , it only contains in the admins directory not on github.


## Model Overview
The disease detection model is a Convolutional Neural Network (CNN) designed for classifying tomato leaf images into healthy and diseased categories. Here's a concise summary:

- Architecture: The CNN consists of three convolutional layers followed by max-pooling layers for feature extraction, followed by dense layers for classification.
- Training: Trained on a dataset containing images of healthy tomato leaves and leaves affected by leaf mold using the binary cross-entropy loss and Adam optimizer.
- Evaluation: Evaluated on a separate test set, computing metrics like accuracy, precision, recall, and F1-score. A confusion matrix visualizes classification results.
- Deployment: Deployable for real-world applications in crop disease monitoring, facilitating early detection and management of leaf diseases in tomato plants.

## Dataset
- Classes: Healthy and Leaf Mold
- Size: Sufficient samples for effective model training and evaluation
- Preprocessing: Standardized to 128x128 pixels, pixel normalization
- Split: Typical 80% training, 20% testing split
- Source: PlantVillage dataset
- Usage: Training and evaluating disease detection model
- Availability: PlantVillage or similar agricultural repositories

## Data Analysis and Visualization

- Accuracy: Measures overall model performance, indicating the proportion of correctly classified samples.
- Confusion Matrix: Provides detailed insights into classification results, helping identify misclassifications and model strengths.
