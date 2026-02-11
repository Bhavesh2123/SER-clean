# Speech Emotion Recognition using Convolutional Neural Networks

## Project Overview

This project implements a Speech Emotion Recognition (SER) system using deep learning. The goal is to classify speech recordings into different emotional categories by extracting meaningful acoustic features and training a Convolutional Neural Network (CNN).

The model learns patterns from MFCC (Mel Frequency Cepstral Coefficient) representations of audio signals and predicts the underlying emotion expressed in speech.

This project focuses not only on model training but also on proper feature engineering, regularization, performance evaluation, and visualization of results.

---

## Objectives

* Extract meaningful time-frequency features from raw speech signals
* Build a CNN-based classifier for emotion recognition
* Apply audio augmentation to improve generalization
* Evaluate performance using learning curves and confusion matrix analysis
* Structure the project in a modular and reusable way

---

The project follows a modular design so that feature extraction, modeling, and configuration are separated cleanly.

---

## Feature Engineering

### Audio Processing

* Sample Rate: 22050 Hz
* Duration: 3 seconds
* MFCC Coefficients: 40
* Fixed frame length: 130

Each audio file is:

1. Loaded using Librosa
2. Converted into MFCC features
3. Padded or truncated to maintain consistent input size

MFCC features are treated as 2D feature maps, similar to grayscale images, allowing CNNs to capture local time-frequency patterns.

---

## Data Augmentation

To improve robustness and reduce overfitting, the following augmentations were applied to training data:

* Gaussian noise injection
* Time stretching
* Pitch shifting

These augmentations simulate real-world variations such as recording noise and speaker differences.

---

## Model Architecture

The model is a Convolutional Neural Network composed of:

* 3 Convolutional blocks:

  * Conv2D
  * Batch Normalization
  * MaxPooling
  * Dropout
* Flatten layer
* Fully connected Dense layers
* Softmax output layer (8 emotion classes)

### Why CNN?

MFCCs are structured time-frequency representations. CNNs are effective at extracting local spatial patterns, making them suitable for modeling spectral variations in speech.

---

## Training Configuration

* Optimizer: Adam (learning rate = 0.0001)
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy
* Regularization:

  * Dropout
  * Batch Normalization

---

## Evaluation & Analysis

Model performance is evaluated using:

### 1️ Accuracy Tracking

Training and validation accuracy are monitored per epoch.

### 2️ Learning Curves

Loss and accuracy curves are plotted to observe convergence behavior and detect overfitting.

### 3️ Confusion Matrix

A confusion matrix is used to analyze class-level performance and identify commonly misclassified emotions.

This helps in understanding which emotions are acoustically similar and challenging to distinguish.

---

## Results

The model successfully learns discriminative acoustic features and achieves stable convergence. Validation curves remain close to training curves, indicating effective regularization.

Emotion-level analysis through confusion matrices reveals overlapping classes, particularly among acoustically similar emotions.

---

## Future Improvements

* Replace MFCC with log-Mel spectrograms
* Add LSTM layers (CNN + RNN hybrid architecture)
* Experiment with attention mechanisms
* Hyperparameter tuning
* Class imbalance handling
* Deploy as a Streamlit or Flask web app

---

## Key Learnings

* Importance of consistent feature dimensions in audio modeling
* Impact of augmentation on generalization
* How CNNs can model time-frequency representations
* Interpreting learning curves and confusion matrices
* Structuring ML projects cleanly for reproducibility

---

## Conclusion

This project demonstrates the end-to-end pipeline of a speech-based deep learning system — from feature extraction to model evaluation. It reflects a strong understanding of audio preprocessing, neural network architecture design, and performance analysis.

---

If you want, I can now:

* Make it **more technical (for ML engineer roles)**
* Make it **cleaner and shorter (for recruiters)**
* Or help you turn this into a GitHub portfolio project that stands out visually**

What role are you targeting — ML Engineer, Data Scientist, or AI Research Intern?
