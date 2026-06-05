# Privacy-Preserving Federated Learning Framework for Multi-Disease Prediction

A scalable and privacy-preserving healthcare prediction framework using Federated Learning and deep learning-based multimodal diagnosis.

## Overview

This project proposes a dual-stage intelligent healthcare framework for predicting multiple diseases while preserving patient privacy. The system combines Federated Model Distillation (FedMD) with deep learning-based medical signal and image analysis to support secure and scalable medical diagnosis.

The framework performs:

* Stage-1: Federated multi-disease screening using structured clinical data
* Stage-2: Confirmatory diagnosis using ECG signal analysis and ultrasound imaging

The system supports prediction and validation for:

* Chronic Kidney Disease (CKD)
* Diabetes Mellitus
* Heart Disease

## Key Features

* Privacy-preserving federated learning architecture
* Multi-disease prediction using FedMD
* No raw patient data sharing
* Multimodal diagnosis pipeline
* ECG-based heart disease validation using 1D-CNN
* Kidney ultrasound analysis using 2D-CNN
* Scalable modular architecture for adding new disease domains
* Clinically inspired staged diagnosis workflow

## System Architecture

The proposed framework follows a dual-stage diagnostic pipeline:

### Stage-1: Federated Screening

* Local clients train disease-specific models independently
* Clients share only logits on a public reference dataset
* FedMD aggregates knowledge without sharing raw data
* A global student model performs multi-label disease screening

### Stage-2: Confirmatory Diagnosis

* ECG signal analysis validates heart disease predictions
* Ultrasound imaging validates kidney abnormalities
* Triggered only for high-risk cases identified in Stage-1

## Technologies Used

### Programming Languages

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* Scikit-learn

### Models Used

* Multi-Layer Perceptron (MLP)
* Logistic Regression
* Gradient Boosting
* 1D Convolutional Neural Network (1D-CNN)
* 2D Convolutional Neural Network (2D-CNN)

### Data Processing & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

## Dataset Information

The project utilizes multiple clinical and medical datasets for disease prediction and confirmatory diagnosis.

### Stage-1 Clinical Datasets

* Chronic Kidney Disease Dataset
* Diabetes Dataset
* Heart Disease Dataset

### Stage-2 Datasets

* MIT-BIH Arrhythmia ECG Dataset
* Kidney Ultrasound Dataset

## Dataset Repository

Due to GitHub file size limitations, the datasets are maintained separately.

Dataset Repository:
[https://github.com/avankrish/Multi-Disease-FL-Dataset-Archive.git]

## Performance Results

### Stage-1 Federated Screening

| Metric               | Score  |
| -------------------- | ------ |
| Exact Match Accuracy | 91.67% |
| Hamming Accuracy     | 97.22% |
| Micro F1-Score       | 0.952  |
| Macro F1-Score       | 0.947  |

### Stage-2 Confirmatory Models

| Model             | Accuracy |
| ----------------- | -------- |
| ECG 1D-CNN        | 98.13%   |
| Ultrasound 2D-CNN | 100.00%* |

* Evaluated on the tested dataset.

## Project Structure

```text
federated_medical_ai/
│
├── data/
├── evaluation/
├── feature_engineering/
├── fedmd/
├── inference/
├── models/
├── notebooks/
├── stage_2/
├── training/
├── requirements.txt
└── run_system.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/avankrish/federated_medical_ai.git
```

Move into the project directory:

```bash
cd federated_medical_ai
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the complete framework:

```bash
python run_system.py
```

## Research Contribution

This project introduces:

* A dual-stage federated healthcare framework
* Privacy-preserving multi-disease prediction
* Multimodal confirmatory diagnosis
* A scalable and modular medical AI pipeline

## Future Improvements

* Explainable AI (XAI) integration
* Differential Privacy (DP)
* Secure Multi-Party Computation (SMPC)
* Personalized medical recommendations
* Clinical decision support integration

## Authors

* Avanthika K
* Sivabalakrishnan M
* Haripriya Yogambaram

## Reference

Based on the research paper:

"Privacy Preserving and Scalable Federated Learning Framework for Multi-Disease Prediction with Staged Medical Diagnosis"

## License

This project is intended for research and educational purposes.
