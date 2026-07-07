# Federated Medical AI for Multi-Disease Prediction

A privacy-preserving federated learning framework for multi-disease prediction using clinical data, ECG signal analysis, and medical imaging.

---

## Project Overview

This project focuses on building a scalable and privacy-aware healthcare AI framework capable of predicting multiple diseases without sharing sensitive patient data between institutions.

The system combines:

* Federated Learning
* Federated Model Distillation (FedMD)
* Deep Learning
* Multimodal Medical Analysis

to create a dual-stage diagnostic pipeline inspired by real-world clinical workflows.

The framework supports prediction and validation for:

* Chronic Kidney Disease (CKD)
* Diabetes Mellitus
* Heart Disease

---

## Key Features

* Privacy-preserving federated learning architecture
* Multi-disease screening system
* No raw patient data sharing
* Federated Model Distillation (FedMD)
* ECG-based heart disease validation using 1D-CNN
* Kidney ultrasound analysis using 2D-CNN
* Modular and scalable healthcare AI pipeline
* Multimodal diagnosis workflow

---

## System Workflow

### Stage 1 — Federated Disease Screening

Clinical datasets are distributed across multiple federated clients.

Each client independently trains disease-specific models using local data:

* CKD Client → Multi-Layer Perceptron (MLP)
* Diabetes Client → Logistic Regression
* Heart Disease Client → Gradient Boosting

Instead of sharing raw patient records, clients exchange only model logits using Federated Model Distillation (FedMD).

A global student model is trained to perform multi-label disease prediction.

---

### Stage 2 — Confirmatory Diagnosis

High-risk cases identified during Stage 1 are validated using disease-specific deep learning models.

#### Heart Disease Validation

* ECG signal analysis
* 1D Convolutional Neural Network (1D-CNN)

#### Kidney Disease Validation

* Ultrasound image analysis
* 2D Convolutional Neural Network (2D-CNN)

This staged workflow mimics real-world healthcare diagnosis pipelines.

---

## Tech Stack

### Programming Language

* Python

### Machine Learning & Deep Learning

* PyTorch 2.10.0
* Scikit-learn 1.8.0

### Software Environment

* Python 3.14.3
* PyTorch 2.10.0
* scikit-learn 1.8.0
* NumPy 2.4.1

### Models Used

* Multi-Layer Perceptron (MLP)
* Logistic Regression
* Gradient Boosting
* 1D-CNN
* 2D-CNN

### Libraries

* NumPy 2.4.1
* Pandas
* Matplotlib
* Seaborn

---

## Datasets Used

### Clinical Datasets

* Chronic Kidney Disease Dataset
* Diabetes Dataset
* Heart Disease Dataset

### Medical Signal & Imaging Datasets

* MIT-BIH Arrhythmia ECG Dataset
* Kidney Ultrasound Dataset

---

## Dataset Repository

Due to GitHub storage limitations, datasets are maintained separately.

Dataset Repository:
[https://github.com/avankrish/Multi-Disease-FL-Dataset-Archive]

---

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

---
## Reproducing Results

1. Train client models:
```bash
python training/train_ckd.py
python training/train_diabetes.py
python training/train_heart.py
```

2. Generate FedMD logits:
```bash
python fedmd/generate_logits_ckd.py
python fedmd/generate_logits_diabetes.py
python fedmd/generate_logits_heart.py
```

3. Aggregate logits and train student:
```bash
python fedmd/server_aggregate.py
python fedmd/train_student.py
```

4. Run Stage-1 evaluation:
```bash
python inference/run_stage1_test.py
```

5. Run ablation study:
```bash
python fedmd/run_5round_variance_test.py
```

6. Run Stage-2 ECG model:
```bash
python stage_2/heart_ecg/train.py
```

7. Run Stage-2 ultrasound model:
```bash
python stage_2/kidney_ultrasound/train.py
```

## Dataset Repository

Clinical datasets and public reference data:
https://github.com/avankrish/Multi-Disease-FL-Dataset-Archive
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

---

## Running the Project

Run the complete system:

```bash
python run_system.py
```

---

## Results

### Stage 1 — Federated Multi-Disease Screening

| Metric               | Score  |
| -------------------- | ------ |
| Exact Match Accuracy | 91.67% |
| Hamming Accuracy     | 97.22% |
| Micro F1-Score       | 0.952  |
| Macro F1-Score       | 0.947  |

### Stage 2 — Confirmatory Models

| Model             | Accuracy |
| ----------------- | -------- |
| ECG 1D-CNN        | 98.13%   |
| Ultrasound 2D-CNN | 100.00%* |

* Evaluated on the tested dataset.

---

## Project Highlights

* Developed a dual-stage federated healthcare AI pipeline
* Implemented privacy-preserving knowledge sharing using FedMD
* Combined structured clinical data with medical signals and imaging
* Designed a scalable architecture for future disease expansion
* Applied multimodal deep learning for confirmatory diagnosis

---

## Future Improvements

* Explainable AI (XAI)
* Differential Privacy integration
* Secure Multi-Party Computation (SMPC)
* Personalized healthcare recommendations
* Clinical decision support integration

---

## Maintainer

* Avanthika K


---

## Note

This repository represents a research-oriented healthcare AI project focused on federated learning and multimodal disease diagnosis.

---

## License

This project is intended for educational and research purposes.
