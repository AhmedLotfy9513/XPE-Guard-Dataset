# XPE-Guard Dataset and Code

This repository contains the dataset and the training script for our paper: "XPE-Guard: A Transparent and Family-Balanced Static Malware Detection Architecture via SMOTE and SHAP".

## Overview
Detecting malware using static analysis is safer and faster than dynamic execution. However, real-world datasets are usually imbalanced (many Trojans, but very few Ransomware or Spyware samples). Also, many deep learning models act as black boxes.

In this project, we extracted 55 structural features directly from PE headers using the `pefile` library. Then, we applied SMOTE to balance the dataset and used XGBoost with SHAP to get high accuracy while keeping the model explainable.

## Dataset
The dataset `XPE_Guard_Dataset.csv` contains 9,600 balanced samples:
- Benign: 4,800
- Trojans: 1,600
- Ransomware: 1,600
- Spyware: 1,600

The target variable is `label` (0 for benign, 1 for malicious).

## Models Compared
We tested different algorithms on the tabular PE data. XGBoost gave the best results compared to spatial/sequential models:
- XGBoost: 99.12% Accuracy
- CNN: 96.20%
- RNN: 94.80%
- KNN: 88.50%

## Requirements
To run the code, install the following packages:
pip install pandas xgboost shap matplotlib scikit-learn

## Usage
Simply run the python script to load the data, train the XGBoost model, print the classification report, and generate the SHAP plot.
