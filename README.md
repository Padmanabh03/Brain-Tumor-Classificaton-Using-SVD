# Brain-Tumor-Classificaton-Using-SVD

## Requirements
- Python 3.x
- Libraries: scikit-learn, xgboost, numpy, pandas, matplotlib, seaborn

## Overview
This project focuses on the classification of brain tumors using machine learning techniques. It utilizes a dataset of MRI images to identify and classify brain tumors into categories such as glioma, meningioma, nontumor and pituitary.

## Dataset
The dataset comprises MRI brain images, which are preprocessed and transformed using Singular Value Decomposition (SVD) for feature extraction.

## Features
- The dataset is transformed into a lower-dimensional space using SVD, retaining significant features for classification.
- The features extracted from SVD are used to train machine learning models.

## Models
Several classifiers are used in this project:
- Logistic Regression
- Gradient Boosting
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

A voting classifier is also implemented to combine predictions from all individual models using a soft voting strategy.

## Evaluation
The models are evaluated using cross-validation techniques. Performance metrics such as accuracy, precision, recall, and F1 score are calculated to assess the effectiveness of each classifier.

## Usage
To run the project:
1. Load the dataset.
2. Preprocess the images and extract features using SVD.
3. Train individual models and the voting classifier.
4. Evaluate the models on a test set.
5. Show the confusion matrix for Random Forest.

## Acknowledgments
- This project was inspired by my studies in Math 620, where I was introduced to Singular Value Decomposition (SVD). The intrigue in understanding SVD's application in image classification motivated the development of this project.

