# Mobile Game In-App Purchase Prediction (CNN-GRU-RNN Hybrid Model)

## Overview
This repository contains a Jupyter Notebook (`lab9.ipynb`) that implements an end-to-end machine learning pipeline to predict whether a mobile game user will make an in-app purchase within 7 days (`purchased_within_7d`). The project leverages a **CNN-GRU-RNN hybrid neural network architecture** trained on sequential user activity data.

## Dataset
The model is trained on the `mobile_game_inapp_purchases.csv` dataset, which consists of **3,024 user records** across **13 original features**:
* **Demographics & Device:** `UserID`, `Age`, `Gender`, `Country`, `Device`
* **Gameplay Metrics:** `GameGenre`, `SessionCount`, `AverageSessionLength`
* **Monetization & Purchase History:** `SpendingSegment`, `InAppPurchaseAmount`, `FirstPurchaseDaysAfterInstall`, `PaymentMethod`, `LastPurchaseDate`

## Pipeline & Methodology
1. **Data Cleaning & Imputation:**
   * Numerical missing values (`Age`) are imputed using the median.
   * Categorical missing values (`Gender`, `Country`, `Device`, `GameGenre`) are imputed using the mode.
   * Purchase-specific missing values are imputed with domain-specific defaults (`0.0` for `InAppPurchaseAmount`, `-1` for `FirstPurchaseDaysAfterInstall`, `'No Payment'` for `PaymentMethod`, and `'1900-01-01'` for `LastPurchaseDate`) to explicitly represent non-purchasing users.
2. **Feature Engineering & Preprocessing:**
   * One-hot encoding applied to categorical features (`Gender`, `Country`, `Device`, `GameGenre`, `SpendingSegment`, `PaymentMethod`), expanding the dataset to 59 features.
   * Standardized all column names to lowercase snake_case (`lowercase_with_underscores`).
   * Engineered target variable (`purchased_within_7d`) and formatted user activity into sequential structures suitable for deep learning.
   * Applied feature scaling and train/validation/test splitting.
3. **Model Architecture:**
   * Built a **CNN-GRU-RNN hybrid model** combining 1D Convolutional layers (for local feature extraction) with Gated Recurrent Units (GRU) and Recurrent Neural Network (RNN) layers (to capture temporal dependencies in user behavior).
4. **Training & Evaluation:**
   * Trained the hybrid network and evaluated its predictive performance using standard classification metrics (Accuracy, Precision, Recall, F1-Score, and ROC-AUC).

## Tech Stack & Dependencies
To run this notebook locally or in Google Colab, ensure you have the following Python libraries installed:
* `pandas`
* `numpy`
* `scikit-learn`
* `tensorflow` / `keras`
* `matplotlib` / `seaborn`

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
