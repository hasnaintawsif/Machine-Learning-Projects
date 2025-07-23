# 🏦 Bank Customer Churn Prediction

##  Business Objective

A bank aims to identify customers who are likely to leave (churn) based on their profile and activity data. Early detection enables targeted retention strategies, such as personalized offers or customer service outreach, to reduce churn and maintain revenue stability.

##  Project Overview

This machine learning project develops a churn prediction system using structured customer data. The workflow includes:

* Exploratory Data Analysis (EDA)
* Feature engineering and preprocessing
* Model training with cross-validation
* Hyperparameter tuning using RandomizedSearchCV
* Model evaluation with multiple performance metrics

Several models were evaluated, and the best results were obtained using **XGBoost** after hyperparameter optimization.

##  Final Model Performance – Tuned XGBoost

| Metric    | Test Score |
| --------- | ---------- |
| Accuracy  | 0.9454     |
| Recall    | 0.9733     |
| Precision | 0.9621     |
| F1 Score  | 0.9677     |

Key Insight

The tuned XGBoost model achieves high overall accuracy and recall, making it highly effective at identifying customers who are likely to churn. This allows the bank to take timely preventive actions.



Here’s a polished and professional version of your **README** section for the **Spam Email Detection** project. It keeps your content intact but enhances clarity, structure, and formatting for better presentation:

---



---
# 📧 Spam Email Detection (Stacking Ensemble)

##  Project Objective

This project builds a **spam email classifier** using a **stacking ensemble** approach to accurately detect whether an email is spam or not.

##  Key Highlights

*  **95% test accuracy** achieved
*  Ensemble of base models:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
  * Extra Trees
  * Bagging
  * AdaBoost
  * Gradient Boosting
  * XGBoost
*  **Meta-model**: XGBoost
*  Trained models saved with `joblib` for deployment
*  Evaluated using test data and cross-validation

##  Download Trained Models

The trained model files exceed GitHub’s size limit.
 Download from:
[📁 Google Drive – Trained Models](https://drive.google.com/drive/folders/1XuF7MoJdLVEyBvEujw9af9zMsxijkk9I?dmr=1&ec=wgc-drive-globalnav-goto)

##  How to Predict on New Data

To make predictions on new input data:

```python
import joblib
import numpy as np
import pandas as pd

# Load new feature set
x_new = pd.read_csv('your_input.csv')

# Load models
base_models = joblib.load('base_models.pkl')
meta_model = joblib.load('spam_detection.pkl')

# Create blended feature set from base model predictions
blend = np.zeros((x_new.shape[0], len(base_models)))
for i, model in enumerate(base_models):
    blend[:, i] = model.predict_proba(x_new)[:, 1]

# Predict with meta-model
predictions = meta_model.predict(blend)
```

---
---
# 👥 Customer Segmentation Using KMeans Clustering

## Overview / Objective

In this project, I performed customer segmentation using KMeans clustering on a large transactional dataset containing over 1 million records with 5 key engineered features. The objective was to identify meaningful customer groups based on transaction behavior and demographics.

## Data Preparation and Feature Engineering

- Extracted **CustomerAge** by calculating the difference between the transaction date and the customer’s date of birth (DOB), converting it into an age feature to capture relevant demographic information.
- Addressed skewness in financial variables by applying logarithmic transformations to **`CustAccountBalance`** and **`TransactionAmount (INR)`**, creating `LogBalance` and `LogTransactionAmount`.
- Dropped the original `CustAccountBalance` and `TransactionAmount (INR)` columns to avoid redundancy and multicollinearity.
- The final feature set for clustering included:  
  - `TransactionTime`  
  - `CustomerAge` (engineered from DOB and transaction date)  
  - `CustGender_M`  
  - `LogBalance`  
  - `LogTransactionAmount`
- Used StandardScaler to normalize features before clustering.

## Clustering Methodology

- Applied KMeans clustering on the full dataset (over 1 million rows).
- Explored cluster counts (`k`) from 2 to 9 and selected the optimal number based on the **silhouette score**.
- Used PCA for dimensionality reduction and visualization of cluster separation.

## Key Findings

- The silhouette score peaked at **k=2 (0.6076)**, indicating two distinct customer clusters.
- At k=2, clusters were primarily divided by gender and financial behavior:  
  - Cluster 0: Female customers with higher balances and larger transactions.  
  - Cluster 1: Male customers with slightly lower balances and smaller transactions.
- At k=3 (silhouette score 0.6070), an additional meaningful cluster emerged:  
  - Cluster 0: Female customers, moderate balances and transactions.  
  - Cluster 1: Younger male customers with lower balances and smaller transactions.  
  - Cluster 2: Older male customers with the highest balances and transaction amounts.
- PCA visualizations confirmed clear and meaningful cluster separation.

## Business Insights

- The segmentation reveals actionable customer groups for targeted marketing, personalized offerings, and loyalty programs.
- Distinct age and gender segments provide deeper understanding of customer behavior patterns.
- The identification of a high-value older male segment offers opportunities for premium product engagement.

- ---

---

# 🏠 House Price Prediction (kc\_house\_data)

## 📌 Project Objective

This project aims to predict house prices using various regression models trained on the **kc\_house\_data** dataset. The goal is to build a model that can make accurate price predictions for real estate in King County.

## ⚙️ Final Model: Stacking Regressor

After experimenting with several regression algorithms—including Decision Tree, Random Forest, Extra Trees, XGBoost, and Gradient Boosting—a **Stacking Regressor** was selected as the final model. This ensemble technique combines the predictive power of multiple base models and a meta-model to enhance overall accuracy.

## 📈 Performance on Test Data

| Metric                             | Value    |
| ---------------------------------- | -------- |
| **R² Score**                       | 0.9979   |
| **Mean Absolute Error (MAE)**      | \$7,600  |
| **Root Mean Squared Error (RMSE)** | \$14,400 |

## 📊 Interpretation

* ✅ **R² Score of 0.9979** indicates that the model explains **nearly 99.8%** of the variance in house prices, showing an excellent fit.
* 📉 **MAE of \$7,600** suggests that, on average, predictions deviate slightly from actual values—reasonable given the price range.
* 📦 **RMSE of \$14,400** reflects occasional larger errors, but overall predictions are reliable and consistent.

## ✅ Conclusion

The **Stacking Regressor** provides a strong balance of **accuracy and generalization**, making it a practical solution for predicting house prices. For deployment in production or use in real estate applications, it's recommended to validate performance on **new or real-time data** to ensure consistent results.

---
---

# 🏠 House Price Prediction

## 📌 Project Objective

This project aims to build a machine learning model to accurately predict house prices using a structured housing dataset. Multiple regression models were tested to find the most reliable solution for practical use.


## ✅ Final Model Selection: ExtraTreesRegressor

After evaluating a range of regression models—including Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Stacking Regressor—the **ExtraTreesRegressor** demonstrated the best performance.

This model produced the **lowest prediction error** and a **reasonable R² score**, making it the most suitable choice for this task.


## 📈 Performance Summary (on Test Data)

| Metric                        | Value      |
| ----------------------------- | ---------- |
| **Mean Absolute Error (MAE)** | \$8,623.50 |
| **R² Score**                  | **0.67**   |


## 🧠 Interpretation

* 🔍 **MAE of \$8,623.50** indicates low average deviation from actual house prices, showing strong real-world applicability.
* 📉 **R² score of 0.67** means the model explains **67% of the variance** in house prices—showing moderate but reliable predictive ability.


## 🚀 Conclusion

The **ExtraTreesRegressor** offers a strong balance of **accuracy and stability**, outperforming other tested models in this project. While the R² score leaves room for improvement, the low MAE suggests it is effective for practical housing price predictions. Further improvement may be achieved through advanced feature engineering or additional data collection.

---

---
