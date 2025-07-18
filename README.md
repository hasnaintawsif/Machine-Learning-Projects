# Spam-email-detector
# 📧 Spam Email Detection (Stacking Ensemble)

This project uses a stacking ensemble to classify emails as spam or not spam.

## 💼 Key Highlights

- ✅ Achieved 95% test accuracy using stacked classifiers
- 🤖 Models: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Extra Trees
- 🔝 Meta-model: XGBoost
- 💾 Models saved with `joblib` for deployment
- 📊 Evaluation done using test set and cross-validation

## 🚀 How to Predict on New Data

```python
import joblib
import numpy as np
import pandas as pd

x_new = pd.read_csv('your_input.csv')

base_models = joblib.load('base_models.pkl')
meta_model = joblib.load('spam_detection.pkl')

blend = np.zeros((x_new.shape[0], len(base_models)))
for i, model in enumerate(base_models):
    blend[:, i] = model.predict_proba(x_new)[:, 1]

predictions = meta_model.predict(blend)
