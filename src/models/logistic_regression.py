import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import (accuracy_score, classification_report,precision_score,recall_score,f1_score)
from sklearn.metrics import roc_auc_score

# Read the dataset (relative path)
dataset = pd.read_csv("dataset/processed/cleaned.csv")

# Binary classification: books with rating > 3 are considered popular
popularityBound = 3
dataset['label'] = (dataset['rating'] > popularityBound).astype(int)

# Select features
features = ['publication_year','num_pages','ratings_count']
print(dataset[['title','rating','label']].head())  # optional

D = dataset[features]  # Feature matrix
y = dataset['label']   # Target variable

# Split the dataset into 70% train and 30% test
D_train, D_test, y_train, y_test = train_test_split(D, y, test_size=0.3, random_state=42)

# Scaler for feature normalization
scaler = StandardScaler()
D_train = scaler.fit_transform(D_train)
D_test = scaler.transform(D_test)

# Logistic Regression MODEL
model = LogisticRegression()
model.fit(D_train, y_train)

# Prediction and evaluation
y_pred = model.predict(D_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(D_test)[:,1])
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization: Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(D_test[y_test==0][:,1], D_test[y_test==0][:,2], color='red', label='Not Popular')
plt.scatter(D_test[y_test==1][:,1], D_test[y_test==1][:,2], color='green', label='Popular')
plt.xlabel('Num Pages (standardized)')
plt.ylabel('Ratings Count (standardized)')
plt.legend()
plt.title('Books Popularity Scatter Plot')
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(model, D_test, y_test)
plt.title("Logistic Regression ROC Curve")
plt.show()


