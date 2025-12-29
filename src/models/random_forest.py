import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,RocCurveDisplay)

# Load dataset
dataset = pd.read_csv("data/processed/cleaned.csv")

# Binary classification: books with rating > 3 are considered popular
popularityBound = 3
dataset["label"] = (dataset["rating"] > popularityBound).astype(int)

# Feature selection
features = ["publication_year","num_pages","ratings_count","average_rating"]

X = dataset[features]
y = dataset["label"]

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42,stratify=y)  

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Random Forest Performance")
print("-" * 40)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\nFeature Importances:")
print(feat_imp)

# Visualization
plt.figure(figsize=(8, 5))
feat_imp.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Roc curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("Random Forest ROC Curve")
plt.show()

