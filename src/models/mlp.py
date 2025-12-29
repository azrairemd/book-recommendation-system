import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score, roc_auc_score,classification_report,RocCurveDisplay)

# Read the dataset (relative path)
dataset = pd.read_csv("data/processed/cleaned.csv")

# Define a rule for classification
popularityBound = 3
dataset['label'] = (dataset['rating'] > popularityBound).astype(int)

# Select features
features = ['publication_year', 'num_pages', 'ratings_count','average_rating']

D = dataset[features]  # Feature matrix
y = dataset['label']   # Target variable

# Split the dataset into 70% train and 30% test
D_train, D_test, y_train, y_test = train_test_split( D, y,test_size=0.3,random_state=42,stratify=y)

# Scaler for feature normalization
scaler = StandardScaler()
D_train_scaled = scaler.fit_transform(D_train)
D_test_scaled = scaler.transform(D_test)

# Balance classes using smote
smote = SMOTE(random_state=42, sampling_strategy=0.5)
D_train_res, y_train_res = smote.fit_resample(D_train_scaled, y_train)

# Define MLP as base learner
mlp_model = MLPClassifier(
    hidden_layer_sizes=(32,16,8),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=2000,
    early_stopping=True,
    random_state=42
)

# Configure Bagging Ensemble
bagging = BaggingClassifier(
    estimator=mlp_model,
    n_estimators=50,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# Train Bagging Ensemble
bagging.fit(D_train_res, y_train_res)

# Prediction and evaluation
y_pred = bagging.predict(D_test_scaled)
y_proba = bagging.predict_proba(D_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
RocCurveDisplay.from_estimator(bagging, D_test_scaled, y_test)
plt.title("Optimized Bagging + MLP ROC Curve")
plt.show()

