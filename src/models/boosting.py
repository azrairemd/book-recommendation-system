import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,RocCurveDisplay)

# Read the dataset (relative path)
dataset = pd.read_csv("dataset/processed/cleaned.csv")

# Create binary popularity label
dataset['label'] = (dataset['rating'] > 3).astype(int)

# Select features
features = ['publication_year', 'num_pages', 'ratings_count', 'average_rating']

D = dataset[features]  # Feature matrix
y = dataset['label']   # Target variable

# Split the dataset into 70% train and 30% test
D_train, D_test, y_train, y_test = train_test_split(D, y,test_size=0.3,random_state=42,stratify=y)

# Scaler for feature normalization
scaler = StandardScaler()
D_train_scaled = scaler.fit_transform(D_train)
D_test_scaled = scaler.transform(D_test)

# Weak learner
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# Configure AdaBoost with decision tree base learner
adaboost_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.5,  # Weight of each weak learner
    random_state=42
)

# Train AdaBoost model
adaboost_model.fit(D_train_scaled, y_train)

# Prediction and evaluation
y_pred = adaboost_model.predict(D_test_scaled)
y_proba = adaboost_model.predict_proba(D_test_scaled)[:, 1]

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

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC curve 
RocCurveDisplay.from_estimator(adaboost_model, D_test_scaled, y_test)
plt.title("AdaBoost ROC Curve")
plt.show()


