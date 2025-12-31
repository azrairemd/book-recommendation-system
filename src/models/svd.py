import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read the dataset (relative path)
df = pd.read_csv("dataset/processed/cleaned.csv")

# Define binary popularity for ROC curve (rating > 3 as popular)
df['label'] = (df['rating'] > 3).astype(int)

# Create Surprise dataset
reader = Reader(rating_scale=(1,5))  # Define rating scale for Surprise
data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)

# Split the dataset into 70% train and 30% test
trainset, testset = train_test_split(data, test_size=0.3, random_state=42)

# Define and train SVD model
model = SVD(
    n_factors=50,        # Number of latent factors
    n_epochs=30,         # Number of training iterations
    lr_all=0.005,        # Learning rate for all parameters
    reg_all=0.02,        # Regularization term
    random_state=42
)

model.fit(trainset)

# Prediction and evaluation
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)  # Root Mean Squared Error
mae = accuracy.mae(predictions)    # Mean Absolute Error
print(f" SVD RMSE: {rmse}, MAE: {mae}")

# ROC Curve preparation
y_true = np.array([pred.r_ui > 3 for pred in predictions])  # Binary labels
y_score = np.array([pred.est for pred in predictions])      # Predicted ratings
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
print(f"SVD ROC-AUC: {roc_auc:.4f}")

# ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVD Recommendation Model')
plt.legend(loc="lower right")
plt.show()

