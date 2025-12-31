"""
We evaluated multiple recommendation models during the experimentation phase
Among all tested models, SVD achieved the best performance in terms of RMSE and MAE.

This script retrains the final SVD model on the full training data
and saves it for deployment in the recommendation system.
"""

import pandas as pd
import pickle
import os

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Read dataset
df = pd.read_csv("dataset/processed/cleaned.csv")

# Create Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    df[['user_id', 'book_id', 'rating']],
    reader
)

# Train-test split
# The test set is not required here since this model
# will be used for inference in the deployed recommender system
trainset, _ = train_test_split(
    data,
    test_size=0.3,
    random_state=42
)

# Train SVD model
model = SVD(
    n_factors=50,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)
model.fit(trainset)

# Create results folder
os.makedirs("results", exist_ok=True)

# Save ONLY the trained model
with open("results/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SVD model saved to results/svd_model.pkl")

