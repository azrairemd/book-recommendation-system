import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.recommender.svd_recommender import SVDRecommender

# paths
MODEL_PATH = "results/svd_model.pkl"
DATA_PATH = "data/processed/cleaned.csv"

# recommender instance
recommender = SVDRecommender(MODEL_PATH, DATA_PATH)

def get_recommendation(top_n=5):
    user_id = recommender.get_random_user()
    recommendations = recommender.recommend(user_id, top_n)
    return user_id, recommendations
