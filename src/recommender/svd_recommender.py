import pandas as pd
import random
import pickle
from surprise import SVD


class SVDRecommender:
    """
    This class loads a trained SVD recommendation model
    and generates personalized book recommendations for users.
    """

    def __init__(self, model_path, data_path):
        # load the trained model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # load data
        self.df = pd.read_csv(data_path)

        # cache for speed
        self.all_books = self.df['book_id'].unique()  # all book's id in dataset
        self.all_users = self.df['user_id'].unique()

    # choice a random user function
    def get_random_user(self):
        return random.choice(self.all_users)

    # recommendation function
    def recommend(self, user_id, top_n=10):
        # handle cold-start users
        if user_id not in self.all_users:
            raise ValueError("User ID not found in dataset.")

        seen_books = self.df[self.df['user_id'] == user_id]['book_id'].tolist()  # holds user's book
        candidate_books = [b for b in self.all_books if b not in seen_books]

        predictions = []  # this list holds predictions for each 'not seen' book
        for book in candidate_books:
            pred = self.model.predict(user_id, book)
            predictions.append((pred.iid, pred.est))

        # sort top_n books in descending order
        top_predictions = sorted(
            predictions,
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # create a dataframe for HTML
        rec_df = pd.DataFrame(
            top_predictions,
            columns=['book_id', 'predicted_rating']
        )

        rec_df = rec_df.merge(
            self.df[['book_id', 'title']].drop_duplicates(),
            on='book_id'
        )

        # format ratings for better display
        rec_df['predicted_rating'] = rec_df['predicted_rating'].round(2)

        return rec_df[['title', 'predicted_rating']]
