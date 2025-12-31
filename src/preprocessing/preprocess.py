import pandas as pd
from pathlib import Path
import re

# setting and constants
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_BOOKS = BASE_DIR / "dataset" / "raw" / "books_metadata_large.csv.gz"
INPUT_INTERACTIONS = BASE_DIR / "dataset" / "raw" / "interactions_large.csv.gz"
OUTPUT_FILE = BASE_DIR / "dataset" / "processed" / "cleaned.csv"

# columns to keep
COLS_BOOK = ["book_id", "title", "average_rating", "ratings_count","publication_year", "num_pages"]
COLS_INTERACTION = ["user_id", "book_id", "rating"]

# regex for titles containing only Latin characters
latin_regex = re.compile(r'^[A-Za-z0-9\s\.,:;!?\'"-]+$')

# load the data
print("Loading data...")
# Load books metadata
df_books = pd.read_csv(INPUT_BOOKS, usecols=COLS_BOOK)
# Load user interactions
df_interactions = pd.read_csv(INPUT_INTERACTIONS, usecols=COLS_INTERACTION)

# filtering
print("Filtering books...")
# filtering by Latin characters
df_books = df_books[df_books['title'].apply(lambda x: bool(latin_regex.match(str(x))))] # keep Latin titles
# year filter
df_books = df_books[(df_books['publication_year'] >= 1500) & (df_books['publication_year'] <= 2025)]
print("Filtering interactions...")
# rating filter
df_interactions = df_interactions[(df_interactions['rating'] >= 1) & (df_interactions['rating'] <= 5)]

# merging
print("Merging tables...")
# merge datasets
df_merged = pd.merge(df_interactions, df_books, on="book_id", how="inner")

# cleaning
print("Performing final cleaning...")
# remove missing data
df_merged = df_merged.dropna()
# remove duplicates
df_merged = df_merged.drop_duplicates(subset=['user_id', 'book_id'])
# remove invalid values
df_merged = df_merged[(df_merged['num_pages'] > 0) & (df_merged['ratings_count'] >= 0)]

# export
print("Saving file...")
df_merged.info()
# keep only essential columns
final_output = df_merged[
    ["user_id", "book_id", "rating", 
     "publication_year", "num_pages", 
     "ratings_count", "average_rating", "title"]
]
final_output.to_csv(OUTPUT_FILE, index=False)

print(f"Process completed! '{OUTPUT_FILE}' created.")