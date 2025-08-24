import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Step 1: Load Perfume Dataset
df = pd.read_csv("fra_cleaned.csv", encoding="latin1", delimiter=";")
df = df.fillna("")
df = df.iloc[1:, 1:] # Drop the first row and column

# Represent perfumes by their notes using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")

df["Notes"] = (
    "Top Notes: " + df["Top"].astype(str) + "; " +
    "Middle Notes: " + df["Middle"].astype(str) + "; " +
    "Base Notes: " + df["Base"].astype(str) + "; " +
    "Main Accords: " + df["mainaccord1"].astype(str) + ", " +
    df["mainaccord2"].astype(str) + ", " +
    df["mainaccord3"].astype(str) + ", " +
    df["mainaccord4"].astype(str) + ", " +
    df["mainaccord5"].astype(str)
)

X = vectorizer.fit_transform(df["Notes"])
