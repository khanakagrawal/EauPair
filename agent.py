import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from rapidfuzz import fuzz, process
from serpapi import GoogleSearch


load_dotenv()

# Step 1: Load Perfume Dataset
df = pd.read_csv("fra_cleaned.csv", encoding="latin1", delimiter=";")
df = df.fillna("")
df = df.iloc[1:, 1:] # Drop the first row and column

# Clean dataset columns
df["Perfume_clean"] = df["Perfume"].astype(str).str.lower().str.strip()
df["Brand_clean"] = df["Brand"].astype(str).str.lower().str.strip()

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

# === Flask app ===
app = Flask(__name__)

# === Groq client ===
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_perfume_price(query):
    search = GoogleSearch({
        "q": f"{query} perfume price",
        "api_key": os.environ.get("SERP_API_KEY"),
        "location": "United States"
    })
    results = search.get_dict()

    if "shopping_results" in results and results["shopping_results"]:
        first_item = results["shopping_results"][0]
        return first_item.get("extracted_price") or first_item.get("price")

    return None

@app.route("/dupes", methods=["GET"])
def find_dupes():
    raw_perfume = request.args.get("perfume")
    raw_brand = request.args.get("brand")

    if not raw_perfume:
        return jsonify({"error": "Perfume name must be provided"}), 400

    # Normalized version for dataset lookup
    perfume_name = raw_perfume.lower().replace(" ", "-").strip()
    brand_name = raw_brand.lower().replace(" ", "-").strip()

    choices = df["Perfume_clean"].tolist()
    best_match_perfume, score, idx = process.extractOne(
        perfume_name, choices, scorer=fuzz.token_sort_ratio
    )

    if score < 70:  # threshold for a confident match
        return jsonify({"error": f"Perfume '{raw_perfume}' not found"}), 404

    # Optionally check brand if provided
    if brand_name:
        if df.loc[idx, "Brand_clean"] != brand_name:
            return jsonify({
                "error": f"Perfume '{raw_perfume}' found, but brand does not match '{raw_brand}'"
            }), 404

    # Use the fuzzy-matched index
    idx = df.index[idx]

    # Get the user's perfume row
    matched_row = df.loc[idx].to_dict()
    matched_row["Price"] = get_perfume_price(matched_row["Perfume"])

    sim_scores = cosine_similarity(X[idx], X).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:21]  # top 20

    results = []
    for i in similar_indices:
        results.append({
            "Perfume": df.loc[i, "Perfume"],
            "Brand": df.loc[i, "Brand"],
            "Notes": df.loc[i, "Notes"],
            "Price": get_perfume_price(df.loc[i, "Perfume"])
        })

    

    # === Use Groq to generate explanation ===
    rec_text = "\n".join(
        [f"- {r['Perfume']} by {r['Brand']}, price = ${r['Price']}| {r['Notes']}" for r in results]
    )

    prompt = f"""
    You are an expert in finding perfume dupes.
    A user likes the perfume {raw_perfume}:'{matched_row}'.
    Here are 20 similar perfumes:\n{rec_text}\n
    Task:
- Pick the best dupe considering similarity in notes, accords, and gender,
- BUT also balance **affordability vs similarity** (a cheaper but still similar perfume might be better than an expensive one),
- Explain your reasoning clearly but keep it concise.
- Return valid JSON with:
  - best_dupe (string)
  - explanation (string)
  - considered (array of objects with perfume name, % similarity, reason, and price).
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        explanation = f"Error generating AI explanation: {str(e)}"


    return jsonify({
        "perfume": perfume_name,
        "results": results,
        "ai_explanation": explanation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)