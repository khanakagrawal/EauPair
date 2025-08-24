import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

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

@app.route("/dupes", methods=["GET"])
def find_dupes():
    raw_perfume = request.args.get("perfume")
    raw_brand = request.args.get("brand")

    if not raw_perfume or not raw_brand:
        return jsonify({"error": "Both perfume and brand must be provided"}), 400

    # Normalized version for dataset lookup
    perfume_name = raw_perfume.lower().replace(" ", "-").strip()
    brand_name = raw_brand.lower().replace(" ", "-").strip()

    match = df[
        (df["Perfume"].str.lower() == perfume_name) &
        (df["Brand"].str.lower() == brand_name)
    ]

    if match.empty:
        return jsonify({"error": f"Perfume '{perfume_name}' by '{brand_name}' not found"}), 404

    idx = match.index[0]
    sim_scores = cosine_similarity(X[idx], X).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:2]  # top 20

    results = []
    for i in similar_indices:
        results.append({
            "Perfume": df.loc[i, "Perfume"],
            "Brand": df.loc[i, "Brand"],
            "Notes": df.loc[i, "Notes"],
            "Similarity": round(sim_scores[i], 3)
        })

    # === Use Groq to generate explanation ===
    rec_text = "\n".join(
        [f"- {r['Perfume']} by {r['Brand']} (Similarity {r['Similarity']}) | {r['Notes']}" for r in results]
    )

    prompt = f"""
    A user likes the perfume '{perfume_name}'.
    Here are 20 similar perfumes:\n{rec_text}\n
    Please explain in a friendly way why these make good dupes,
    highlighting note overlaps, accords, and value. Give each perfume 
    a score from 1 to 10 based on their similarity to the original perfume
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