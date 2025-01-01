import joblib
import json

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Save vocabulary and IDF values to JSON
tfidf_config = {
    "vocabulary": vectorizer.vocabulary_,
    "idf": vectorizer.idf_.tolist()
}

with open("./tfidf_config.json", "w") as f:
    json.dump(tfidf_config, f)

print("TF-IDF configuration saved to tfidf_config.json")
