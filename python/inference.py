import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the trained model and preprocessing tools
model = joblib.load("models/inventory_recommender.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder_category = joblib.load("models/label_encoder_category.pkl")
label_encoder_status = joblib.load("models/label_encoder_status.pkl")
label_encoder_recommendation = joblib.load("models/label_encoder_recommendation.pkl")

# Input example
input_data = {
    "description": "Comfortable chair, slightly worn",
    "category": "Furniture",
    "status": "Not Used"
}

# Preprocess input
try:
    # Vectorize the description using TF-IDF
    tfidf_vector = vectorizer.transform([input_data["description"]]).toarray()

    # Encode categorical features
    category_encoded = label_encoder_category.transform([input_data["category"]])[0]
    status_encoded = label_encoder_status.transform([input_data["status"]])[0]

    # Combine all features into a DataFrame with appropriate column names
    tfidf_feature_names = vectorizer.get_feature_names_out()
    feature_dict = {name: value for name, value in zip(tfidf_feature_names, tfidf_vector[0])}
    feature_dict["category_encoded"] = category_encoded
    feature_dict["status_encoded"] = status_encoded

    input_features = pd.DataFrame([feature_dict])  # Create a DataFrame

    # Run inference
    prediction = model.predict(input_features)[0]

    # Decode prediction
    recommendation = label_encoder_recommendation.inverse_transform([prediction])[0]

    print("Recommendation:", recommendation)

except Exception as e:
    print("Error during inference:", str(e))
