from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import joblib

# Initialize the app and ONNX runtime
app = Flask(__name__)
session = ort.InferenceSession("models/inventory_recommender.onnx")

# Load TF-IDF Vectorizer and Label Encoders from .pkl files
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder_category = joblib.load("models/label_encoder_category.pkl")
label_encoder_status = joblib.load("models/label_encoder_status.pkl")
label_encoder_recommendation = joblib.load("models/label_encoder_recommendation.pkl")

# Encode categorical features
def encode_category(category):
    try:
        return label_encoder_category.transform([category])[0]
    except ValueError:
        return -1  # Return an invalid value if the category is not found

def encode_status(status):
    try:
        return label_encoder_status.transform([status])[0]
    except ValueError:
        return -1  # Return an invalid value if the status is not found

# Decode the model's prediction
def decode_prediction(prediction):
    try:
        return label_encoder_recommendation.inverse_transform([prediction])[0]
    except ValueError:
        return "Unknown"

# Preprocess input
def preprocess_input(data):
    description = data["description"]
    category = data["category"]
    status = data["status"]

    # Vectorize the description using TF-IDF
    tfidf_vector = vectorizer.transform([description]).toarray()[0]
    category_encoded = encode_category(category)
    status_encoded = encode_status(status)

    # Ensure valid inputs
    if category_encoded == -1 or status_encoded == -1:
        raise ValueError("Invalid category or status value")

    # Combine features into a single vector
    input_features = np.concatenate([tfidf_vector, [category_encoded, status_encoded]])

    # Pad or trim the feature vector to match the expected input size of the ONNX model
    expected_input_size = session.get_inputs()[0].shape[1]
    if len(input_features) < expected_input_size:
        input_features = np.pad(input_features, (0, expected_input_size - len(input_features)))
    elif len(input_features) > expected_input_size:
        input_features = input_features[:expected_input_size]

    return input_features.astype(np.float32).reshape(1, -1)

# Inference route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.json

        # Preprocess input
        input_features = preprocess_input(data)

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_features})
        prediction = int(np.argmax(outputs[0]))  # Assuming classification model

        # Decode prediction
        recommendation = decode_prediction(prediction)

        return jsonify({"recommendation": recommendation})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred during inference", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
