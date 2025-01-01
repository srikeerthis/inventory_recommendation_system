import joblib
import json

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Save label mappings
label_mapping = {
    "classes": label_encoder.classes_.tolist()
}

with open("./label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

print("Label mapping saved to label_mapping.json")
