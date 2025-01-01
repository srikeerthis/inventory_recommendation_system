import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the Random Forest model
model = joblib.load("inventory_recommender.pkl")

# Define the input type
feature_count = 500  # Update this to match your feature size
initial_type = [("float_input", FloatTensorType([None, feature_count]))]

# Convert the model to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model
with open("./inventory_recommender.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model exported to inventory_recommender.onnx")
