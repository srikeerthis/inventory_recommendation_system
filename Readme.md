# Inventory Recommendation System - ONNX Model Deployment

## Overview

This project demonstrates the development and deployment of a machine learning model for inventory recommendation. The model processes input data describing inventory items (e.g., description, category, and usage status) and provides actionable recommendations such as "Buy," "Sell," or "Rent." The model is trained using Python, exported to ONNX format, and prepared for deployment.

## Features

- Model Training: Implements a Random Forest Classifier to predict inventory recommendations.
- Preprocessing: Utilizes TF-IDF vectorization for text descriptions and label encoding for categorical features.
- ONNX Export: Converts the trained model to ONNX format for lightweight and efficient inference.
- Preprocessing Configuration: Exports TF-IDF vocabulary and label mappings to JSON for integration with various runtimes.

## Directory Structure

```
project/
├── python/
│ ├── export_tfidf.py   # Python script to export TF-IDF config
│ ├── export_labels.py  # Python script to export label mappings
│ ├── export_onnx.py    # Python script to convert model to ONNX
│ ├── recommendation.py # Python script to train and store the model
```

## Requirements

### Python Environment

- Python 3.7+
- Required Libraries:
  - scikit-learn
  - joblib
  - skl2onnx

## Setup

### Python Environment

1. Set up Python virtual environment:

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train and Save the Model:
   Run recommender.py to generate the Random Forest model (inventory_recommender.pkl) and also the preprocessing objects (tfidf_vectorizer.pkl, label_encoder.pkl).

```
python recommender.py
```

3. Export Preprocessing Configurations:
   Run the scripts to export preprocessing configurations:

```
python export_tfidf.py
python export_labels.py
```

4. Convert Model to ONNX:

```
python export_onnx.py
```

This will generate the ONNX model file (inventory_recommender.onnx).
