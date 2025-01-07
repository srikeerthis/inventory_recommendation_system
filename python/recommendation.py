import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Dataset
data = {
    "name": [
        "Hammer", "Sofa", "Tennis Racket", "Jacket", "Guitar", 
        "Drill", "Table", "Football", "Coat", "Piano",
        "Screwdriver", "Chair", "Basketball", "Gloves", "Violin",
        "Ladder", "Bed", "Cricket Bat", "Hat", "Keyboard",
        "Saw", "Desk", "Soccer Ball", "Shirt", "Flute"
    ],
    "description": [
        "Heavy-duty tool, slightly worn",
        "Comfortable three-seater sofa, old",
        "Sports gear, like new condition",
        "Winter coat, damaged zipper",
        "Vintage musical instrument, excellent condition",
        "Electric drill, in good condition",
        "Dining table, slightly scratched",
        "Football, worn but usable",
        "Raincoat, missing a button",
        "Classic piano, needs tuning",
        "Basic screwdriver, new condition",
        "Wooden chair, solid frame",
        "Basketball, slightly deflated",
        "Winter gloves, in good condition",
        "Antique violin, needs minor repair",
        "Aluminum ladder, sturdy build",
        "Queen-sized bed, good mattress",
        "Professional-grade cricket bat",
        "Stylish summer hat",
        "Digital keyboard, lightly used",
        "Hand saw, sharp blade",
        "Ergonomic office desk",
        "Soccer ball, new",
        "Cotton shirt, barely used",
        "Classic wooden flute, well-maintained"
    ],
    "category": [
        "Tools", "Furniture", "Sporting Gear", "Clothing", "Musical Instrument",
        "Tools", "Furniture", "Sporting Gear", "Clothing", "Musical Instrument",
        "Tools", "Furniture", "Sporting Gear", "Clothing", "Musical Instrument",
        "Tools", "Furniture", "Sporting Gear", "Clothing", "Musical Instrument",
        "Tools", "Furniture", "Sporting Gear", "Clothing", "Musical Instrument"
    ],
    "status": [
        "Frequently Used", "Not Used", "Less Frequently Used", "Faulty", "Wishlist",
        "Frequently Used", "Not Used", "Less Frequently Used", "Faulty", "Wishlist",
        "Frequently Used", "Not Used", "Less Frequently Used", "Faulty", "Wishlist",
        "Frequently Used", "Not Used", "Less Frequently Used", "Faulty", "Wishlist",
        "Frequently Used", "Not Used", "Less Frequently Used", "Faulty", "Wishlist"
    ],
    "recommendation": [
        "Buy", "Sell", "Rent", "Sell", "Buy",
        "Buy", "Sell", "Rent", "Sell", "Buy",
        "Buy", "Sell", "Rent", "Sell", "Buy",
        "Buy", "Sell", "Rent", "Sell", "Buy",
        "Buy", "Sell", "Rent", "Sell", "Buy"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 2: Encode Categorical Features
label_encoder_category = LabelEncoder()
label_encoder_status = LabelEncoder()
label_encoder_recommendation = LabelEncoder()

df["category_encoded"] = label_encoder_category.fit_transform(df["category"])
df["status_encoded"] = label_encoder_status.fit_transform(df["status"])
y = label_encoder_recommendation.fit_transform(df["recommendation"])

# Step 3: Vectorize Text Descriptions
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])

# Step 4: Combine All Features
X = pd.concat(
    [
        pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()),
        df[["category_encoded", "status_encoded"]],
    ],
    axis=1,
)

# Step 5: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Step 7: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Step 8: Train the Model with Best Hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 9: Evaluate the Model
y_pred = best_model.predict(X_test)
target_names = label_encoder_recommendation.inverse_transform(range(len(label_encoder_recommendation.classes_)))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1))

# Step 10: K-Fold Cross-Validation
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"K-Fold Cross Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Step 11: Save the Model and Preprocessing Tools
joblib.dump(best_model, "inventory_recommender.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder_category, "label_encoder_category.pkl")
joblib.dump(label_encoder_status, "label_encoder_status.pkl")
joblib.dump(label_encoder_recommendation, "label_encoder_recommendation.pkl")

print("Model, vectorizer, and label encoders saved.")
