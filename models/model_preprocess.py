import yaml
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# Load and clean data


with open("CONFIGS.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = Path(config["data_path"])
MODEL_PATH = Path(config["model_path"])

df = pd.read_csv(DATA_PATH)

# Normalize column capitalization
df.rename(columns={"Residence_type": "residence_type"}, inplace=True)

# Drop unnecessary columns if they exist
df.drop(columns=["id"], errors="ignore", inplace=True)

# Convert BMI to numeric; invalid values become NaN
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# Remove rows without a target label
df.dropna(subset=["stroke"], inplace=True)

# Display dataset information
print("\nDataset information:")
print(df.info())

print("\nTarget distribution:")
print(df["stroke"].value_counts())

print("\nTarget proportions:")
print(df["stroke"].value_counts(normalize=True))


# Separate features and target


X = df.drop(columns=["stroke"])
y = df["stroke"]

# Define feature groups
numerical_cols = ["age", "avg_glucose_level", "bmi"]

categorical_cols = [
    "gender",
    "work_type",
    "ever_married",
    "residence_type",
    "smoking_status",
]

# Validate the expected input schema
expected_cols = numerical_cols + categorical_cols

missing_cols = set(expected_cols) - set(X.columns)

if missing_cols:
    raise ValueError(f"Missing expected feature columns: {missing_cols}")

# Retain only the intended model features
X = X[expected_cols]


# Split into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split complete.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# Define preprocessing pipelines


# Numerical features:
# Replace missing values with the training-set median
numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# Categorical features:
# Replace missing values and one-hot encode categories
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# Define the Random Forest model


random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# Combine preprocessing and model into one pipeline


model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", random_forest)])


# Train the model


print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete!")


# Generate predictions


y_pred = model.predict(X_test)

# Probability that each test sample belongs to class 1
y_proba = model.predict_proba(X_test)

# Locate the probability column corresponding to stroke = 1
classifier = model.named_steps["classifier"]
positive_class_index = np.where(classifier.classes_ == 1)[0][0]

stroke_probabilities = y_proba[:, positive_class_index]


# Evaluate model performance


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

roc_auc = roc_auc_score(y_test, stroke_probabilities)

print("\n" + "=" * 50)
print("RANDOM FOREST EVALUATION")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Display feature importances


feature_names = model.named_steps["preprocessor"].get_feature_names_out()

feature_importances = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importances}
)

importance_df = importance_df.sort_values(by="importance", ascending=False)

print("\nTop 15 Feature Importances:")
print(importance_df.head(15).to_string(index=False))


# Save the complete trained pipeline


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
