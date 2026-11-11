# Imports
import numpy as np
import yaml
from app.main import app
import joblib
import time
from time import perf_counter
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Test under similar conditions, how long does our model take?

# Import app, not testing load time
client = TestClient(app)

# Variable for times
times = []

with open("CONFIGS.yaml", "r") as f:
    config = yaml.safe_load(f)

# Define payload before creating the DataFrame
payload = {
    "age": 34.0,
    "avg_glucose_level": 85.2,
    "bmi": 22.1,
    "gender": "Male",
    "work_type": "Private",
    "ever_married": "No",
    "residence_type": "Urban",
    "smoking_status": "never smoked",
}

input_df = pd.DataFrame([payload])

# input must match column order expected by the model
expected_cols = [
    "age",
    "avg_glucose_level",
    "bmi",
    "gender",
    "work_type",
    "ever_married",
    "residence_type",
    "smoking_status",
]
input_df = input_df[expected_cols]
model_path = Path(config["model_path"])

model = joblib.load(model_path)

# test 100 times
for i in range(100):
    # Time diagnostics
    run_start = time.perf_counter()
    predicted_class = model.predict(input_df)
    run_end = time.perf_counter()
    # Add to list (in milliseconds)
    times.append((run_end - run_start) * 1000)

# Calculating p50 and p95 latency
p_50 = np.percentile(times, 50)
p_95 = np.percentile(times, 95)

print(f"p50 latency = {p_50} milliseconds. p95 latency = {p_95} milliseconds")