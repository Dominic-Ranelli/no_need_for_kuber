from pathlib import Path
import yaml
from fastapi import FastAPI, status
from models.simple_model import fetch_pred_record
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import numpy as np
from time import perf_counter
import time


# Startup validation helper
def validate_artifact():
    with open("CONFIGS.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["model_path"])

    if not model_path.is_file():
        raise RuntimeError(f"Model artifact not found at expected path: {model_path}")

    try:
         _ = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact at {model_path}: {exc}")



app = FastAPI()


load_time_start = time.perf_counter()
with open("CONFIGS.yaml", "r") as f:
    config = yaml.safe_load(f)


@app.on_event("startup")
def on_startup():
    validate_artifact()

model_path = Path(config["model_path"])

model = joblib.load(model_path)
load_time_end = time.perf_counter()

class Input(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: int = Field(ge=0, le=120)
    avg_glucose_level: float = Field(ge=0, le=2000)
    bmi: float
    work_type: str
    ever_married: Literal["Yes", "No"]
    residence_type: Literal["Urban", "Rural"]
    smoking_status: str

@app.get("/")
async def root():
    return {"Message": "Hello World"}

# Adding load time request
@app.get("/load")
async def load_time(start = load_time_start, end = load_time_end):
    return {"Load time of model:": f"{(end - start)*1000} milliseconds"}


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    return {"Status": "Healthy"}


@app.get("/version")
def get_version():
    return {"Model version": "1.0.0"}


@app.post("/predict")
def post_prediction(input: Input):
    # Construct dataframe
    input_df = pd.DataFrame([input.dict()])

    # input must match column order input
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

    # Make prediction
    run_time_start = time.perf_counter()
    predicted_class = model.predict(input_df)
    run_time_end = time.perf_counter()

    stroke_dict = {0: "Non-stroke", 1: "Stroke"}

    return [f"Predicted class is {stroke_dict[predicted_class.item()]}", f"Ran in {(run_time_end-run_time_start)*1000} milliseconds"]
