from fastapi import FastAPI, status
from models.simple_model import fetch_pred_record
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import numpy as np


class Input(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    age: int = Field(ge=0, le=120)
    avg_glucose_level: float = Field(ge=0, le=2000)
    bmi: float
    work_type: str
    ever_married: Literal['Yes', 'No']
    residence_type: Literal['Urban', 'Rural']
    smoking_status: str




app = FastAPI()

with open('models/random_forest_stroke.joblib', 'rb') as f:
    model = joblib.load(f)
    
@app.get("/")
async def root():
    return {"Message": "Hello World"}


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
        "age", "avg_glucose_level", "bmi",
        "gender", "work_type", "ever_married",
        "residence_type", "smoking_status"
    ]
    input_df = input_df[expected_cols]

    # Make prediction
    predicted_class = model.predict(input_df)

    stroke_dict = {0: "Non-stroke", 1: "Stroke"}


    return [f"Predicted class is {stroke_dict[predicted_class.item()]}", get_version()]