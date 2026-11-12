import logging
import uuid
from contextvars import ContextVar
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
import joblib

# Context Var for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


# Logging filter
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


# Logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
logger.addFilter(RequestIdFilter())


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


# Adding hints to base model
class Input(BaseModel):
    gender: Literal["Male", "Female", "Other"] = Field(
        ..., description="Gender of the patient", examples=["Male", "Female"]
    )
    age: int = Field(
        ge=0, le=120, description="Age of patient (0 to 120)", examples=[67, 3]
    )
    avg_glucose_level: float = Field(
        ge=0,
        le=2000,
        description="Average blood glucose level",
        examples=[89.0, 1000.45],
    )
    bmi: float = Field(
        ..., description="Body Mass Index of the patient", examples=[23.9, 97]
    )
    work_type: str = Field(
        ...,
        description="Employment category(e.g., Govt_job, Private, Self-employed)",
        examples=["Govt_job"],
    )
    ever_married: Literal["Yes", "No"] = Field(
        ..., description="If patient has been married", examples=["Yes", "No"]
    )
    residence_type: Literal["Urban", "Rural"] = Field(
        ..., description="Type of living area", examples=["Rural"]
    )
    smoking_status: str = Field(
        ...,
        description="Smoking habit (e.g., formerly smoked, never smoked, smokes, Unknown)",
        examples=["Unknown"],
    )


@app.get("/")
async def root():
    return {"Message": "Hello World"}


# Adding load time request
@app.get("/load")
async def load_time(start=load_time_start, end=load_time_end):
    logger.info(f"Received prediction request time length...")
    return {"Load time of model:": f"{(end - start) * 1000} milliseconds"}


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    logger.info(f"Received health request")
    return {"Status": "Healthy"}


@app.get("/version")
def get_version():
    logger.info(f"Received version requst...")
    return {"Model version": "1.0.0"}


@app.post("/predict")
def post_prediction(input: Input):
    # Logging info
    logger.info(f"Received prediction request for input payload: {input.dict()}")
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

    # Adding better outputs
    stroke_dict = {0: "Non-stroke", 1: "Stroke"}
    result_class = stroke_dict[predicted_class.item()]
    duration_ms = (run_time_end - run_time_start) * 1000

    # Logs
    logger.info(
        f"Prediction successful: {result_class} (Inference time: {duration_ms:.2f} ms)"
    )

    return [
        f"Predicted class is {result_class}",
        f"Ran in {round(duration_ms, 2)} milliseconds",
    ]
