from fastapi import FastAPI, status
from models.simple_model import fetch_pred_record
from pydantic import BaseModel, Field
from typing import Optional

class Input(BaseModel):
    name: str
    age: int  = Field(ge=0, le=150, description="Age must come between 0 and 150")
    eye_color: str | None = None
    salary: Optional[float] = Field(ge=0, description="salary must be positive")
    state: str | None = None


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    return {"Status": "Healthy"}

@app.get("/version")
def get_version():
    return {"Version": "1.0.0"}

@app.post("/predict")
def post_prediction(input: Input):
    return [fetch_pred_record(input), get_version()]
