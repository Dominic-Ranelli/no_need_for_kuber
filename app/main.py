from fastapi import FastAPI, status
from models.simple_model import fetch_pred_record


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    return {"Status": "Healthy"}

@app.post("/predict")
def post_prediction():
    return fetch_pred_record()
