from fastapi import FastAPI
from pydantic import BaseModel
from backend.predict import predict_processing_time

app = FastAPI()

class VisaInput(BaseModel):
    country: int
    visa_type: int
    processing_center: int

@app.post("/predict")
def predict(data: VisaInput):
    result = predict_processing_time(
        data.country,
        data.visa_type,
        data.processing_center
    )
    return {"estimated_processing_days": result}
