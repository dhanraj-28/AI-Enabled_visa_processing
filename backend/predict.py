import joblib
import numpy as np

model = joblib.load("models/visa_model.pkl")

def predict_processing_time(country, visa_type, processing_center):
    features = np.array([[country, visa_type, processing_center]])
    prediction = model.predict(features)
    return int(prediction[0])
