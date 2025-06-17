import joblib
import numpy as np

scaler = joblib.load("model/scaler.joblib")

def preprocess_input(data):
    input_array = np.array([
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ])
    input_scaled = scaler.transform([input_array])
    return input_scaled[0]
