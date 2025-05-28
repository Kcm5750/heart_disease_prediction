from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve custom UI at root "/"
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

# Prediction input model
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load model
model = joblib.load("model/model.joblib")

# Prediction endpoint
@app.post("/predict")
async def predict(data: HeartInput):
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    prediction = model.predict(features)[0]
    result = "High risk of heart disease" if prediction == 1 else "Low risk"
    return {"prediction": int(prediction), "result": result}
