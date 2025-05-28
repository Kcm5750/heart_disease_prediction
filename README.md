# Heart Disease Prediction API (Localhost)

This project builds an ensemble machine learning model to predict heart disease and serves it via a FastAPI app running locally.

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python model/train_model.py
   ```

3. Start the API server:
   ```bash
   uvicorn app:app --reload
   ```

4. Send a POST request to:
   ```
   http://localhost:8000/predict
   ```

   Example JSON body:
   ```json
   {
       "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
       "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
       "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
   }
   ```

5. Output:
   ```json
   {
     "prediction": 1,
     "probability": 0.89
   }
   ```