from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to array
        features = np.array(list(data.values())).reshape(1, -1)

        prediction = model.predict(features)

        return {
            "prediction": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}