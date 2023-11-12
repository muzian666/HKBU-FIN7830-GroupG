from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib
from typing import Dict

# Define the requested data model
class InputData(BaseModel):
    features: Dict[str, float]  # Assume each feature is a floating point number

# 创建 FastAPI 实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all heads
)

# Load ONNX model
model_path = r"C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\Resources\Model\model.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load scaler
scaler_path = r"C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\Resources\Model\scaler.joblib"
scaler = joblib.load(scaler_path)

# Define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.post("/predict/")
async def generate_prediction(input_data: InputData):
    try:
        # Convert input data to the correct format
        input_features = np.array(list(input_data.features.values())).reshape(1, -1)

        # Apply the same preprocessing (normalization)
        input_features_scaled = scaler.transform(input_features)

        # Inferences
        outputs = session.run([output_name], {input_name: input_features_scaled.astype(np.float32)})

        # Apply softmax function to obtain probability distribution
        probabilities = softmax(outputs[0][0])

        # Get the probability and convert it to Python native type
        prob_class_0 = float(probabilities[0])
        prob_class_1 = float(probabilities[1])
        return {"probability_0": prob_class_0, "probability_1": prob_class_1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
