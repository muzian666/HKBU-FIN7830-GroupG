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

# Create FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all heads
)

# Load ONNX model
model_path = "./Resources/Model/model.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load scaler
scaler_path = "./Resources/Model/scaler.joblib"
scaler = joblib.load(scaler_path)

# Define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.post("/predict/NN/")
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

@app.post("/predict/LR/")
async def generate_lr_prediction(input_data: InputData):
    try:
        # 提取特征值
        features = input_data.features
        # 计算线性回归公式的结果
        result = (14.8900 - 0.3333 * features["BusinessTravel"] + 0.0387 * features["DistanceFromHome"]
                  - 0.2899 * features["Education"] - 0.6012 * features["EnvironmentSatisfaction"]
                  - 0.4115 * features["Gender"] - 0.8499 * features["JobInvolvement"]
                  - 0.7004 * features["JobLevel"] + 0.0935 * features["JobRole"]
                  - 0.5549 * features["JobSatisfaction"] + 0.1118 * features["NumCompaniesWorked"]
                  + 0.9542 * features["OverTime"] - 0.6318 * features["PerformanceRating"]
                  - 0.4653 * features["RelationshipSatisfaction"] - 1.0489 * features["StockOptionLevel"]
                  - 0.0549 * features["TotalWorkingYears"] - 0.3474 * features["TrainingTimesLastYear"]
                  - 0.6085 * features["WorkLifeBalance"] + 0.1193 * features["YearsAtCompany"]
                  - 0.1431 * features["YearsInCurrentRole"] + 0.1651 * features["YearsSinceLastPromotion"]
                  - 0.2030 * features["YearsWithCurrManager"])

        # 将线性回归结果转换为概率
        probability = 1 / (1 + np.exp(-result))

        return {"probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))