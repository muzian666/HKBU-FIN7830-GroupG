from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib
from typing import Dict

# 定义请求的数据模型
class InputData(BaseModel):
    features: Dict[str, float]  # 假设每个特征都是浮点数

# 创建 FastAPI 实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 加载 ONNX 模型
model_path = r"C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\model.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 加载scaler
scaler_path = r"C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\scaler.joblib"
scaler = joblib.load(scaler_path)

# 定义softmax函数
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.post("/predict/")
async def generate_prediction(input_data: InputData):
    try:
        # 将输入数据转换为正确的格式
        # 假设我们接收到的数据字典是按照模型输入顺序排列的
        input_features = np.array(list(input_data.features.values())).reshape(1, -1)

        # 应用相同的预处理（标准化）
        input_features_scaled = scaler.transform(input_features)

        # 进行推理
        outputs = session.run([output_name], {input_name: input_features_scaled.astype(np.float32)})

        # 应用softmax函数来获取概率分布
        probabilities = softmax(outputs[0][0])

        # 获取概率，并转换为Python原生类型
        prob_class_0 = float(probabilities[0])
        prob_class_1 = float(probabilities[1])
        return {"probability_0": prob_class_0, "probability_1": prob_class_1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
