import requests
import pandas as pd

# 假设的API URL，你需要替换成你自己的URL
api_url = "http://127.0.0.1:8000/predict/"

# 读取CSV文件
df = pd.read_csv(r'C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\Resources\Data\Encoded_Resampled_HR_Analytics.csv')  # 替换成你的CSV文件路径
df = df.drop('Attrition', axis=1)

column = 0
# 遍历CSV中的每一行
for index, row in df.iterrows():
    column += 1
    print(column)
    # 构造请求体，确保这里的字典键与API期待的输入匹配
    request_data = {"features": row.to_dict()}

    # 发送POST请求到API
    response = requests.post(api_url, json=request_data)

    # 检查响应状态码
    if response.status_code == 200:
        try:
            print(response.json())
        except ValueError:  # 包括json解码错误
            print("Response is not in JSON format.")
    else:
        print(f"Error: {response.status_code}, {response.text}")
