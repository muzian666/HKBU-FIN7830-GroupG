import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU.")
else:
    device = torch.device("cpu")
    print("Training on CPU.")

# 加载数据
print("加载数据...")
file_path = r'C:\Users\LQA\Desktop\class\HKBU-FIN7830-GP\Resources\Data\Encoded_Resampled_HR_Analytics.csv'
data = pd.read_csv(file_path)

# 数据预处理
print("数据预处理...")
X = data.drop('Attrition', axis=1).values
y = data['Attrition'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
print("拆分数据集...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)

# 定义逻辑回归模型
print("定义模型...")
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# 初始化模型和优化器
input_dim = X_train_tensor.shape[1]
model = LogisticRegressionModel(input_dim).to(device)
criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
print("训练模型...")
epochs = 100000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 预测和评估
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred_class = (y_test_pred > 0.5).float()

    accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_test_pred_class.cpu().numpy())
    roc_auc = roc_auc_score(y_test_tensor.cpu().numpy(), y_test_pred.cpu().numpy())
    f1 = f1_score(y_test_tensor.cpu().numpy(), y_test_pred_class.cpu().numpy())

print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}")
print(f"F1 Score: {f1}")

# 打印模型架构和参数
print("模型架构和参数：")
print(model)

# 打印逻辑回归方程
print("逻辑回归方程:")
for name, param in model.named_parameters():
    if name == 'linear.weight':
        print(f"权重: {param.data.cpu().numpy()}")
    else:
        print(f"偏置: {param.data.cpu().numpy()}")

# 获取特征名称
feature_names = data.columns[:-1]

# 获取模型权重
weights = model.linear.weight.data.cpu().numpy().flatten()

# 匹配特征名称和权重
feature_weights = dict(zip(feature_names, weights))

# 打印特征和相应的权重
print("特征与权重匹配：")
for feature, weight in feature_weights.items():
    print(f"{feature}: {weight}")

# 构建逻辑回归方程的字符串表示形式
equation_str = "ln(p / (1 - p)) = "
intercept = model.linear.bias.data.cpu().numpy()[0]  # 获取偏置项（截距）

# 添加偏置项到方程中
equation_str += f"{intercept:.4f} "

# 添加权重和特征名称
for feature, weight in feature_weights.items():
    sign = '+' if weight >= 0 else '-'
    equation_str += f"{sign} {abs(weight):.4f} * {feature} "

print("逻辑回归方程:")
print(equation_str)
