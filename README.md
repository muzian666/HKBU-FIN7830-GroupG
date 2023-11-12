# HKBU FIN7830 GroupG 2023/11/15

 <img src="./Resources/IMG/img.png" width = "1792" height = "1024" alt="Front-Image" align=center />

# Project Structure Explain
```
├─API (API Relate File)
├─NotUse (Processing File, Ignore)
└─Resources
    ├─Data
    │  └─WebData
    ├─IMG
    └─Model
```

## How to start API
```
uvicorn API:app --host 0.0.0.0 --port 8000 --reload
```

## Dataset Citation
```
Karanth, M. (2020). Tabular summary of HR analytics dataset. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4088439
```

## /Resources/Data 文件说明
原数据集：[HR_Analytics.csv.csv](Resources%2FData%2FHR_Analytics.csv.csv)

编码后原数据集：[Encoded_HR_Analytics.csv](Resources%2FData%2FEncoded_HR_Analytics.csv)

平衡后编码数据集：[Encoded_Resampled_HR_Analytics.csv](Resources%2FData%2FEncoded_Resampled_HR_Analytics.csv)

平衡后解码数据集：[Decoded_Resampled_HR_Analytics.csv](Resources%2FData%2FDecoded_Resampled_HR_Analytics.csv)

数据集需要编解码的原因：
1. 原数据集中包含<kbd>Object</kbd>类项目，不能直接拿来建模（逻辑回归或神经网络）
2. SMOTE算法不支持对<kbd>Object</kbd>类进行扩充，需要转为<kbd>int</kbd>类
3. 为了方便进行可视化，需要对编码后的数据集进行解码，因为可视化中包含了<kbd>Object</kbd>的相关类

# 数据结构
## 数据平衡
由于数据偏差，考虑使用SMOTE算法平衡数据
### 使用第三方库进行平衡
参见<kbd>/DataBalance_Fix.py</kbd>

### 无第三方库，纯算法平衡
参见<kbd>/NotUse/SMOTE-Pure-Algrithm.py</kbd>

# 数据可视化结果
## 原数据-数值型变量可视化
![avatar](Resources/IMG/RawDataBasic1.png)
## 原数据-分类型变量可视化
![avatar](Resources/IMG/RawDataBasic2.png)
## 原数据-数值型变量与Attrition的关系
![avatar](Resources/IMG/RawDataRelation1.png)
## 原数据-分类型变量与Attrition的关系
![avatar](Resources/IMG/RawDataRelation2.png)
## 平衡后数据-数值型变量可视化
![avatar](Resources/IMG/BalanceDataBasic1.png)
## 平衡后数据-分类型变量可视化
![avatar](Resources/IMG/BalanceDataBasic2.png)
## 平衡后数据-数值型变量与Attrition的关系
![avatar](Resources/IMG/BalanceDataRelation1.png)
## 平衡后数据-分类型变量与Attrition的关系
![avatar](Resources/IMG/BalanceDataRelation2.png)

# 逻辑回归
## Statsmodel
Wait for result



# 神经网络
使用Pytorch构建神经网络
```python
# 安装CPU版Pytorch
pip3 install torch torchvision torchaudio
# 安装GPU(CUDA12)版Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
模型架构：
```python
class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(34, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
```
基于该模型架构，实现了平均90.551%的准确率，详细信息如下：
```
optimizer = optim.Adam(model.parameters(), lr=0.001)
Average Accuracy: 0.9055078795443908
Average F1 Score: 0.9074476653232741
Average ROC AUC: 0.9054820446989895
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
Linear-1                [-1, 1, 64]           2,240
Dropout-2                [-1, 1, 64]               0
Linear-3               [-1, 1, 128]           8,320
Dropout-4               [-1, 1, 128]               0
Linear-5               [-1, 1, 256]          33,024
Linear-6               [-1, 1, 128]          32,896
Linear-7                [-1, 1, 64]           8,256
Linear-8                 [-1, 1, 2]             130
================================================================
Total params: 84,866
Trainable params: 84,866
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.01
Params size (MB): 0.32
Estimated Total Size (MB): 0.33
----------------------------------------------------------------
```