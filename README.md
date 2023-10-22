# HKBU-FIN7830-GP
```python
# 读取数据集
import pandas as pd

# Load the dataset
file_path = 'HR_Analytics.csv.csv'
df = pd.read_csv(file_path)

# Show some basic information about the dataset
df.info(), df.head()
```

# 数据结构
由于数据偏差，考虑使用SMOTE算法平衡数据
## 使用第三方库进行平衡
```python
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables
label_columns = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for column in label_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Separate features and target variable
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the distribution of the target variable after SMOTE
y_resampled.value_counts()
```

## 无第三方库，纯算法平衡
```python
# Import LabelEncoder again and re-run the steps
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables again and separate features and target variable
df = pd.read_csv(file_path)
label_columns = df.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for column in label_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Apply manual SMOTE
target_label = 1  # 'Yes' label is encoded as 1
X_resampled, y_resampled = manual_SMOTE(X, y, target_label, random_state=42)

# Check the distribution of the target variable after SMOTE
y_resampled.value_counts()

```


# 神经网络
使用Tensorflow构建神经网络
```python
# 需要导入图的库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

``` python
# 定义模型结构
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```