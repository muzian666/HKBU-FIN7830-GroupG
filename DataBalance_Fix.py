from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import json

# 读取数据
df = pd.read_csv('HR_Analytics.csv.csv')

le_dict = {}

# 编码分类列
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# 保存编码后的数据集
df_encoded.to_csv('Encoded_HR_Analytics.csv', index=False)

# 收集编码映射信息
encoded_mappings = {}
for col in categorical_cols:
    le = le_dict[col]
    encoded_mappings[col] = list(le.classes_)

# 如果需要，保存编码映射信息为 JSON 文件
with open('encoded_mappings.json', 'w') as f:
    json.dump(encoded_mappings, f)

# 准备特征矩阵和目标向量
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# 应用SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 将重采样后的数据转换回DataFrame
X_res_df = pd.DataFrame(X_res, columns=X.columns)
y_res_series = pd.Series(y_res)

# 将数字标签转换回原始的 'Yes' 或 'No' 标签
le = le_dict['Attrition']
y_res_decoded = le.inverse_transform(y_res_series.astype(int))

# 保存编码和重采样后的数据集
encoded_resampled_df = pd.concat([X_res_df, y_res_series], axis=1)
encoded_resampled_df.to_csv('Encoded_Resampled_HR_Analytics.csv', index=False)

# 解码重采样后的数据
X_res_decoded = X_res_df.copy()
for col in categorical_cols:
    if col in X_res_decoded.columns:  # 确保该列存在
        le = le_dict[col]  # 获取之前用于这一列的LabelEncoder实例
        X_res_decoded[col] = le.inverse_transform(X_res_df[col].astype(int))

# 保存解码和重采样后的数据集
decoded_resampled_df = pd.concat([X_res_decoded, pd.Series(y_res_decoded, name='Attrition')], axis=1)
decoded_resampled_df.to_csv('Decoded_Resampled_HR_Analytics.csv', index=False)