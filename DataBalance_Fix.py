from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import json

# Load Data
df = pd.read_csv('Resources/Data/HR_Analytics.csv.csv')

le_dict = {}

# Encoded categorical column
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# Save the encoded dataset
df_encoded.to_csv('Encoded_HR_Analytics.csv', index=False)

# Gather encoding mapping information
encoded_mappings = {}
for col in categorical_cols:
    le = le_dict[col]
    encoded_mappings[col] = list(le.classes_)

# Save encoding mapping information as a JSON file
with open('encoded_mappings.json', 'w') as f:
    json.dump(encoded_mappings, f)

# Prepare feature matrix and target vector
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Convert resampled data back to DataFrame
X_res_df = pd.DataFrame(X_res, columns=X.columns)
y_res_series = pd.Series(y_res)

# Convert numeric labels back to original labels
le = le_dict['Attrition']
y_res_decoded = le.inverse_transform(y_res_series.astype(int))

# Save the encoded and resampled dataset
encoded_resampled_df = pd.concat([X_res_df, y_res_series], axis=1)
encoded_resampled_df.to_csv('Encoded_Resampled_HR_Analytics.csv', index=False)

# Decode resampled data
X_res_decoded = X_res_df.copy()
for col in categorical_cols:
    if col in X_res_decoded.columns:
        le = le_dict[col]
        X_res_decoded[col] = le.inverse_transform(X_res_df[col].astype(int))

# Save the decoded and resampled dataset
decoded_resampled_df = pd.concat([X_res_decoded, pd.Series(y_res_decoded, name='Attrition')], axis=1)
decoded_resampled_df.to_csv('Decoded_Resampled_HR_Analytics.csv', index=False)