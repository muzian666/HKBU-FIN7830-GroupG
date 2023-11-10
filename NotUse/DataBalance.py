import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '../Resources/Data/HR_Analytics.csv.csv'
df = pd.read_csv(file_path)

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

# Merge features and target variables into a new DataFrame
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df['Attrition'] = y_resampled

# Export balanced dataset to CSV file
balanced_df.to_csv('Balanced_HR_Analytics.csv', index=False)
