import pandas as pd
# Read 4 datasets (data1=Original Dataset, data2=Encoded Dataset, data3=Encoded & Resampled Dataset, data4=Decoded & Resampled Dataset)
data1=pd.read_csv('HR_Analytics.csv')
data2=pd.read_csv('Encoded_HR_Analytics.csv')
data3=pd.read_csv('Encoded_Resampled_HR_Analytics.csv')
data4=pd.read_csv('Decoded_Resampled_HR_Analytics.csv')

# Analyze the range of values for each variable in the datasets
variable_ranges = data1.agg(['min', 'max']).transpose()
variable_ranges

variable_ranges = data2.agg(['min', 'max']).transpose()
variable_ranges

variable_ranges = data3.agg(['min', 'max']).transpose()
variable_ranges

variable_ranges = data4.agg(['min', 'max']).transpose()
variable_ranges

# Analyze the data size (rows, columns) of all variables in the datasets
num_variables = data1.shape
print(num_variables)

num_variables = data2.shape
print(num_variables)

num_variables = data3.shape
print(num_variables)

num_variables = data4.shape
print(num_variables)

# Write down all the independent variables (34) and dependent variable (Only 1: 'Attrition') in data1 or data2 or data3 or data4 (All are the same)
variable_names = data1.columns.tolist()
print(variable_names)

variable_names = data2.columns.tolist()
print(variable_names)

variable_names = data3.columns.tolist()
print(variable_names)

variable_names = data4.columns.tolist()
print(variable_names)












