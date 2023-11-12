# Original Model
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set a DataFrame as df and load the csv file "Encoded_Resampled_HR_Analytics"
df = pd.read_csv(r'Resources/Data/Encoded_Resampled_HR_Analytics.csv')

# Define Independent Variables
X = df.drop('Attrition', axis=1)  # remove the dependent variable
# Define Dependent Variable
y = df['Attrition']

# add an intercept column to the features
X = sm.add_constant(X)

# Delete those variables, that have same value, in X
# Delete EmployeeCount, Over18 and StandardHours
for col in X.columns:
        if X[col].nunique() == 1:
                print(f"Deleting {col}")
                del X[col]

# split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# add an intercept column to the features 
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# create and initialize a Logistic regression and fit the model with the data
model = sm.Logit(y_train, X_train)
result = model.fit()

# Print the result summary
print(result.summary())

# Prediction of dependent variable
y_pred = result.predict(X_test)
y_pred_label = (y_pred > 0.5).astype(int)

# Calculate and print the result of the performance indicators
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred_label)
print(f'AUC: {roc_auc:.4f}')
print(f'Confusion Matrix:\n{cm}')

# Calculate and print the accuracy of the model
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
print(f'Accuracy: {accuracy}')
# AUC = 0.9105, Accuracy = 0.8357

#%%
# Improvement
# Checking Multicollinearity by calculating variance-inflation-factor (VIF)
# of each variable, and remove those variables with high VIF
# VIF > 10 means collinearity exists in the independent variable
# delete those independent variables that VIF larger than 10
# delete independent variables: Age, Department, HourlyRate, JobInvolvement,
# JobLevel, MonthlyIncome, PercentSalaryHike, PerformanceRating,
# TotalWorkingYears, WorkLifeBalance and YearsAtCompany
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.drop(['Attrition', 'Age', 'Department', 'HourlyRate', 'JobInvolvement',
            'JobLevel', 'MonthlyIncome', 'PercentSalaryHike', 'PerformanceRating',
            'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany'], axis=1)
y = df['Attrition']

# add an intercept column to the features for the set of independent variable
X = sm.add_constant(X)

# Delete those variables, that is with same value, in X
# Delete EmployeeCount, Over18 and StandardHours
for col in X.columns:
        if X[col].nunique() == 1:
                print(f"Deleting {col}")
                del X[col]
                
# Checking Multicollinearity by calculating variance-inflation-factor
# of each variable
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# split data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# add intercept columns to the features 
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# create and initialize a Logistic regression and fit the model with the data
model = sm.Logit(y_train, X_train)
result = model.fit()

# Print the result summary
print(result.summary())

# Prediction of dependent variable
y_pred = result.predict(X_test)
y_pred_label = (y_pred > 0.5).astype(int)

# Calculate and print the result of the performance indicators
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred_label)
print(f'AUC: {roc_auc:.4f}')
print(f'Confusion Matrix:\n{cm}')

# Calculate and print the accuracy of the model
accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
print(f'Accuracy: {accuracy}')

# AUC = 0.8728, Accuracy = 0.7911

# after we ran the improvement, we found out that the AUC is 0.8728 and 
# the accuracy is 0.7911, which is lower than what we original model's
# AUC and accuracy. Hence, we reject the improvement that we tried in this
# section.
