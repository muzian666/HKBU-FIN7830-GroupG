# HKBU FIN7830 Group G 2023/11/15

 <img src="./Resources/IMG/ProjectIMG.png" alt="Front-Image" align=center />

# Abstract
In order to study the influencing factors of the company's employee turnover rate, we grabbed a human resource information dataset on Zenodo, and analyzed the degree of influence on the factors of "distance from home", "number of companies worked", "number of years in the company", and "number of years since the last promotion", "business trip", "department", "field of education", "gender", "job role", and "overtime" on the turnover rate of the employees. Using SMOTE algorithm to preprocess and improve the data, and adopting two methods of logistic regression and neural network respectively, it was concluded that factors such as "distance from home", "job role", "number of companies worked for", "number of years in the company", and "number of years since the last promotion" have a greater impact on the employee separation rate, and relevant suggestions were made for the important factors.

# Introduction
Employee turnover is a significant factor in company performance. A high turnover rate not only increases the monetary and time costs of recruiting and training new employees for the company, but also may cause delays or interruptions in work assignments, affecting productivity, as well as team morale and company image. At the same time, we used a crawler to collect discussions about leaving jobs on the Hong Kong forum uwants, and wanted to briefly understand people's thoughts and reasons for leaving jobs by analyzing the word cloud of the forum's discussion content, and the results showed that the keywords "time", "company The results show that the keywords "time", "company", "go", "work", "team", "leave The discussion of "company", "go", "work", "team", "leave", etc. is very hot. Separation issues and high turnover rates are hot issues in the current socio-economic situation. Therefore, it is of great significance for enterprise development to analyze and understand in detail the important factors affecting the departure of employees and take appropriate measures to reduce the turnover rate.

![WordCloud.png](Resources%2FIMG%2FWordCloud.png)

# Business Problem
What factors are associated with employee turnover in a company?

There may be various reasons for employees to leave their jobs, but the root of the problem is that the individual's expectations of the job do not match the company's current situation.

a.Salary level and welfare benefits, whether the company has a perfect salary system and a scientific and reasonable salary and welfare system.

b.Promotion opportunities and career development, whether the employee's work ability, work experience and position are suitable, whether the company has a clear and enforceable promotion channel for employees and an employee training and development program.

c.Work environment and atmosphere, including the management ability and management style of company leaders, company culture and values.

d.Work content and challenges, whether the work can stimulate the enthusiasm of employees to work, to help employees realize their personal values.

e.Work-life balance, including the intensity of the work, overtime work, the distance of the commute, which can affect the employee's satisfaction with the job

# Detail of this Project
## Project Introduction
```
This project is for HKBU FIN7830 Financial Computing with Python (Section 1) Group G
This project is written by following person:

23420820 | Chen, Ziwei
23473193 | Cheng, Hoi Laam
23410868 | Hunag, Chengyi
23417145 | Ji, Yanbo
23447214 | Li, Qingan
23461608 | Qu, Yibo
23423927 | Wang, Jiayi
23417056 | Xie, Siqi

Ranking in no particular order, all the authors contributed to this project equally.
```
## Project Structure
```
├─Abstract
├─Introduction (Motivation)
├─Business Problem
├─Data Description
│   ├─Data Structure & Pre Process
│   └─Data Visualization
├─Data Analysis
│   ├─Logical Regression Model
│   │  ├─Model Build
│   │  ├─Temporary Result
│   │  ├─Verfication
│   │  ├─Improvement
│   │  ├─Final Result
│   │  ├─Analysis Result
│   │  └─Recommendation
│   ├─Neural Network
│   │  ├─Model Build & Result
│   └─ └─Application
├─Conclusion
└─Reference
```

## Environment package dependencies
```
See requirements.txt for detail

Install all package by using:
cd <The dir to this project>
pip3 install -r requirements.txt
```

## Dataset Citation
```
Karanth, M. (2020). Tabular summary of HR analytics dataset. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4088439
```

## Description of <kbd>./Resources/Data</kdb>
Raw Data：[HR_Analytics.csv.csv](Resources%2FData%2FHR_Analytics.csv.csv)

Data after Encode：[Encoded_HR_Analytics.csv](Resources%2FData%2FEncoded_HR_Analytics.csv)

Resample Data after Encode：[Encoded_Resampled_HR_Analytics.csv](Resources%2FData%2FEncoded_Resampled_HR_Analytics.csv)

Resample Data after Decode：[Decoded_Resampled_HR_Analytics.csv](Resources%2FData%2FDecoded_Resampled_HR_Analytics.csv)

Reasons why the data set needs encoding and decoding:
1. The original data set contains <kbd>Object</kbd> items and cannot be directly used for modeling (logistic regression or neural network)
2. The SMOTE algorithm does not support the expansion of the <kbd>Object</kbd> class and needs to be converted to the <kbd>int</kbd> class.
3. In order to facilitate visualization, the encoded data set needs to be decoded, because the visualization contains related classes of <kbd>Object</kbd>

## Mapping for Decode and Encode
```json
{
  "Attrition": ["No", "Yes"],
  "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
  "Department": ["Human Resources", "Research & Development", "Sales"],
  "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
  "Gender": ["Female", "Male"],
  "JobRole": ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager", "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"],
  "MaritalStatus": ["Divorced", "Married", "Single"],
  "Over18": ["Y"],
  "OverTime": ["No", "Yes"]
}
```

## Data Visualization
### Original data-numeric variable visualization
![avatar](Resources/IMG/RawDataBasic1.png)
### Original data - visualization of categorical variables
![avatar](Resources/IMG/RawDataBasic2.png)
### Original data - relationship between numerical variables and Attrition
![avatar](Resources/IMG/RawDataRelation1.png)
### Original data - relationship between categorical variables and Attrition
![avatar](Resources/IMG/RawDataRelation2.png)
### Balanced data-numeric variable visualization
![avatar](Resources/IMG/BalanceDataBasic1.png)
### Balanced data-visualization of categorical variables
![avatar](Resources/IMG/BalanceDataBasic2.png)
### Balanced data - relationship between numerical variables and Attrition
![avatar](Resources/IMG/BalanceDataRelation1.png)
### Post-balanced data - relationship between categorical variables and Attrition
![avatar](Resources/IMG/BalanceDataRelation2.png)
### Correlation Heat Map
![CorrelationHeadMap.png](Resources%2FIMG%2FCorrelationHeadMap.png)
### Clustered Scatterplot
![swarmplot.png](Resources%2FIMG%2Fswarmplot.png)

## Logical Regression
### Statsmodel Build
```python
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
```

### Result of variables
![LRR1.png](Resources%2FIMG%2FLRR1.png)
![LRR2.png](Resources%2FIMG%2FLRR2.png)

### Result of Function
![LRF.png](Resources%2FIMG%2FLRF.png)


## Neural Network
Using Pytorch to build Neural Network
```python
# 安装CPU版Pytorch
pip3 install torch torchvision torchaudio
# 安装GPU(CUDA12)版Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Model Structure：
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
Based on this model architecture, an average accuracy of 90.551% was achieved. The details are as follows:
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
### Training Loss
![NNLossOutput.png](Resources%2FIMG%2FNNLossOutput.png)

### Model Structure
![ONNXModelSimulate.png](Resources%2FIMG%2FONNXModelSimulate.png)

## Application
### How to start API
```
cd <dir to project>
uvicorn API:app --host 0.0.0.0 --port 8000 --reload
```
### Single Post Sample
See: [test1.py](API%2Ftest1.py)


# Conclusion
Combining the results of the sample data analysis derived from the Logical Regression, Stock Option Level has a more significant impact on the employee turnover rate on the negative side, and Over Time has a more significant impact on the positive side. Therefore, we give the following optimization suggestions for these influencing factors:

A.Strengthen enterprise management, sound management system, such as performance appraisal system, training programs, etc., to help improve the quality and efficiency of employees' work.

B. Improve the salary management system, set fair and reasonable salaries, and set stock option incentives to enhance employees' sense of belonging to the company and development expectations.

C. Improve the company's career development system, open up promotion channels for employees, and pay timely attention to employees' ability level, work experience, length of service, age, and their career development and the matching of corresponding positions.

D. Develop a complete and clear employee training program, and pay attention to the career development and personal ability of employees.

E. Pay attention to the mental health and working condition of employees, require all departments to use labor reasonably, combine work and leisure, and ensure that employees have a reasonable range of working hours and a comfortable office environment.
