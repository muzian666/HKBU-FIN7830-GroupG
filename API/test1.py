import requests

api_url = "http://127.0.0.1:8000/predict/"

# Use the data in data_head to construct the request body
request_data = {
    "features": {
        "Age": 20,
        "BusinessTravel": 2,
        "DailyRate": 1102,
        "Department": 2,
        "DistanceFromHome": 1,
        "Education": 2,
        "EducationField": 1,
        "EmployeeCount": 1,
        "EmployeeNumber": 1,
        "EnvironmentSatisfaction": 2,
        "Gender": 0,
        "HourlyRate": 94,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": 7,
        "JobSatisfaction": 4,
        "MaritalStatus": 2,
        "MonthlyIncome": 5993,
        "MonthlyRate": 19479,
        "NumCompaniesWorked": 8,
        "Over18": 0,
        "OverTime": 1,
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 0,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 6,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 5
    }
}

# Send POST request to API
response = requests.post(api_url, json=request_data)

# Print response
print(response.json())
