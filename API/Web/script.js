document.getElementById('predictNNButton').addEventListener('click', function() {
    var formData = {
        "features": {
            "Age": parseInt(document.getElementById('Age').value),
            "BusinessTravel": parseInt(document.getElementById('BusinessTravel').value),
            "DailyRate": parseInt(document.getElementById('DailyRate').value),
            "Department": parseInt(document.getElementById('Department').value),
            "DistanceFromHome": parseInt(document.getElementById('DistanceFromHome').value),
            "Education": parseInt(document.getElementById('Education').value),
            "EducationField": parseInt(document.getElementById('EducationField').value),
            "EmployeeCount": parseInt(document.getElementById('EmployeeCount').value),
            "EmployeeNumber": parseInt(document.getElementById('EmployeeNumber').value),
            "EnvironmentSatisfaction": parseInt(document.getElementById('EnvironmentSatisfaction').value),
            "Gender": parseInt(document.getElementById('Gender').value),
            "HourlyRate": parseInt(document.getElementById('HourlyRate').value),
            "JobInvolvement": parseInt(document.getElementById('JobInvolvement').value),
            "JobLevel": parseInt(document.getElementById('JobLevel').value),
            "JobRole": parseInt(document.getElementById('JobRole').value),
            "JobSatisfaction": parseInt(document.getElementById('JobSatisfaction').value),
            "MaritalStatus": parseInt(document.getElementById('MaritalStatus').value),
            "MonthlyIncome": parseInt(document.getElementById('MonthlyIncome').value),
            "MonthlyRate": parseInt(document.getElementById('MonthlyRate').value),
            "NumCompaniesWorked": parseInt(document.getElementById('NumCompaniesWorked').value),
            "Over18": parseInt(document.getElementById('Over18').value),
            "OverTime": parseInt(document.getElementById('OverTime').value),
            "PercentSalaryHike": parseInt(document.getElementById('PercentSalaryHike').value),
            "PerformanceRating": parseInt(document.getElementById('PerformanceRating').value),
            "RelationshipSatisfaction": parseInt(document.getElementById('RelationshipSatisfaction').value),
            "StandardHours": parseInt(document.getElementById('StandardHours').value),
            "StockOptionLevel": parseInt(document.getElementById('StockOptionLevel').value),
            "TotalWorkingYears": parseInt(document.getElementById('TotalWorkingYears').value),
            "TrainingTimesLastYear": parseInt(document.getElementById('TrainingTimesLastYear').value),
            "WorkLifeBalance": parseInt(document.getElementById('WorkLifeBalance').value),
            "YearsAtCompany": parseInt(document.getElementById('YearsAtCompany').value),
            "YearsInCurrentRole": parseInt(document.getElementById('YearsInCurrentRole').value),
            "YearsSinceLastPromotion": parseInt(document.getElementById('YearsSinceLastPromotion').value),
            "YearsWithCurrManager": parseInt(document.getElementById('YearsWithCurrManager').value)
        }
    };

    fetch("http://127.0.0.1:8000/predict/NN/", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        const prediction = data;
        document.getElementById('result').textContent = 'Prediction: ' + JSON.stringify(data);
        document.getElementById('result').textContent = 'Prediction: ' + JSON.stringify(prediction);
        const probabilityPercent = prediction.probability_1 * 100;
        const probabilityBar = document.getElementById('probabilityBar');
        probabilityBar.style.width = probabilityPercent + '%';
        probabilityBar.innerText = probabilityPercent.toFixed(2) + '%';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'Error: ' + error;
    });
});


document.getElementById('predictLRButton').addEventListener('click', function() {
    var formData = {
        "features": {
            "Age": parseInt(document.getElementById('Age').value),
            "BusinessTravel": parseInt(document.getElementById('BusinessTravel').value),
            "DailyRate": parseInt(document.getElementById('DailyRate').value),
            "Department": parseInt(document.getElementById('Department').value),
            "DistanceFromHome": parseInt(document.getElementById('DistanceFromHome').value),
            "Education": parseInt(document.getElementById('Education').value),
            "EducationField": parseInt(document.getElementById('EducationField').value),
            "EmployeeCount": parseInt(document.getElementById('EmployeeCount').value),
            "EmployeeNumber": parseInt(document.getElementById('EmployeeNumber').value),
            "EnvironmentSatisfaction": parseInt(document.getElementById('EnvironmentSatisfaction').value),
            "Gender": parseInt(document.getElementById('Gender').value),
            "HourlyRate": parseInt(document.getElementById('HourlyRate').value),
            "JobInvolvement": parseInt(document.getElementById('JobInvolvement').value),
            "JobLevel": parseInt(document.getElementById('JobLevel').value),
            "JobRole": parseInt(document.getElementById('JobRole').value),
            "JobSatisfaction": parseInt(document.getElementById('JobSatisfaction').value),
            "MaritalStatus": parseInt(document.getElementById('MaritalStatus').value),
            "MonthlyIncome": parseInt(document.getElementById('MonthlyIncome').value),
            "MonthlyRate": parseInt(document.getElementById('MonthlyRate').value),
            "NumCompaniesWorked": parseInt(document.getElementById('NumCompaniesWorked').value),
            "Over18": parseInt(document.getElementById('Over18').value),
            "OverTime": parseInt(document.getElementById('OverTime').value),
            "PercentSalaryHike": parseInt(document.getElementById('PercentSalaryHike').value),
            "PerformanceRating": parseInt(document.getElementById('PerformanceRating').value),
            "RelationshipSatisfaction": parseInt(document.getElementById('RelationshipSatisfaction').value),
            "StandardHours": parseInt(document.getElementById('StandardHours').value),
            "StockOptionLevel": parseInt(document.getElementById('StockOptionLevel').value),
            "TotalWorkingYears": parseInt(document.getElementById('TotalWorkingYears').value),
            "TrainingTimesLastYear": parseInt(document.getElementById('TrainingTimesLastYear').value),
            "WorkLifeBalance": parseInt(document.getElementById('WorkLifeBalance').value),
            "YearsAtCompany": parseInt(document.getElementById('YearsAtCompany').value),
            "YearsInCurrentRole": parseInt(document.getElementById('YearsInCurrentRole').value),
            "YearsSinceLastPromotion": parseInt(document.getElementById('YearsSinceLastPromotion').value),
            "YearsWithCurrManager": parseInt(document.getElementById('YearsWithCurrManager').value)
        }
    };

    fetch("http://127.0.0.1:8000/predict/LR/", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        const prediction = data;
        document.getElementById('result').textContent = 'LR Prediction: ' + JSON.stringify(data);
        const probabilityPercent = prediction.probability * 100;
        const probabilityBar = document.getElementById('probabilityBar');
        probabilityBar.style.width = probabilityPercent + '%';
        probabilityBar.innerText = probabilityPercent.toFixed(2) + '%';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').textContent = 'Error: ' + error;
    });
});


document.addEventListener('DOMContentLoaded', (event) => {
  document.getElementById('randomButton').addEventListener('click', fillRandomNumbers);
});

function fillRandomNumbers() {
  var inputs = document.querySelectorAll('#predictForm input[type="number"]');
  inputs.forEach(input => {
    let min = parseInt(input.min) || 0;
    let max = parseInt(input.max) || 100;
    input.value = Math.floor(Math.random() * (max - min + 1)) + min;
  });

  var selects = document.querySelectorAll('#predictForm select');

  selects.forEach(select => {
    let randomIndex = Math.floor(Math.random() * (select.options.length - 1)) + 1;
    select.selectedIndex = randomIndex;
  });
}

