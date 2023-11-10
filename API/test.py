import requests
import pandas as pd

api_url = "http://127.0.0.1:8000/predict/"

df = pd.read_csv(r'Resources/Data/Encoded_Resampled_HR_Analytics.csv')
df = df.drop('Attrition', axis=1)

column = 0

for index, row in df.iterrows():
    column += 1
    print(column)
    # Construct the request body, ensuring that the dictionary keys here match the input expected by the API
    request_data = {"features": row.to_dict()}

    # Send POST request to API
    response = requests.post(api_url, json=request_data)

    # Check response status code
    if response.status_code == 200:
        try:
            print(response.json())
        except ValueError:  # Include json decoding error
            print("Response is not in JSON format.")
    else:
        print(f"Error: {response.status_code}, {response.text}")
