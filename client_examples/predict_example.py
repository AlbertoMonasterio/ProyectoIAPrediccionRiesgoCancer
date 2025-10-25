import requests

URL = "http://127.0.0.1:8000/predict"

payload = {
    "age": 55,
    "bmi": 27.5,
    "liver_function_score": 1.2,
    "alpha_fetoprotein_level": 3.4,
    "hepatitis_b": 0,
    "hepatitis_c": 0,
    "cirrhosis_history": 0,
    "family_history_cancer": 0,
    "diabetes": 0,
    "gender": "male",
    "alcohol_consumption": "low",
    "smoking_status": "never",
    "physical_activity_level": "moderate",
}

r = requests.post(URL, json=payload)
print(r.status_code, r.text)
