import requests


data = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "educationNum": 14,
    "maritalStatus": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "hoursPerWeek": 50,
    "nativeCountry": "United-States"
}


response = requests.post(url="https://mlops-udacity-project.herokuapp.com/",
                         json=data)
print(f"""status: {response.status_code}\ndata: {response.json()}""")
