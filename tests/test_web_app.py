"""
Test web API module
"""


import pytest
from fastapi.testclient import TestClient
from web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Greeting!"}


def test_post_larger_50(client):
    response = client.post("/", json={
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "educationNum": 9,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 45,
        "nativeCountry": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"predict": ">50K"}


def test_post_smaller_50(client):
    response = client.post("/", json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "educationNum": 13,
        "maritalStatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert response.status_code == 200
    assert response.json() == {"predict": "<=50K"}
