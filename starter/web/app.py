import os
import pandas as pd
from joblib import load

from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import inference as predict


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class User(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    educationNum: int
    maritalStatus: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hoursPerWeek: int
    nativeCountry: str

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


app = FastAPI()
MODEL = load("./starter/model/model.joblib")
ENCODER = load("./starter/model/encoder.joblib")
LB = load("./starter/model/lb.joblib")
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@app.get("/")
async def get():
    return {"message": "Greeting!"}


@app.post("/")
async def inference(user: User):
    body = [[user.age,
             user.workclass,
             user.fnlgt,
             user.education,
             user.educationNum,
             user.maritalStatus,
             user.occupation,
             user.relationship,
             user.race,
             user.sex,
             user.hoursPerWeek,
             user.nativeCountry]]

    df_temp = pd.DataFrame(data=body, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
        df_temp,
        categorical_features=CAT_FEATURES,
        encoder=ENCODER, lb=LB, training=False)
    pred = predict(MODEL, X)
    pred_label = LB.inverse_transform(pred)[0]
    return {"predict": pred_label}
