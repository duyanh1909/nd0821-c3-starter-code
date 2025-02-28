"""
Test ml model module.
"""

import joblib
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (train_model,
                      inference,
                      compute_model_metrics)


@pytest.fixture
def model():
    return joblib.load('./model/model.joblib')


@pytest.fixture
def lb():
    return joblib.load('./model/lb.joblib')


@pytest.fixture
def encoder():
    return joblib.load('./model/encoder.joblib')


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture(scope='function')
def df():
    return pd.read_csv('./data/cleaned_data.csv')


def test_train_model(df, cat_features):
    X, y, _, _ = process_data(df,
                              categorical_features=cat_features,
                              label='salary', training=True, encoder=encoder,
                              lb=lb)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference(model, encoder, lb, df, cat_features):
    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              label='salary', training=False, encoder=encoder,
                              lb=lb)

    preds = inference(model, X)
    assert len(preds) > 0


def test_compute_model_metrics(model, encoder, lb, df, cat_features):
    X, y, _, _ = process_data(df,
                              categorical_features=cat_features,
                              label='salary', training=False, encoder=encoder,
                              lb=lb)

    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
