# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
import logging

from starter.starter.ml.data import process_data
from starter.starter.ml.model import (train_model,
                                      inference,
                                      compute_model_metrics,
                                      compute_metrics_model_based_slice)

logging.basicConfig(level=logging.INFO,
                    filename="./starter/logs/slice_outputs.txt",
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

data = pd.read_csv("./starter/data/cleaned_data.csv")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)

joblib.dump(model, "./starter/model/model.joblib")
joblib.dump(encoder, "./starter/model/encoder.joblib")
joblib.dump(lb, "./starter/model/lb.joblib")

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info('All: precision: %s, recall: %s, fbeta: %s',
             precision, recall, fbeta)

for cate in cat_features:
    metrics = compute_metrics_model_based_slice(
        test, cate, cat_features, model, encoder, lb)

    for _, value in metrics.items():
        logging.info(
            "%s: precision: %s, recall: %s, fbeta: %s.",
            cate, value['precision'], value['recall'],
            value['fbeta'])
