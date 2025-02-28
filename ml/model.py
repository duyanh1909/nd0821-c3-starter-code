"""
The file consist of 4 function to evaluate model,
inference and computing metrics
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=15,
                                   min_samples_split=4,
                                   min_samples_leaf=3,
                                   max_features=0.5,
                                   random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_model_based_slice(df,
                                      cate,
                                      cat_features,
                                      model,
                                      encoder,
                                      lb):
    """
    Validates the trained machine learning model
    using precision, recall, and F1 based on slice.
    """
    dict_preds = {}
    for value in df[cate].unique():
        df_cate = df[df[cate] == value]
        X, y, _, _ = process_data(df_cate,
                                  cat_features,
                                  label='salary',
                                  training=False,
                                  encoder=encoder,
                                  lb=lb)

        preds = model.predict(X)

        precision, recall, fbeta = compute_model_metrics(y, preds)
        dict_preds[value] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta}
    return dict_preds
