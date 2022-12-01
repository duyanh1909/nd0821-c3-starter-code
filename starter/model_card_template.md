# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Using random forest model with hyperparameters following:
- n_estimators=100,
- max_depth=15,
- min_samples_split=4,
- min_samples_leaf=3,
- max_features=0.5,
- random_state=0

## Intended Use
The random forest model is used to predict a person's salary based on that person's personal information.

## Training Data
Randomly taking 80% of the census dataset we get the training data. Source: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
Randomly taking 20% of the census dataset we get the evaluation data. Source: https://archive.ics.uci.edu/ml/datasets/census+income

## Metrics
This model is evaluated based on the measures of Precision(0.64), Recall(0.57), Fbeta(0.6)

## Ethical Considerations


## Caveats and Recommendations
