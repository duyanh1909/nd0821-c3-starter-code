#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import pandas as pd


def basic_cleaning(path, list_label, output):
    """
    The basic cleaning function
    """
    df = pd.read_csv(path)
    # Remove spaces where put front of columns name.
    df.columns = [col.strip() for col in df.columns]
    # Drop columns from dataset
    df.drop(list_label, axis=1, inplace=True)
    # Select columns which have dtype as object.
    df_obj = df.select_dtypes(['object'])
    # Remove spaces where put front of value.
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    # Drop rows which any columns contain "?" character.
    df = df[(df != '?').all(1)]
    df.to_csv(output, index=False)


if __name__ == '__main__':
    path = './starter/data/census.csv'
    output = './starter/data/cleaned_data.csv'
    droped_label = [
        'capital-gain',
        'capital-loss'
    ]
    basic_cleaning(path, droped_label, output)
