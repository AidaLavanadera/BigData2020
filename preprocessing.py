import pandas as pd
import numpy as np
def preprocessing():
    # Read input data
    train = pd.read_csv("train.csv")
    categorical = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

    #   Convert categorical data
    for c in categorical:
        group_by = train.groupby(by=c)["y"].mean().reset_index().rename(columns={"y": "{}_converted".format(c)})
        train = pd.merge(train, group_by, how='inner', on=c)

    train = train.drop(categorical, axis=1)

    #   Drop the ID column
    X = train.drop("ID", axis=1).drop("y", axis=1)
    y = train["y"]
    return X, y