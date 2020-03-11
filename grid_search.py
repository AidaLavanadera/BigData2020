# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold  # import KFold
from sklearn.metrics import r2_score
import json
@timeout(3600)
def grid_search(cv_splits, boosting_rounds):
    #   Read input data
    X, y = preprocessing()

    #   Create dataframe to collect the results
    tests_columns = ["test_nr", "cv_mean", "cv_min", "cv_max", "cv_median", "params"]
    test_id = 0
    tests = pd.DataFrame(columns=tests_columns)

    #   Cross validation number of splits
    kf = KFold(n_splits=cv_splits)

    #   Execute until timeout occurs
    #with timeout(RuntimeError):
    while(True):
        #   Get the grid
        grid_iter, keys, length = get_grid_iterable()
        try:    #   For every element of the grid
            for df_grid in grid_iter:
                        #   Prepare a list to collect the scores
                score = []
                params = dict(zip(keys, df_grid))

                        #   The objective function
                params["objective"] = "reg:squarederror"

                        #   For each fold, train XGBoost and spit out the results
                for train_index, test_index in kf.split(X.values):

                            #   Get X train and X test
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

                            #   Get y train and y test
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                            #   Convert into DMatrix
                    d_train = xgb.DMatrix(X_train, label=y_train)
                    d_valid = xgb.DMatrix(X_test, label=y_test)
                    d_test = xgb.DMatrix(X_test)
                    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

                            #   Create the classifier using the current grid params. Apply early stopping of 50 rounds
                    clf = xgb.train(params, d_train, boosting_rounds, watchlist, early_stopping_rounds=50,
                                        feval=xgb_r2_score, maximize=True, verbose_eval=10)
                    y_hat = clf.predict(d_test)

                            #   Append Scores on the fold kept out
                    score.append(r2_score(y_test, y_hat))

                        #   Store the result into a dataframe
                score_df = pd.DataFrame(columns=tests_columns, data=[
                    [test_id, np.mean(score), np.min(score), np.max(score), np.median(score),
                        json.dumps(dict(zip(keys, [str(g) for g in df_grid])))]])
                test_id += 1
                tests = pd.concat([tests, score_df])
                tests.to_csv("grid-search.csv", index=False)
                print(tests)
        except RuntimeError:
                    #   When timeout occurs an exception is raised and the main cycle is broken
            pass

    #   Spit out the results
    tests.to_csv("grid-search.csv", index=False)
    print(tests)
import itertools
grid_search(cv_splits=4, boosting_rounds=500)