from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC, SVR
from scipy.stats import randint, uniform
import os
import pandas as pd
from collections import namedtuple

Model_Collection = namedtuple("Model_Collection", "model_class hyperparameters")

LinearRegressionParams = {
    "fit_intercept": [True, False]
}

LogisticRegressionParams = {
    "penalty": ["l1", "l2", None],
    "C": uniform(0.1, 10)
}

SupportVectorClassifierParams = {
    "kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree" : randint(1, 7),
}

SupportVectorRegressionParams = {
    "kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree" : randint(1, 7),
}

models = {
    "Linear Regression": Model_Collection(LinearRegression, LinearRegressionParams),
    "Logistic Regression": Model_Collection(LogisticRegression, LogisticRegressionParams),
    "Supper Vector Regressor": Model_Collection(SVR, SupportVectorRegressionParams),
    "Support Vector Classifier": Model_Collection(SVC, SupportVectorClassifierParams)
}

def prepare_data(path_to_data: str, y_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    path_to_data = os.path.normpath(path_to_data)
    file_extension = os.path.splitext(path_to_data)[1]

    if file_extension == ".csv":
        data = pd.read_csv(path_to_data)
    elif file_extension == ".xlsx":
        data = pd.read_excel(path_to_data)
    else:
        raise RuntimeError("The data file specified is not CSV. Please use a CSV file instead")
    
    X = data.drop(y_column, axis=1)
    mask = X > X.median(0)
    X.where(mask, 1, inplace=True)
    X.where(~mask, 0, inplace=True)
    y = data[y_column]

    return X, y

def train_model(model_name : str, X: np.ndarray, y: np.ndarray, n_iter: int = 32):
    model_collection = models[model_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    opt = RandomizedSearchCV(model_collection.model_class(), model_collection.hyperparameters, cv=4, n_iter=200)
    opt.fit(X_train.values, y_train)
    y_pred = np.rint(opt.predict(X_test.values))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"), opt
