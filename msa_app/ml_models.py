from typing import Optional, Tuple
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
    "fit_intercept": [True, False],
    "positive": [True, False]
}

LogisticRegressionParams = {
    "penalty": ["l1", "l2", None],
    "C": uniform(0, 4)
}

SupportVectorClassifierParams = {
    'kernel': ['sigmoid', 'linear', 'rbf', 'poly'],
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'degree': randint(1, 10)
}

SupportVectorRegressionParams = {
    'kernel': ['sigmoid', 'linear', 'rbf', 'poly'],
    'C': uniform(0.1, 10),
    'gamma': ['scale', 'auto'],
    'degree': randint(1, 10)
}

models = {
    "Linear Regression": Model_Collection(LinearRegression, LinearRegressionParams),
    "Logistic Regression": Model_Collection(LogisticRegression, LogisticRegressionParams),
    "Supper Vector Regressor": Model_Collection(SVR, SupportVectorRegressionParams),
    "Support Vector Classifier": Model_Collection(SVC, SupportVectorClassifierParams)
}

def prepare_data(data_file_path: str, y_column: str, y_column_type: str, voxels_file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    data_file_path, data_file_extension = process_path(data_file_path)

    if data_file_extension == ".csv":
        data = pd.read_csv(data_file_path)
    else:
        data = pd.read_excel(data_file_path)

    
    X = data.drop(y_column, axis=1)

    if voxels_file_path:
        voxels_file_path, voxels_file_extension = process_path(voxels_file_path)
        if voxels_file_extension == ".csv":
            voxels = pd.read_csv(voxels_file_path, header=None)
        else:
            voxels = pd.read_excel(voxels_file_path, header=None)
        voxels = pd.Series(voxels[1].values, index=voxels[0])

        assert set(X.columns) == set(voxels.index), "Brain Regions in Datafile are different from brain regions in Voxel file"
    else:
        voxels = pd.Series(np.ones(len(X.columns)), index=X.columns)


    if 'rob' not in X.columns.str.lower():
        X['rob'] = 0
        voxels['rob'] = 0

    mask = X > X.median(0)
    X.where(mask, 1, inplace=True)
    X.where(~mask, 0, inplace=True)
    y = data[y_column]

    y = y.max() - y if y_column_type=="NIHSS Score" else y

    return X, y, voxels

def process_path(data_file_path):
    data_file_path = os.path.normpath(data_file_path)
    data_file_extension = os.path.splitext(data_file_path)[1]
    if data_file_extension not in (".csv", ".xlsx"):
        raise RuntimeError("The file specified is not CSV or xlsx. Please use a CSV or xlsx file instead")
    return data_file_path,data_file_extension

def train_model(model_name : str, X: np.ndarray, y: np.ndarray, n_iter: int = 32):
    model_collection = models[model_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    opt = RandomizedSearchCV(model_collection.model_class(), model_collection.hyperparameters, cv=5, n_iter=400)
    opt.fit(X_train.values, y_train)
    y_pred = np.rint(opt.predict(X_test.values))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"), opt
