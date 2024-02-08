from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
from scipy.stats import randint, uniform
import os
import pandas as pd
from collections import namedtuple

from typeguard import typechecked

Model_Collection = namedtuple("Model_Collection", "model_class hyperparameters")

LinearRegressionParams = {"fit_intercept": [True, False], "positive": [True, False]}

LogisticRegressionParams = {"penalty": ["l1", "l2", None], "C": uniform(0, 4)}

SupportVectorClassifierParams = {
    "kernel": ["sigmoid", "linear", "rbf", "poly"],
    "C": uniform(0.1, 10),
    "gamma": ["scale", "auto"],
    "degree": randint(1, 10),
}

SupportVectorRegressionParams = {
    "kernel": ["sigmoid", "linear", "rbf", "poly"],
    "C": uniform(0.1, 10),
    "gamma": ["scale", "auto"],
    "degree": randint(1, 5),
}

DecisionTreeRegressorParams = {
    "max_depth": [None] + list(range(1, 20)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", None],
}

RandomForestRegressorParams = {
    "n_estimators": randint(1, 100),
    "max_features": ["sqrt", "log2", None],
    "max_depth": randint(1, 8),
    "min_samples_split": randint(1, 10),
    "min_samples_leaf": randint(1, 10),
    "bootstrap": [True, False],
}

models = {
    "Linear Regression": Model_Collection(LinearRegression, LinearRegressionParams),
    "Logistic Regression": Model_Collection(
        LogisticRegression, LogisticRegressionParams
    ),
    "Supper Vector Regressor": Model_Collection(SVR, SupportVectorRegressionParams),
    "Support Vector Classifier": Model_Collection(SVC, SupportVectorClassifierParams),
    "Decision Tree Regressor": Model_Collection(
        DecisionTreeRegressor, DecisionTreeRegressorParams
    ),
    "Random Forest Regressor": Model_Collection(
        RandomForestRegressor, RandomForestRegressorParams
    ),
}


@typechecked
def prepare_data(
    data_file_path: str,
    y_column: str,
    y_column_type: str,
    voxels_file_path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare the data for model training and evaluation.

    Parameters:
        data_file_path (str): The path to the data file. The file can be in CSV or xlsx format. Data file contains the percentage of alteration of each brain ROI
        y_column (str): The name of the column containing the target variable. It could be NIHSS Score, or the performance
        y_column_type (str): The type of the target variable. Possible values are "NIHSS Score" or "Performance".
        voxels_file_path (str): The path to the voxels file. The file can be in CSV or xlsx format. If not provided, the number of voxels for each ROI will be set to 1.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the following:
            - X: The feature matrix as a pandas DataFrame.
            - y: The target variable as a pandas Series.
            - voxels: The voxels information as a pandas Series.

    Raises:
        RuntimeError: If the data file extension is not CSV or xlsx.
        AssertionError: If the data values are not between 0 and 1.
        AssertionError: If the brain regions in the data file are different from the brain regions in the voxels file.
        AssertionError: If the y_column does not exist in the data file.

    """
    data_file_path, data_file_extension = process_path(data_file_path)

    if data_file_extension == ".csv":
        data = pd.read_csv(data_file_path)
    else:
        data = pd.read_excel(data_file_path)

    X = data.drop(y_column, axis=1) / 100

    assert (
        np.max(X.values) <= 1 and np.min(X.values) >= 0
    ), "Data should be percentage of alteration in ROI, i.e. between 0 and 100"

    if voxels_file_path:
        voxels_file_path, voxels_file_extension = process_path(voxels_file_path)
        if voxels_file_extension == ".csv":
            voxels = pd.read_csv(voxels_file_path, header=None)
        else:
            voxels = pd.read_excel(voxels_file_path, header=None)
        voxels = pd.Series(voxels[1].values, index=voxels[0])

        assert set(X.columns) == set(
            voxels.index
        ), "Brain Regions in Datafile are different from brain regions in Voxel file"
    else:
        voxels = pd.Series(np.ones(len(X.columns)), index=X.columns)

    if "rob" not in X.columns.str.lower():
        X["rob"] = 0
        voxels["rob"] = 0

    assert y_column in data.columns, f"Column {y_column} doesn't exist in data file"

    y = data[y_column]

    # Y should be converted to performance in case it is NIHSS Score
    y = y.max() - y if y_column_type == "NIHSS Score" else y

    return X, y, voxels


@typechecked
def binarize_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Binarizes the input data by replacing values above the median with 1 and values below or equal to the median with 0.

    Parameters:
        X (pd.DataFrame): The input data to be binarized.

    Returns:
        pd.DataFrame: The binarized data.

    """
    mask = X > np.median(X.values)
    X = X.where(mask, 0)
    X = X.where(~mask, 1)
    return X


@typechecked
def process_path(data_file_path: str) -> Tuple[str, str]:
    data_file_path = os.path.normpath(data_file_path)
    data_file_extension = os.path.splitext(data_file_path)[1]
    if data_file_extension not in (".csv", ".xlsx"):
        raise RuntimeError(
            "The file specified is not CSV or xlsx. Please use a CSV or xlsx file instead"
        )
    return data_file_path, data_file_extension


@typechecked
def train_model(
    model_name: str, X: pd.DataFrame, y: pd.Series
) -> Tuple[float, float, RandomizedSearchCV]:
    model_collection = models[model_name]
    opt = RandomizedSearchCV(
        model_collection.model_class(),
        model_collection.hyperparameters,
        cv=4,
        n_iter=200,
    )
    opt.fit(X.values, y)
    y_pred = np.rint(opt.predict(X.values))
    return accuracy_score(y, y_pred), f1_score(y, y_pred, average="macro"), opt
