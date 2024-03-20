from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
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
    "degree": randint(1, 4),
}

SupportVectorRegressionParams = {
    "kernel": ["sigmoid", "linear", "rbf", "poly"],
    "C": uniform(0.1, 10),
    "gamma": ["scale", "auto"],
    "degree": randint(1, 3),
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
KNeighborsClassifierParams = {
    "n_neighbors": [1],
}

models = {
    "Linear Regression": Model_Collection(LinearRegression, LinearRegressionParams),
    "Logistic Regression": Model_Collection(
        LogisticRegression, LogisticRegressionParams
    ),
    "Support Vector Regressor": Model_Collection(SVR, SupportVectorRegressionParams),
    "Support Vector Classifier": Model_Collection(SVC, SupportVectorClassifierParams),
    "Decision Tree Regressor": Model_Collection(
        DecisionTreeRegressor, DecisionTreeRegressorParams
    ),
    "Random Forest Regressor": Model_Collection(
        RandomForestRegressor, RandomForestRegressorParams
    ),
    "Full MSA (Advanced)": Model_Collection(
        KNeighborsClassifier, KNeighborsClassifierParams
    ),
}

list_of_ml_models = list(models.keys())


@typechecked
def prepare_data(
    *,
    data_file_path: str,
    score_file_path: str,
    voxels_file_path: str,
    is_score_performance: bool,
    add_rob_if_not_present: bool
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares the data for model training and evaluation by loading and processing data files, score files, and optionally voxel files.

    Parameters:
        data_file_path (str): Path to the data file, which should contain the percentage alteration of each brain ROI. Supports CSV or Excel formats.
        score_file_path (str): Path to the file containing scores (e.g., NIHSS Scores or performance metrics). Supports CSV or Excel formats.
        voxels_file_path (str, optional): Path to the voxels file, providing the number of voxels for each ROI. Supports CSV or Excel formats. If not provided, the number of voxels for each ROI is set to 1.
        is_score_performance (bool): If True, the scores are considered as performance scores and are transformed accordingly; otherwise, they are hhtreated directly as NIHSS Scores or similar clinical metrics.
        add_rob_if_not_present (bool): If True, a ROB column will be added with 0 percentage of alteration and 0 voxels. It is ignored if ROB is already present

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the processed feature matrix (X), the target variable (y), and voxel information (voxels), all ready for use in model training and evaluation.

    Raises:
        RuntimeError: If the data file or score file extension is neither CSV nor Excel format.
        AssertionError: If data values are not within the expected range (0 to 100 before normalization).
        AssertionError: If there is a mismatch in the number of records between the data and score files.
        AssertionError: If the set of brain regions in the data file differs from that in the voxels file (when provided).
    """
    data_file_path, data_file_extension = process_path(data_file_path)
    score_file_path, score_file_extension = process_path(score_file_path)

    X = read_file(data_file_path, data_file_extension, is_voxel_file=False) / 100
    y = read_file(score_file_path, score_file_extension, is_voxel_file=False).iloc[:, 0]

    assert (
        np.max(X.values) <= 1 and np.min(X.values) >= 0
    ), "Data should be percentage of alteration in ROI, i.e. between 0 and 100"

    assert len(X) == len(
        y
    ), "Mismatch in number of patients between Data and Scores Files"

    if voxels_file_path:
        voxels_file_path, voxels_file_extension = process_path(voxels_file_path)
        voxels = read_file(voxels_file_path, voxels_file_extension, is_voxel_file=True)

        assert set(X.columns) == set(
            voxels.index
        ), "Brain Regions in Datafile are different from brain regions in Voxel file"
    else:
        voxels = pd.Series(np.ones(len(X.columns)), index=X.columns)

    if add_rob_if_not_present:
        add_rob(X, voxels)

    X.columns = X.columns.str.lower()
    voxels.index = voxels.index.str.lower()

    y = y if is_score_performance else y.max() - y

    return X, y, voxels


def add_rob(X, voxels):
    if "rob" not in X.columns.str.lower():
        X["rob"] = 0
        voxels["rob"] = 0


@typechecked
def read_file(data_file_path: str, data_file_extension: str, is_voxel_file: bool):
    """
    Reads data from a file and returns it as a pandas DataFrame or Series, depending on the file type.

    Parameters:
        data_file_path (str): The path to the file to be read.
        data_file_extension (str): The extension of the file, used to determine the appropriate read method.
        is_voxel_file (bool): Flag indicating whether the file is a voxel file. Affects the structure of the returned data.

    Returns:
        pd.DataFrame or pd.Series: The data read from the file. Returns a Series if reading a voxel file; otherwise, returns a DataFrame.
    """
    header = None if is_voxel_file else 0

    if data_file_extension == ".csv":
        data = pd.read_csv(data_file_path, header=header)
    else:
        data = pd.read_excel(data_file_path, header=header)

    return pd.Series(data[1].values, index=data[0]) if is_voxel_file else data


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
    return X.astype(int)


@typechecked
def process_path(data_file_path: str) -> Tuple[str, str]:
    """
    Normalizes the file path and extracts the file extension to ensure it is a supported format (.csv or .xlsx).

    Parameters:
        data_file_path (str): The path to the file.

    Returns:
        Tuple[str, str]: The normalized file path and its extension.

    Raises:
        RuntimeError: If the file extension is not supported (.csv or .xlsx).
    """
    data_file_path = os.path.normpath(data_file_path)
    data_file_extension = os.path.splitext(data_file_path)[1]
    if data_file_extension not in (".csv", ".xlsx"):
        raise RuntimeError(
            "The file specified is not CSV or xlsx. Please use a CSV or xlsx file instead"
        )
    return data_file_path, data_file_extension


@typechecked
def train_model(
    model_name: str, X: pd.DataFrame, y: pd.Series, random_seed: int
) -> Tuple[float, float, float, RandomizedSearchCV]:
    """
    Trains a machine learning model using randomized search over a predefined hyperparameter space and evaluates its performance.

    Parameters:
        model_name (str): The name of the model to be trained, as defined in the global 'models' dictionary.
        X (pd.DataFrame): The feature matrix for training the model.
        y (pd.Series): The target variable.
        random_seed (int): The Random Seed?

    Returns:
        Tuple[float, float, float, RandomizedSearchCV]: The accuracy score, F1 score, r2_score, and the trained RandomizedSearchCV object.

    Raises:
        AssertionError: If the input data X is not normalized between 0 and 1.
    """
    assert (
        np.max(X.values) <= 1 and np.min(X.values) >= 0
    ), "Data passed is not beween 1 and 0. There is some error somewhere!"

    model_collection = models[model_name]
    opt = RandomizedSearchCV(
        model_collection.model_class(),
        model_collection.hyperparameters,
        cv=4,
        n_iter=200,
        verbose=2,
        random_state=random_seed,
    )
    opt.fit(X.values, y)
    y_pred = np.rint(opt.predict(X.values))
    return (
        accuracy_score(y, y_pred),
        f1_score(y, y_pred, average="macro"),
        r2_score(y, y_pred),
        opt,
    )
