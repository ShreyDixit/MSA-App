import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV
from msa_app.ml_models import binarize_data, process_path, train_model, prepare_data


class TestBinarization:
    def test_binarize_data_positive_negative_values(self):
        # Initialize test data
        X = pd.DataFrame({"A": [1, -2, 3, -4, 5]})

        # Invoke the function
        result = binarize_data(X)

        # Check the result
        expected_result = pd.DataFrame({"A": [0, 0, 1, 0, 1]})
        assert result.equals(expected_result)

    # The function correctly binarizes a DataFrame with only positive values.
    def test_binarize_data_only_positive_values(self):
        # Initialize test data
        X = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

        # Invoke the function
        result = binarize_data(X)

        # Check the result
        expected_result = pd.DataFrame({"A": [0, 0, 0, 1, 1]})
        assert result.equals(expected_result)

    # The function correctly binarizes a DataFrame with only negative values.
    def test_binarize_data_only_negative_values(self):
        # Initialize test data
        X = pd.DataFrame({"A": [-1, -2, -3, -4, -5]})

        # Invoke the function
        result = binarize_data(X)

        # Check the result
        expected_result = pd.DataFrame({"A": [1, 1, 0, 0, 0]})
        assert result.equals(expected_result)

    # The function raises a TypeError if the input is not a pandas DataFrame.
    def test_binarize_data_type_error(self):
        # Initialize test data
        X = np.array([1, 2, 3])

        # Invoke the function and check for TypeError
        with pytest.raises(TypeError):
            binarize_data(X)

    def test_binarize_data_floats(self):
        # Initialize test data
        X = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5], "B": [0.6, 0.7, 0.8, 0.9, 1]})

        # Invoke the function
        result = binarize_data(X)

        # Check the result
        expected_result = pd.DataFrame({"A": [0, 0, 0, 0, 0], "B": [1, 1, 1, 1, 1]})
        assert result.equals(expected_result)


class TestProcessPath:
    # Returns the data_file_path and data_file_extension as a tuple for a valid CSV file path.
    def test_valid_csv_file_path(self):
        # Arrange
        data_file_path = "data.csv"

        # Act
        result = process_path(data_file_path)

        # Assert
        assert result == ("data.csv", ".csv")

    def test_valid_xlsx_file_path(self):
        # Arrange
        data_file_path = "data.xlsx"

        # Act
        result = process_path(data_file_path)

        # Assert
        assert result == ("data.xlsx", ".xlsx")

    # Raises a RuntimeError if the file specified is not CSV or xlsx.
    def test_invalid_file_extension(self):
        # Arrange
        data_file_path = "data.txt"

        # Act and Assert
        with pytest.raises(RuntimeError):
            process_path(data_file_path)


class TestModelTraining:
    # The function should be able to train a model successfully with valid input data.
    def test_train_model_valid_input(self):
        # Arrange
        model_name = "Linear Regression"
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act
        accuracy, f1, opt = train_model(model_name, X, y)

        # Assert
        assert isinstance(accuracy, float)
        assert isinstance(f1, float)
        assert isinstance(opt, RandomizedSearchCV)

    def test_train_model_type_error(self):
        # Arrange
        model_name = "Linear Regression"
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        # Act and Assert
        with pytest.raises(TypeError):
            train_model(model_name, X, y)

    def test_train_model_key_error(self):
        # Arrange
        model_name = "Invalid Model"
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act and Assert
        with pytest.raises(KeyError):
            train_model(model_name, X, y)

    def test_train_model_assertion_error(self):
        # Arrange
        model_name = "Linear Regression"
        X = pd.DataFrame(np.random.rand(100, 5) * 2)
        y = pd.Series(np.random.randint(0, 2, 100))

        # Act and Assert
        with pytest.raises(AssertionError):
            train_model(model_name, X, y)


class TestPrepareData:
    data_file_path = "data/roi_data.csv"
    score_file_path = "data/nihss_scores.csv"
    voxels_file_path = "data/num_voxels.csv"

    # Given a valid CSV data file path, y_column name, y_column_type, and voxels file path, the function should return a tuple containing X, y, and voxels.
    @pytest.mark.parametrize("is_score_performance", [True, False])
    def test_valid_file_path_with_voxels(self, is_score_performance):

        # Act
        X, y, voxels = prepare_data(
            data_file_path=self.data_file_path,
            score_file_path=self.score_file_path,
            voxels_file_path=self.voxels_file_path,
            is_score_performance=is_score_performance,
        )

        # Assert
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(voxels, pd.Series)

    # Given a valid CSV data file path, y_column name, y_column_type, and no voxels file path, the function should return a tuple containing X, y, and voxels with all voxels set to 1.
    @pytest.mark.parametrize("is_score_performance", [True, False])
    def test_valid_csv_file_path_without_voxels(self, is_score_performance):
        # Arrange
        voxels_file_path = ""

        # Act
        X, y, voxels = prepare_data(
            data_file_path=self.data_file_path,
            score_file_path=self.score_file_path,
            voxels_file_path=voxels_file_path,
            is_score_performance=is_score_performance,
        )

        # Assert
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(voxels, pd.Series)
        assert all(voxels.drop("rob") == 1)

    def test_nihss_score_calculations(self):
        # Act
        X, y, voxels = prepare_data(
            data_file_path=self.data_file_path,
            score_file_path=self.score_file_path,
            voxels_file_path=self.voxels_file_path,
            is_score_performance=False,
        )
        y_real = pd.read_csv(self.score_file_path).iloc[:, 0]
        assert y.min() == 0
        assert y.max() == y_real.max() - y_real.min()
        assert any(y_real != y)
