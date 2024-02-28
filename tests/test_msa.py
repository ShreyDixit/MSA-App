# Dependencies:
import numpy as np
import pandas as pd
import pytest
import customtkinter as ctk
from msa_app import ml_models
from msa_app.msa import MSA


class FakeEstimator:
    def fit(self, X, y):
        pass

    def predict(self, X: np.ndarray):
        return X.sum(1).astype(float)


class TestMSA:
    data_file_path = "data/roi_data.csv"
    score_file_path = "data/nihss_scores.csv"
    voxels_file_path = "data/num_voxels.csv"
    is_score_performance = False
    root = ctk.CTk()
    progress_bar = ctk.CTkProgressBar(root)
    model_name = "Support Vector Regressor"
    X = (
        pd.DataFrame(
            {
                "ABC": [10.0] * 100,
                "BCD": [20.0] * 100,
                "CDE": [30.0] * 100,
                "EFG": [40.0] * 100,
                "FGH": [50.0] * 100,
                "rob": [50.0] * 100,
            }
        )
        / 100
    )
    elements = list(X.columns)
    y = pd.Series(np.random.randint(0, 8, 100))
    voxels = pd.Series(data=[40, 20, 30, 40, 50, 60], index=elements)

    msa_instance_binarized = MSA(
        data_file_path=data_file_path,
        score_file_path=score_file_path,
        model_name=model_name,
        voxels_file_path=voxels_file_path,
        progress_bar=progress_bar,
        root=root,
        is_score_performance=is_score_performance,
        binarize_data=True,
        run_interaction_2d=False,
        random_seed=42,
        num_permutation=10,
        add_rob_if_not_present=True
    )

    msa_instance_unbinarized = MSA(
        data_file_path=data_file_path,
        score_file_path=score_file_path,
        model_name=model_name,
        voxels_file_path=voxels_file_path,
        progress_bar=progress_bar,
        root=root,
        is_score_performance=is_score_performance,
        binarize_data=False,
        run_interaction_2d=False,
        random_seed=42,
        num_permutation=10,
        add_rob_if_not_present=True
    )

    def _is_binary(self, df: pd.DataFrame):
        return df.applymap(lambda x: x in [0, 1]).all().all()

    # Data can be prepared for model training and evaluation.
    @pytest.mark.parametrize("binarize", [True, False])
    def test_prepare_data_for_model_training_and_evaluation(self, mocker, binarize):
        mocker.patch.object(ml_models, "prepare_data")
        ml_models.prepare_data.return_value = (self.X, self.y, self.voxels)

        if binarize:
            self.msa_instance_binarized.prepare_data()
            assert all(self.msa_instance_binarized.X_unbinorized == self.X)
            assert self._is_binary(self.msa_instance_binarized.X)
            assert all(self.msa_instance_binarized.y == self.y)
            assert all(self.msa_instance_binarized.voxels == self.voxels)
            assert "rob" in list(self.msa_instance_binarized.X.columns)
            assert 6 == len(self.msa_instance_binarized.elements)
        else:
            self.msa_instance_unbinarized.prepare_data()
            assert all(self.msa_instance_unbinarized.X_unbinorized == self.X)
            assert all(self.msa_instance_unbinarized.X == self.X)
            assert all(self.msa_instance_unbinarized.y == self.y)
            assert all(self.msa_instance_unbinarized.voxels == self.voxels)
            assert "rob" in list(self.msa_instance_unbinarized.X.columns)
            assert 6 == len(self.msa_instance_unbinarized.elements)

        # MSA can remove a region of interests

    @pytest.mark.parametrize("binarize", [True, False])
    def test_model_training_and_evaluation(self, mocker, binarize):
        mocker.patch.object(ml_models, "train_model")

        ml_models.train_model.return_value = (1.0, 1.0, FakeEstimator())

        if binarize:
            self.msa_instance_binarized.train_model()
            assert self.msa_instance_binarized.f1 == 1
            assert self.msa_instance_binarized.accuracy == 1
        else:
            self.msa_instance_unbinarized.train_model()
            assert self.msa_instance_unbinarized.f1 == 1
            assert self.msa_instance_unbinarized.accuracy == 1

        # MSA can run and generate Shapley values.

    @pytest.mark.parametrize("binarize", [True, False])
    def test_run_msa(self, binarize):
        if binarize:
            self.msa_instance_binarized.run_msa()
            assert isinstance(self.msa_instance_binarized.shapley_table, pd.DataFrame)
            assert all(-1 == self.msa_instance_binarized.shapley_table.shapley_values)
            assert all(self.msa_instance_binarized.X_unbinorized == self.X)
            assert self._is_binary(self.msa_instance_binarized.X)
            assert all(self.msa_instance_binarized.y == self.y)
            assert all(self.msa_instance_binarized.voxels == self.voxels)
            assert "rob" in list(self.msa_instance_binarized.X.columns)

        else:
            self.msa_instance_unbinarized.run_msa()
            assert all(self.msa_instance_unbinarized.X_unbinorized == self.X)
            assert all(self.msa_instance_unbinarized.X == self.X)
            assert all(self.msa_instance_unbinarized.y == self.y)
            assert all(self.msa_instance_unbinarized.voxels == self.voxels)
            assert "rob" in list(self.msa_instance_unbinarized.X.columns)

    @pytest.mark.parametrize("binarize", [True, False])
    def test_add_roi_to_rob(self, binarize):
        roi = "ABC"

        if binarize:
            self.msa_instance_binarized.add_roi_to_rob(roi)
            assert self.msa_instance_binarized.voxels["rob"] == 100
            assert all(self.msa_instance_binarized.X_unbinorized["rob"] == 0.34)

        else:
            self.msa_instance_unbinarized.add_roi_to_rob(roi)
            assert self.msa_instance_binarized.voxels["rob"] == 100
            assert all(self.msa_instance_binarized.X_unbinorized["rob"] == 0.34)

    @pytest.mark.parametrize("binarize", [True, False])
    def test_remove_roi(self, binarize):
        roi = "ABC"

        if binarize:
            self.msa_instance_binarized.remove_roi(roi)
            assert roi not in self.msa_instance_binarized.X_unbinorized.columns
            assert roi not in self.msa_instance_binarized.X.columns
            assert roi not in self.msa_instance_binarized.elements
            assert self.voxels.sum() == self.msa_instance_binarized.voxels.sum()

        else:
            self.msa_instance_unbinarized.remove_roi(roi)
            assert roi not in self.msa_instance_unbinarized.X_unbinorized.columns
            assert roi not in self.msa_instance_unbinarized.X.columns
            assert roi not in self.msa_instance_unbinarized.elements
            assert self.voxels.sum() == self.msa_instance_unbinarized.voxels.sum()

    @pytest.mark.parametrize("binarize", [True, False])
    def test_run_msa_iterative(self, mocker, binarize):
        mocker.patch.object(ml_models, "train_model")

        ml_models.train_model.return_value = (1.0, 1.0, FakeEstimator())
        if binarize:
            self.msa_instance_binarized.run_iterative_msa()
            assert isinstance(
                self.msa_instance_binarized.shapley_table_iterative, pd.DataFrame
            )
            assert all(
                -1 == self.msa_instance_binarized.shapley_table_iterative.shapley_values
            )
            assert self._is_binary(self.msa_instance_binarized.X)
            assert all(self.msa_instance_binarized.y == self.y)
            assert "rob" in list(self.msa_instance_binarized.X.columns)
            assert self.voxels.sum() == self.msa_instance_binarized.voxels.sum()

        else:
            self.msa_instance_unbinarized.run_iterative_msa()
            assert isinstance(
                self.msa_instance_unbinarized.shapley_table_iterative, pd.DataFrame
            )
            assert all(
                -1
                == self.msa_instance_unbinarized.shapley_table_iterative.shapley_values
            )
            assert all(self.msa_instance_unbinarized.y == self.y)
            assert "rob" in list(self.msa_instance_unbinarized.X.columns)
            assert self.voxels.sum() == self.msa_instance_unbinarized.voxels.sum()
