from itertools import product
import pytest
import customtkinter as ctk
from msa_app.msa import MSA
model_list = ["Linear Regression", "Logistic Regression"]
pass_voxel_file_options = [True, False]
class TestMSALong:
    root = ctk.CTk()
    progress_bar = ctk.CTkProgressBar(root)

    @pytest.mark.parametrize("model_name, pass_voxel_file", list(product(model_list[:-1], pass_voxel_file_options)))
    def test_msa(self, model_name, pass_voxel_file):
        msa = MSA(
            data_file_path="data/roi_data.csv",
            score_file_path="data/nihss_scores.csv",
            voxels_file_path=("data/num_voxels.csv" if pass_voxel_file else ""),
            model_name=model_name,
            progress_bar=self.progress_bar,
            root=self.root,
            is_score_performance=False,
            binarize_data=True,
            run_interaction_2d=True,
            random_seed=42,
            num_permutation=10,
            full_msa=False,
            add_rob_if_not_present=True
        )

        msa.prepare_data()
        msa.train_model()
        msa.run_iterative_msa()

        msa.run_interaction_2d()