from itertools import combinations
import json
import time
from matplotlib import pyplot as plt
import customtkinter as ctk

from msapy import msa
import numpy as np
import pandas as pd
from typeguard import typechecked

from msa_app import ml_models


class MSA:
    def __init__(
        self,
        data_file_path: str,
        y_column: str,
        y_column_type: str,
        model_name: str,
        voxels_file_path: str,
        progress_bar: ctk.CTkProgressBar,
        root: ctk.CTk,
        binarize_data: bool = True,
        run_interaction_2d: bool = True,
    ):
        """
        Initialize the MSA object.

        Parameters:
            data_file_path (str): The path to the data file.
            y_column (str): The name of the target column in the data file.
            y_column_type (str): The type of the target column (e.g., 'categorical', 'continuous').
            model_name (str): The name of the model to be used for training.
            voxels_file_path (str): The path to the voxels file.
            progress_bar (ctk.CTkProgressBar): The progress bar object.
            root (ctk.CTk): The root GUI object.
            binarize_data (bool, optional): Whether to binarize the data or not. Defaults to True.
        """

        self.data_file_path = data_file_path
        self.voxels_file_path = voxels_file_path
        self.y_column = y_column
        self.y_column_type = y_column_type
        self.model_name = model_name
        self.progress_bar = progress_bar
        self.root_gui = root
        self.binarize_data = binarize_data
        self.n_permutation = 1000
        self.smallest_set_of_roi = 6 if run_interaction_2d else 3
        self.RoB = []

    def train_model(self):
        accuracy, f1, trained_model = ml_models.train_model(
            model_name=self.model_name, X=self.X, y=self.y
        )
        self.accuracy = accuracy
        self.f1 = f1
        self.trained_model = trained_model

    def prepare_data(self):
        """
        Prepare the data for the MSA algorithm.

        This method prepares the data by performing the following steps:
        1. Load the data from the specified data file path.
        2. Extract the features (X) and the target variable (y) from the data.
        3. Binarize the features if specified.
        4. Copy the original features (X) and target variable (y) to instance variables.
        5. Copy the voxels data to an instance variable.
        6. Create a list of elements (column names of X) and store it in an instance variable.
        7. Calculate the total number of regions of interest (ROIs) and store it in an instance variable.
        8. Calculate the progress bar step size based on the total number of ROIs.

        Parameters:
            None

        Returns:
            None
        """
        X, y, voxels = ml_models.prepare_data(
            self.data_file_path,
            self.y_column,
            self.y_column_type,
            self.voxels_file_path,
        )
        self.X_unbinorized = X.copy()
        self.X = ml_models.binarize_data(X) if self.binarize_data else X.copy()
        self.y = y.copy()
        self.voxels = voxels.copy()
        self.elements = list(self.X.columns)
        self.total_roi = len(self.elements)
        self.progress_bar_step = 1 / (self.total_roi - self.smallest_set_of_roi + 1)

    def run_msa(self):
        self.shapley_table = msa.interface(
            n_permutations=self.n_permutation,
            elements=list(self.X.columns),
            objective_function=self.objective_function,
        )

    def run_iterative_msa(self):
        """
        Run the Iterative MSA algorithm.

        This method runs the Iterative MSA algorithm by performing the following steps:
        1. Set up the progress bar.
        2. Run the MSA algorithm.
        3. Update the progress bar.
        4. Repeat the following steps until the number of elements is reduced to 3:
            a. Get the lowest contributing region of interest (ROI).
            b. Check if the Rest of Brain (RoB) region is significant.
            c. Save the attributes of the iterative MSA if the RoB region is not significant.
            d. Add the lowest contributing ROI to the RoB region.
            e. Remove the lowest contributing ROI from the elements.
            f. Append the lowest contributing ROI to the RoB list.
            g. Train the model.
            h. Run the MSA algorithm.
            i. Update the progress bar.
        5. Save the attributes of the iterative MSA.

        Parameters:
            None

        Returns:
            None
        """

        self.root_gui.after(0, self.setup_progressbar)

        self.run_msa()

        self.root_gui.after(0, self.update_progressbar)
        self.save_iterative_msa_attributes()

        while len(self.elements) > self.smallest_set_of_roi:
            lowest_contributing_region = self._get_lowest_contributing_region()

            if not self._is_significant("rob"):
                self.save_iterative_msa_attributes()

            self.add_roi_to_rob(lowest_contributing_region)
            self.remove_roi(lowest_contributing_region)
            self.RoB.append(lowest_contributing_region)
            self.train_model()
            self.run_msa()
            self.root_gui.after(0, self.update_progressbar)

    @typechecked
    def add_roi_to_rob(self, roi: str):
        """
        Add a region of interest (ROI) to the Rest of Brain (RoB) region.

        This method calculates the new number of voxels altered in the RoB region after adding a new ROI. It updates the RoB voxels count and recalculates the average value of alteration of the RoB region based on the new voxels count.

        Parameters:
            roi (str): The name of the ROI to be added to the RoB region.

        Returns:
            None
        """
        new_rob_num_voxels_altered = (
            self.X_unbinorized["rob"] * self.voxels["rob"]
        ) + (self.X_unbinorized[roi] * self.voxels[roi])
        self.voxels["rob"] += self.voxels[roi]
        self.X_unbinorized["rob"] = new_rob_num_voxels_altered / self.voxels["rob"]

    def save_iterative_msa_attributes(self):
        self.shapley_table_iterative = self.shapley_table.copy()
        self.accuracy_iterative = self.accuracy
        self.f1_iterative = self.f1
        self.trained_model_iterative = self.trained_model
        self.RoB_iterative = self.RoB.copy()
        self.X_unbinorized_iterative = self.X_unbinorized.copy()
        self.voxels_iterative = self.voxels.copy()

    @typechecked
    def remove_roi(self, roi: str):
        """
        Remove a region of interest (ROI).

        This method removes a specified ROI by performing the following steps:
        1. Drop the ROI from the voxels data.
        2. Drop the ROI column from the unbinarized features (X_unbinorized).
        3. Update the binarized features (X) by either binarizing the updated X_unbinorized or copying it.
        4. Update the list of elements (column names of X) to reflect the removed ROI.

        Parameters:
            roi (str): The name of the ROI to be removed.

        Returns:
            None
        """
        self.voxels.drop(roi, inplace=True)
        self.X_unbinorized.drop(roi, axis=1, inplace=True)
        self.X = (
            ml_models.binarize_data(self.X_unbinorized)
            if self.binarize_data
            else self.X_unbinorized.copy()
        )
        self.elements = list(self.X.columns)

    def update_progressbar(self):
        self.progress_bar.set(self.progress_bar.get() + self.progress_bar_step)
        self.root_gui.update_idletasks()

    def setup_progressbar(self):
        self.progress_bar.configure(
            mode="determinate",
        )
        self.progress_bar.set(0)
        self.root_gui.update_idletasks()

    @typechecked
    def _get_lowest_contributing_region(self) -> str:
        """
        Get the lowest contributing region of interest (ROI).

        This method calculates the ROI with the lowest absolute Shapley value from the Shapley table. If the lowest contributing ROI is the Rest of Brain (RoB), the ROI with the second lowest absolute Shapley value is returned instead.

        Returns:
            str: The name of the lowest contributing ROI.
        """
        lowest_contributing_region = self.shapley_table.shapley_values.abs().idxmin()
        if lowest_contributing_region.lower() == "rob":
            lowest_contributing_region = (
                self.shapley_table.shapley_values.abs().sort_values().index[1]
            )
        return lowest_contributing_region

    def run_interaction_2d(self):
        self.X_unbinorized = getattr(self, "X_unbinorized_iterative", self.X)
        self.voxels = getattr(self, "voxels_iterative", self.voxels)
        self.root_gui.after(0, self.setup_progressbar)
        self.remove_roi("rob")
        self.train_model()

        all_pairs = list(combinations(self.elements, 2))
        self.progress_bar_step = 1 / len(all_pairs)

        self.interactions = 0

        for pair in all_pairs:
            self.interactions += msa.network_interaction_2d(
                n_permutations=self.n_permutation,
                elements=self.elements,
                objective_function=self.objective_function,
                pairs=[pair],
                lazy=True,
            )
            self.update_progressbar()

    @typechecked
    def objective_function(self, complement) -> float:
        x = pd.Series(np.zeros_like(self.X.iloc[0]), index=self.X.columns)
        if complement:
            x[list(complement)] = 1
        return np.maximum(0, self.trained_model.predict(x.values.reshape(1, -1)))[0]

    def save_iterative(self):
        save_dict = {
            "shapley_values_iterative": self.shapley_table_iterative.shapley_values.to_dict(),
            "Rest of Breain": self.RoB_iterative,
            "accuracy": self.accuracy_iterative,
            "f1": self.f1_iterative,
            "model used": self.model_name,
            "model_params": self.trained_model_iterative.best_params_,
        }

        saving_time = time.strftime("%Y%m%d-%H%M%S")
        with open(f"results_iterative_{saving_time}.json", "w") as f:
            json.dump(save_dict, f, indent=4)

        self.shapley_table_iterative.shapley_values.to_csv(
            f"shapley_values_iterative_{saving_time}.csv"
        )

    def save(self):
        save_dict = {
            "shapley_values": self.shapley_table.shapley_values.to_dict(),
            "accuracy": self.accuracy,
            "f1": self.f1,
            "model used": self.model_name,
            "model_params": self.trained_model.best_params_,
        }

        saving_time = time.strftime("%Y%m%d-%H%M%S")
        with open(f"results_{saving_time}.json", "w") as f:
            json.dump(save_dict, f, indent=4)

        self.shapley_table.shapley_values.to_csv(f"shapley_values_{saving_time}.csv")

    @typechecked
    def plot_msa(self, iterative: bool = False):
        # Calculate mean values and confidence intervals (CI) for error bars
        shapley_table = (
            self.shapley_table_iterative if iterative else self.shapley_table
        )
        mean_values = shapley_table.shapley_values
        std_dev = shapley_table.std(axis=0)
        sample_size = shapley_table.shape[0]
        confidence_interval = 1.96 * (std_dev / (sample_size**0.5))  # 95% CI

        # Plotting bar graph with error bars
        plt.figure(figsize=(10, 10))
        plt.barh(
            shapley_table.columns,
            mean_values,
            xerr=confidence_interval,
            capsize=5,
            color="skyblue",
            edgecolor="black",
        )

        plt.xlabel("Shapley Values")
        plt.ylabel("Brain Regions")
        plt.show()

    def plot_network_interaction(self):
        # Plotting the heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(self.interactions, cmap="RdBu", interpolation="nearest")
        plt.xticks(np.arange(len(self.elements)), self.elements, rotation=75)
        plt.yticks(np.arange(len(self.elements)), self.elements)
        plt.colorbar()  # Display color bar
        plt.title("Network Interactions")
        plt.show()

    @typechecked
    def _is_significant(self, brain_region: str) -> np.bool_:
        """
        Check if a brain region is statistically significant based on its Shapley values.

        Parameters:
            brain_region (str): The name of the brain region to check.

        Returns:
            bool: True if the brain region is statistically significant, False otherwise.
        """
        return (
            abs(self.shapley_table[brain_region].mean())
            > self.shapley_table[brain_region].std()
        )
