import json
import time
from typing import Optional
from matplotlib import pyplot as plt

from msapy import msa
import numpy as np
import pandas as pd

from msa_app import ml_models


class MSA:
    def __init__(self, data_file_path: str, y_column: str, y_column_type: str, model_name: str, voxels_file_path: Optional[str] = None):
        """
        Initializes the model with the given data file path, y column, y column type, model name, and optional voxels file path.

        Parameters:
            data_file_path (str): The path to the data file with data on ROI alterations.
            y_column (str): The column representing the target  i.e. NIHSS Score or Performance.
            y_column_type (str): Can have two values, either NIHSS or Performance.
            model_name (str): The name of the model.
            voxels_file_path (str, optional): The path to the voxels file, defaults to None.

        Returns:
            None
        """
        self.data_file_path = data_file_path
        self.voxels_file_path = voxels_file_path
        self.y_column = y_column
        self.y_column_type = y_column_type
        self.model_name = model_name
        self.n_permutation = 1000
        self.RoB = []

    def train_model(self):
        """
        Train the model using the specified model name, input features, and target labels.
        """
        accuracy, f1, trained_model = ml_models.train_model(model_name=self.model_name, X=self.X, y=self.y)
        self.accuracy = accuracy
        self.f1 = f1
        self.trained_model = trained_model

    def prepare_data(self):
        X, y, voxels = ml_models.prepare_data(self.data_file_path, self.y_column, self.y_column_type, self.voxels_file_path)
        self.X = X.copy()
        self.y = y.copy()
        self.voxels = voxels.copy()
        self.elements = list(self.X.columns)

    def run_msa(self):
        self.shapley_table = msa.interface(n_permutations=self.n_permutation,
                                      elements=list(self.X.columns),
                                      objective_function=self.objective_function)
        
    def run_iterative_msa(self):
        """
        Run the iterative MSA algorithm.

        This function iteratively runs the MSA algorithm until the number of elements
        is reduced to 2 or less, or until the RoB contribution is no longer significant.
        """
        self.run_msa()

        while len(self.elements) > 3:
            lowest_contributing_region = self._get_lowest_contributing_region()

            if not self._is_significant("rob"):
                self.shapley_table_iterative = self.shapley_table.copy()
                self.accuracy_iterative = self.accuracy
                self.f1_iterative = self.f1
                self.trained_model_iterative = self.trained_model
                self.RoB_iterative = self.RoB.copy()
            
            new_rob_num_voxels_altered = (self.X['rob'] * self.voxels['rob']) + (self.X[lowest_contributing_region] * self.voxels[lowest_contributing_region])
            self.voxels['rob'] += self.voxels[lowest_contributing_region]
            self.X['rob'] = new_rob_num_voxels_altered / self.voxels['rob']

            self.voxels.drop(lowest_contributing_region, inplace=True)
            self.X.drop(lowest_contributing_region, axis=1, inplace=True)
            self.elements = list(self.X.columns)
            self.RoB.append(lowest_contributing_region)
            self.train_model()
            self.run_msa()


    def _get_lowest_contributing_region(self):
        lowest_contributing_region = self.shapley_table.shapley_values.abs().idxmin()
        if lowest_contributing_region.lower() == "rob":
             lowest_contributing_region = self.shapley_table.shapley_values.abs().sort_values().index[1]
        return lowest_contributing_region

            
        
    def run_interaction_2d(self):
        self.interactions = msa.network_interaction_2d(n_permutations=self.n_permutation,
                                      elements=list(self.X.columns),
                                      objective_function=self.objective_function)

    def objective_function(self, complement):
        x = pd.Series(np.zeros_like(self.X.iloc[0]), index = self.X.columns)
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
            "model_params": self.trained_model_iterative.best_params_
        }
        
        saving_time = time.strftime('%Y%m%d-%H%M%S')
        with open(f"results_iterative_{saving_time}.json", "w") as f:
            json.dump(save_dict, f, indent=4)
            
        self.shapley_table_iterative.shapley_values.to_csv(f"shapley_values_iterative_{saving_time}.csv")

    def save(self):
        save_dict = {
            "shapley_values": self.shapley_table.shapley_values.to_dict(),
            "accuracy": self.accuracy,
            "f1": self.f1,
            "model used": self.model_name,
            "model_params": self.trained_model.best_params_
        }
        
        saving_time = time.strftime('%Y%m%d-%H%M%S')
        with open(f"results_{saving_time}.json", "w") as f:
            json.dump(save_dict, f, indent=4)
            
        self.shapley_table.shapley_values.to_csv(f"shapley_values_{saving_time}.csv")

    def plot_msa(self, iterative=False):
        # Calculate mean values and confidence intervals (CI) for error bars
        shapley_table = self.shapley_table_iterative if iterative else self.shapley_table
        mean_values = shapley_table.shapley_values
        std_dev = shapley_table.std(axis=0)
        sample_size = shapley_table.shape[0]
        confidence_interval = 1.96 * (std_dev / (sample_size ** 0.5))  # 95% CI

        # Plotting bar graph with error bars
        plt.figure(figsize=(10, 10))
        plt.barh(shapley_table.columns, mean_values, xerr=confidence_interval, capsize=5, color='skyblue', edgecolor='black')

        plt.xlabel('Shapley Values') 
        plt.ylabel('Brain Regions') 
        plt.show()

    def plot_network_interaction(self):
        # Plotting the heatmap
        plt.figure(figsize=(8, 6)) 
        plt.imshow(self.interactions, cmap='viridis', interpolation='nearest') 
        plt.xticks(np.arange(len(self.elements)), self.elements)
        plt.yticks(np.arange(len(self.elements)), self.elements)
        plt.colorbar()  # Display color bar
        plt.title('Network Interactions') 
        plt.show()

    def _is_significant(self, brain_region: str) -> bool:
        return abs(self.shapley_table[brain_region].mean()) > self.shapley_table[brain_region].std()





# X, y = get_training_data()

# RoB = []

# 60 ROI
# ..... 40 ROI and average (20 ROB)

# 2 ROI and 1 ROB
# And see where the RoB became significant

# while <condition>:
#     model = train_model(X, y)
#     shapley_values = MSA(model)
#     least_important_ROI = argmin(absolute(shapley_values))
#     if shapley_values[least_important_ROI] < std_err(shapley_values[least_important_ROI]):
#         X.remove(least_important_ROI)
#         RoB.append(least_important_ROI)
#     else:
#         break


# while <condition>
#     model = train_model(X, y)
#     shapley_values = MSA(model)
#     least_important_ROI = argmin(absolute(shapley_values))


# numVoxelA, numVoxelB
# percentageAltA, percentageAltB

# percentageAltAB = (percentageAltA * numVoxelA + percentageAltB * numVoxelB) / (numVoxelA + numbVoxelB)

