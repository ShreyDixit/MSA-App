import json
from typing import Optional
from matplotlib import pyplot as plt

from msapy import msa
import numpy as np
import pandas as pd

from msa_app import ml_models


class MSA:
    def __init__(self, data_file_path: str, y_column: str, y_column_type: str, model_name: str, voxels_file_path: Optional[str] = None):
        self.data_file_path = data_file_path
        self.voxels_file_path = voxels_file_path
        self.y_column = y_column
        self.y_column_type = y_column_type
        self.model_name = model_name
        self.n_permutation = 1000
        self.RoB = []

    def train_model(self):
        test_accuracy, test_f1, trained_model = ml_models.train_model(model_name=self.model_name, X=self.X, y=self.y)
        self.test_accuracy = test_accuracy
        self.test_f1 = test_f1
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
        while len(self.elements) > 2:
            self.run_msa()
            if self._is_significant("rob"):
                break

            lowest_contributing_region = self._get_lowest_contributing_region()
            
            new_rob_num_voxels_altered = (self.X['rob'] * self.voxels['rob']) + (self.X[lowest_contributing_region] * self.voxels[lowest_contributing_region])
            self.voxels['rob'] += self.voxels[lowest_contributing_region]
            self.X['rob'] = new_rob_num_voxels_altered / self.voxels['rob']

            self.voxels.drop(lowest_contributing_region, inplace=True)
            self.X.drop(lowest_contributing_region, axis=1, inplace=True)
            self.elements = list(self.X.columns)
            self.RoB.append(lowest_contributing_region)
            self.train_model()


    def _get_lowest_contributing_region(self):
        lowest_contributing_region = self.shapley_table.shapley_values.abs().idxmin()
        if lowest_contributing_region.lower() == "rob":
             lowest_contributing_region = self.shapley_table.shapley_values.abs().argsort().iloc[1]
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
    
    def save(self):
        save_dict = {
            "shapley_values": self.shapley_table.shapley_values.to_dict(),
            "Rest of Breain": self.RoB,
            "test accuracy": self.test_accuracy,
            "test f1": self.test_f1,
            "model used": self.model_name,
            "model_params": self.trained_model.best_params_
        }

        with open("results.json", "w") as f:
            json.dump(save_dict, f, indent=4)
            
        self.shapley_table.shapley_values.to_csv("shapley_values.csv")

    def plot_msa(self):
        # Calculate mean values and confidence intervals (CI) for error bars
        mean_values = self.shapley_table.shapley_values
        std_dev = self.shapley_table.std(axis=0)
        sample_size = self.shapley_table.shape[0]
        confidence_interval = 1.96 * (std_dev / (sample_size ** 0.5))  # 95% CI

        # Plotting bar graph with error bars
        plt.figure(figsize=(10, int(len(self.elements) * 0.2)))
        plt.barh(self.elements, mean_values, xerr=confidence_interval, capsize=5, color='skyblue', edgecolor='black')

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

