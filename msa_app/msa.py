import json
from matplotlib import pyplot as plt

from msapy import msa
import numpy as np
import pandas as pd

from msa_app import ml_models


class MSA:
    def __init__(self, file_path: str, y_column: str, y_column_type: str, model_name: str):
        self.file_path = file_path
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
        X, y = ml_models.prepare_data(self.file_path, self.y_column, self.y_column_type)
        self.X = X
        self.y = y
        self.elements = list(self.X.columns)

    def run_msa(self):
        self.shapley_table = msa.interface(n_permutations=self.n_permutation,
                                      elements=list(self.X.columns),
                                      objective_function=self.objective_function)
        
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

        plt.ylabel('Shapley Values') 
        plt.xlabel('Brain Regions') 
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
