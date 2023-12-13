import json

from msapy import msa
import numpy as np
import pandas as pd

from msa_app import ml_models


class MSA:
    def __init__(self, file_path: str, y_column: str, model_name: str):
        self.file_path = file_path
        self.y_column = y_column
        self.model_name = model_name
        self.n_permutation = 1000
        self.RoB = []

    def train_model(self):
        test_accuracy, test_f1, trained_model = ml_models.train_model(model_name=self.model_name, X=self.X, y=self.y)
        self.test_accuracy = test_accuracy
        self.test_f1 = test_f1
        self.trained_model = trained_model

    def prepare_data(self):
        X, y = ml_models.prepare_data(self.file_path, self.y_column)
        self.X = X
        self.y = y
        self.elements = list(self.X.columns)

    def run_msa(self):
        self.shapley_table = msa.interface(n_permutations=self.n_permutation,
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
        }

        with open("results.json", "w") as f:
            json.dump(save_dict, f, indent=4)
            
        self.shapley_table.shapley_values.to_csv("shapley_values.csv")