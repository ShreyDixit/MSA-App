import customtkinter as ctk
from tkinter import filedialog
from msa_app import ml_models
from msa_app.msa import MSA
import threading

ctk.set_default_color_theme("dark-blue")


class GUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.geometry("900x400")
        self.root.title()
        self.data_file_path = ctk.StringVar()
        self.voxels_file_path = ctk.StringVar()
        self.create_widget()

    def create_widget(self):
        # Function to handle the file upload
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.browse_button_data_file = ctk.CTkButton(
            self.root, text="Upload Data File", command=self.browse_data_file
        )
        self.browse_button_data_file.grid(row=0, column=0, pady=20, sticky="w")

        # Entry widget to display the file path
        self.data_file_entry = ctk.CTkEntry(
            self.root, textvariable=self.data_file_path, state="readonly", width=200
        )
        self.data_file_entry.grid(row=0, column=1, padx=10, sticky="e")

        self.browse_button_voxels_file = ctk.CTkButton(
            self.root,
            text="Upload Voxels File (Optional)",
            command=self.browse_voxels_file,
        )
        self.browse_button_voxels_file.grid(row=1, column=0, pady=20, sticky="w")

        # Entry widget to display the file path
        self.voxels_file_entry = ctk.CTkEntry(
            self.root, textvariable=self.voxels_file_path, state="readonly", width=200
        )
        self.voxels_file_entry.grid(row=1, column=1, padx=10, sticky="e")

        self.y_column_type = ctk.StringVar(value="NIHSS Score")
        self.y_column_label = ctk.CTkComboBox(
            self.root,
            values=["NIHSS Score", "Performance"],
            variable=self.y_column_type,
        )
        self.y_column_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.y_column = ctk.StringVar()  # set initial value
        self.y_column_entry = ctk.CTkEntry(self.root, textvariable=self.y_column)
        self.y_column_entry.grid(row=2, column=1, padx=10, pady=10, sticky="e")

        self.ml_model_label = ctk.CTkLabel(self.root, text="Machine Learning Model: ")
        self.ml_model_label.grid(row=3, column=0, padx=10, sticky="w")

        self.ml_model = ctk.StringVar()  # set initial value
        self.ml_model_combobox = ctk.CTkComboBox(
            self.root,
            values=list(ml_models.models.keys()),
            variable=self.ml_model,
            width=200,
        )
        self.ml_model_combobox.grid(row=3, column=1, padx=10, sticky="e")

        self.run_iterative_var = ctk.IntVar()
        self.run_iterative_checkbox = ctk.CTkCheckBox(
            self.root, text="Run Iterative", variable=self.run_iterative_var
        )
        self.run_iterative_checkbox.grid(
            row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew"
        )

        # Submit button
        self.msa_button = ctk.CTkButton(
            self.root, text="Run MSA", command=self.click_run_button
        )
        self.msa_button.grid(
            row=5, column=0, columnspan=2, pady=10, padx=10, sticky="ew"
        )

        self.progress_bar = ctk.CTkProgressBar(self.root)
        self.progress_bar.grid(
            row=6, column=0, columnspan=2, pady=10, padx=10, sticky="ew"
        )

        self.text = ctk.CTkTextbox(self.root, height=100)
        self.text.grid(row=7, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        # self.interaction_2d_button = ctk.CTkButton(self.root, text="Run Network Interctions", command=self.run_network_interaction_2d)
        # self.interaction_2d_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    def browse_data_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data_file_path.set(file_path)

    def browse_voxels_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.voxels_file_path.set(file_path)

    def click_run_button(self):
        if not self.run_iterative_var.get():
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
        threading.Thread(target=self.run_msa, daemon=True).start()

    def run_msa(self):
        msa = MSA(
            self.data_file_path.get(),
            self.y_column.get(),
            self.y_column_type.get(),
            self.ml_model.get(),
            self.voxels_file_path.get(),
            self.progress_bar,
            self.root,
        )
        msa.prepare_data()
        msa.train_model()

        if self.run_iterative_var.get():
            msa.run_iterative_msa()
            msa.save_iterative()
        else:
            msa.run_msa()
            self.progress_bar.stop()
            msa.save()

        msa.plot_msa(bool(self.run_iterative_var.get()))

    def run_network_interaction_2d(self):
        msa = MSA(
            self.data_file_path.get(),
            self.y_column.get(),
            self.y_column_type.get(),
            self.ml_model.get(),
        )
        msa.prepare_data()
        msa.train_model()
        msa.run_interaction_2d()
        msa.plot_network_interaction()
