import customtkinter as ctk
from tkinter import filedialog, messagebox
from msa_app import ml_models
from msa_app.msa import MSA
import threading

ctk.set_default_color_theme("dark-blue")


class GUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.geometry("600x700")
        self.root.title("Lesion-Symptom Mapping using MSA")
        self.data_file_path = ctk.StringVar()
        self.voxels_file_path = ctk.StringVar()
        self.y_column_type = ctk.StringVar(value="NIHSS Score")
        self.y_column = ctk.StringVar()
        self.ml_model = ctk.StringVar()
        self.run_network_interaction_2d_var = ctk.IntVar()
        self.binarize_data_var = ctk.IntVar()
        self.run_iterative_var = ctk.IntVar()
        self.create_widget()

    def create_widget(self):
        # Function to handle the file upload
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.browse_button_data_file = ctk.CTkButton(
            self.root,
            text="Upload Data File",
            command=self.browse_data_file,
            font=("Helvetica", 18),
        )
        self.browse_button_data_file.grid(row=0, column=0, pady=20, padx=10, sticky="w")

        # Entry widget to display the file path
        self.data_file_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.data_file_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.data_file_entry.grid(row=0, column=1, padx=10, sticky="e")

        self.browse_button_voxels_file = ctk.CTkButton(
            self.root,
            text="Upload Voxels File (Optional)",
            command=self.browse_voxels_file,
            font=("Helvetica", 18),
        )
        self.browse_button_voxels_file.grid(
            row=1, column=0, pady=20, padx=10, sticky="w"
        )

        # Entry widget to display the file path
        self.voxels_file_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.voxels_file_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.voxels_file_entry.grid(row=1, column=1, padx=10, sticky="e")

        self.y_column_label = ctk.CTkComboBox(
            self.root,
            values=["NIHSS Score", "Performance"],
            variable=self.y_column_type,
            font=("Helvetica", 18),
            width=200,
        )
        self.y_column_label.grid(row=2, column=0, padx=10, pady=20, sticky="w")

        self.y_column_entry = ctk.CTkEntry(
            self.root, textvariable=self.y_column, font=("Helvetica", 18), width=200
        )
        self.y_column_entry.grid(row=2, column=1, padx=10, pady=20, sticky="e")

        self.ml_model_label = ctk.CTkLabel(
            self.root, text="Machine Learning Model: ", font=("Helvetica", 18)
        )
        self.ml_model_label.grid(row=3, column=0, padx=10, sticky="w")

        self.ml_model_combobox = ctk.CTkComboBox(
            self.root,
            values=list(ml_models.models.keys()),
            variable=self.ml_model,
            width=300,
            font=("Helvetica", 18),
        )
        self.ml_model_combobox.grid(row=3, column=1, padx=10, sticky="e")

        self.run_iterative_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Run Iterative",
            variable=self.run_iterative_var,
            font=("Helvetica", 18),
        )
        self.run_iterative_checkbox.grid(
            row=4, column=0, columnspan=1, pady=20, padx=10, sticky="ew"
        )

        self.run_network_interaction_2d_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Run Network Interaction",
            variable=self.run_network_interaction_2d_var,
            font=("Helvetica", 18),
        )
        self.run_network_interaction_2d_checkbox.grid(
            row=4, column=1, columnspan=1, pady=20, padx=10, sticky="ew"
        )

        self.binarize_data_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Binarize Data",
            variable=self.binarize_data_var,
            font=("Helvetica", 18),
        )
        self.binarize_data_checkbox.grid(
            row=5, column=0, columnspan=2, pady=20, padx=10, sticky="ew"
        )

        # Submit button
        self.msa_button = ctk.CTkButton(
            self.root,
            text="Run MSA",
            command=self.click_run_button,
            font=("Helvetica", 18),
        )
        self.msa_button.grid(
            row=6, column=0, columnspan=2, pady=20, padx=10, sticky="ew"
        )

        self.progress_bar = ctk.CTkProgressBar(self.root, mode="indeterminate")
        self.progress_bar.grid(
            row=7, column=0, columnspan=2, pady=20, padx=10, sticky="ew"
        )

        self.text = ctk.CTkTextbox(
            self.root,
            height=150,
            border_width=4,
            border_color="#003660",
            border_spacing=10,
            fg_color="silver",
            text_color="black",
            font=("Helvetica", 18),
            wrap="word",  # Char default, word, none
            activate_scrollbars=True,
            scrollbar_button_color="blue",
            scrollbar_button_hover_color="red",
            state="disabled",
        )
        self.text.grid(row=8, column=0, columnspan=2, pady=20, padx=10, sticky="ew")

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
        try:
            self.call_msa_pipline()

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            self.msa_button.configure(state="normal")
            self.progress_bar.stop()

    def call_msa_pipline(self):
        self.text.insert("end", "Running MSA\n")
        self.msa_button.configure(state="disabled")
        msa = MSA(
            self.data_file_path.get(),
            self.y_column.get(),
            self.y_column_type.get(),
            self.ml_model.get(),
            self.voxels_file_path.get(),
            self.progress_bar,
            self.root,
            bool(self.binarize_data_var.get()),
            bool(self.run_network_interaction_2d_var.get()),
        )
        msa.prepare_data()
        self.text.insert("end", "Prepared Data\n")

        msa.train_model()

        if self.run_iterative_var.get():
            msa.run_iterative_msa()
            self.text.insert("end", "Finished Running Iterative MSA\n")
            msa.save_iterative()
        else:
            msa.run_msa()
            self.text.insert("end", "Finished Running MSA\n")
            msa.save()

        msa.plot_msa(bool(self.run_iterative_var.get()))

        if self.run_network_interaction_2d_var.get():
            msa.run_interaction_2d()
            msa.plot_network_interaction()
            self.text.insert("end", "Finished Running Network Interaction\n")

        self.text.insert("end", "Saved Results\n")
