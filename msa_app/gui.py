import customtkinter as ctk
from tkinter import filedialog, messagebox
from msa_app import ml_models
from msa_app.msa import MSA
import threading

ctk.set_default_color_theme("dark-blue")


class GUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.configure_root()
        self.initialize_variables()
        self.create_widget()
        self.advanced_options = AdvancedOptions(self.root)

    def initialize_variables(self):
        self.data_file_path = ctk.StringVar()
        self.score_file_path = ctk.StringVar()
        self.voxels_file_path = ctk.StringVar()
        self.output_folder_path = ctk.StringVar()
        self.ml_model = ctk.StringVar()
        self.run_network_interaction_2d_var = ctk.IntVar()
        self.binarize_data_var = ctk.IntVar()
        self.run_iterative_var = ctk.IntVar()
        self.is_score_performance = ctk.IntVar()

    def configure_root(self):
        self.root.geometry("600x800")
        self.root.title("Lesion-Symptom Mapping using MSA")

    def create_widget(self):
        # Function to handle the file upload
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.browse_button_data_file = ctk.CTkButton(
            self.root,
            text="Upload Lesion Data",
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

        self.browse_button_score_file = ctk.CTkButton(
            self.root,
            text="Upload Score File",
            command=self.browse_score_file,
            font=("Helvetica", 18),
        )
        self.browse_button_score_file.grid(
            row=1, column=0, pady=20, padx=10, sticky="w"
        )

        # Entry widget to display the file path
        self.score_file_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.score_file_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.score_file_entry.grid(row=1, column=1, padx=10, sticky="e")

        self.browse_button_voxels_file = ctk.CTkButton(
            self.root,
            text="Upload Voxels File (Optional)",
            command=self.browse_voxels_file,
            font=("Helvetica", 18),
        )
        self.browse_button_voxels_file.grid(
            row=2, column=0, pady=20, padx=10, sticky="w"
        )

        # Entry widget to display the file path
        self.voxels_file_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.voxels_file_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.voxels_file_entry.grid(row=2, column=1, padx=10, sticky="e")

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

        self.binarize_data_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Is Score Performance?",
            variable=self.is_score_performance,
            font=("Helvetica", 18),
        )
        self.binarize_data_checkbox.grid(
            row=5, column=1, columnspan=2, pady=20, padx=10, sticky="ew"
        )

        self.browse_button_output_folder = ctk.CTkButton(
            self.root,
            text="Choose Output Folder",
            command=self.browse_output_folder,
            font=("Helvetica", 18),
        )
        self.browse_button_output_folder.grid(
            row=6, column=0, pady=20, padx=10, sticky="w"
        )

        # Entry widget to display the file path
        self.output_folder_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.output_folder_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.output_folder_entry.grid(row=6, column=1, padx=10, sticky="e")

        # Submit button
        self.msa_button = ctk.CTkButton(
            self.root,
            text="Run MSA",
            command=self.click_run_button,
            font=("Helvetica", 18),
        )
        self.msa_button.grid(
            row=7, column=0, columnspan=2, pady=20, padx=10, sticky="ew"
        )

        self.progress_bar = ctk.CTkProgressBar(self.root, mode="indeterminate")
        self.progress_bar.grid(
            row=8, column=0, columnspan=2, pady=20, padx=10, sticky="ew"
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
        )
        self.text.grid(row=9, column=0, columnspan=2, pady=20, padx=10, sticky="ew")

        # Advanced Options Toggle
        self.advanced_toggle_text = ctk.StringVar(value="Advanced Options  +")
        self.advanced_toggle_label = ctk.CTkLabel(
            self.root,
            textvariable=self.advanced_toggle_text,
            font=("Helvetica", 18),
            cursor="hand2",  # Change cursor to indicate clickable
        )
        self.advanced_toggle_label.grid(row=10, column=0, pady=10, padx=10, sticky="w")
        self.advanced_toggle_label.bind("<Button-1>", self.toggle_advanced_options)

    def browse_file(self, path_var: ctk.StringVar):
        file_path = filedialog.askopenfilename()
        if file_path:
            path_var.set(file_path)

    def browse_data_file(self):
        self.browse_file(self.data_file_path)

    def browse_voxels_file(self):
        self.browse_file(self.voxels_file_path)

    def browse_score_file(self):
        self.browse_file(self.score_file_path)

    def browse_output_folder(self):
        file_path = filedialog.askdirectory()
        if file_path:
            self.output_folder_path.set(file_path)

    def toggle_advanced_options(self, event=None):
        if self.advanced_options.frame.winfo_ismapped():
            self.advanced_toggle_text.set(
                "Advanced Options  +"
            )  # Show plus symbol when hidden
            self.advanced_options.frame.grid_remove()
            self.root.geometry("600x800")
        else:
            self.advanced_toggle_text.set(
                "Advanced Options  -"
            )  # Show minus symbol when visible
            self.advanced_options.frame.grid()
            self.root.geometry("600x900")

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
            data_file_path=self.data_file_path.get(),
            voxels_file_path=self.voxels_file_path.get(),
            score_file_path=self.score_file_path.get(),
            model_name=self.ml_model.get(),
            progress_bar=self.progress_bar,
            root=self.root,
            binarize_data=bool(self.binarize_data_var.get()),
            run_interaction_2d=bool(self.run_network_interaction_2d_var.get()),
            is_score_performance=bool(self.is_score_performance.get()),
        )
        msa.prepare_data()
        self.text.insert("end", "Prepared Data\n")

        msa.train_model()

        if self.run_iterative_var.get():
            msa.run_iterative_msa()
            self.text.insert("end", "Finished Running Iterative MSA\n")
            msa.save_iterative(self.output_folder_path.get())
        else:
            msa.run_msa()
            self.text.insert("end", "Finished Running MSA\n")
            msa.save(self.output_folder_path.get())

        msa.plot_msa(bool(self.run_iterative_var.get()))

        if self.run_network_interaction_2d_var.get():
            msa.run_interaction_2d()
            msa.plot_network_interaction()
            self.text.insert("end", "Finished Running Network Interaction\n")

        self.text.insert("end", "Saved Results\n")


class AdvancedOptions:
    def __init__(self, parent):
        self.frame = ctk.CTkFrame(parent)
        self.create_widgets()

    def create_widgets(self):
        # Example advanced option: Number of Iterations for Randomized Search
        self.n_iterations_label = ctk.CTkLabel(
            self.frame, text="N Iterations:", font=("Helvetica", 18)
        )
        self.n_iterations_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.n_iterations_entry = ctk.CTkEntry(
            self.frame, width=100, font=("Helvetica", 18)
        )
        self.n_iterations_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.n_iterations_label_1 = ctk.CTkLabel(
            self.frame, text="N Iterations:", font=("Helvetica", 18)
        )
        self.n_iterations_label_1.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.n_iterations_entry_2 = ctk.CTkEntry(
            self.frame, width=100, font=("Helvetica", 18)
        )
        self.n_iterations_entry_2.grid(row=1, column=1, padx=5, pady=5, sticky="w")
