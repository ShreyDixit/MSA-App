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
        self.advanced_options = AdvancedOptions(self)

    def initialize_variables(self):
        self.data_file_path = ctk.StringVar()
        self.score_file_path = ctk.StringVar()
        self.voxels_file_path = ctk.StringVar()
        self.output_folder_path = ctk.StringVar()
        self.ml_model_var = ctk.StringVar(value=ml_models.list_of_ml_models[2])
        self.run_network_interaction_2d_var = ctk.IntVar()
        self.binarize_data_var = ctk.IntVar()
        self.add_rob_if_not_present_var = ctk.IntVar()
        self.run_iterative_var = ctk.IntVar()
        self.score_type_var = ctk.StringVar(value="Deficit")

    def configure_root(self):
        self.root.geometry("600x700")
        self.root.title("Lesion-Symptom Mapping using MSA")

    def create_widget(self):
        # Function to handle the file upload
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.setup_roi_file()
        self.setup_score_file()
        self.setup_voxels_file()
        self.setup_score_type()
        self.setup_ml_models()
        self.setup_optional_checkboxes()
        self.setup_output_folder()
        self.setup_msa_button()
        self.setup_progressbar()
        self.setup_output_textbox()
        self.setup_advanced_toolboxes()

    def setup_score_type(self):
        self.score_type_label = ctk.CTkButton(
            self.root,
            text="Type of Score: ",
            command=self.browse_data_file,
            font=("Helvetica", 18),
        )
        self.score_type_label.grid(row=3, column=0, pady=12, padx=10, sticky="w")

        self.segemented_button = ctk.CTkSegmentedButton(
            self.root,
            values=["Deficit", "Performance"],
            variable=self.score_type_var,
            font=("Helvetica", 18),
        )
        self.segemented_button.grid(
            row=3, column=1, columnspan=2, pady=12, padx=10, sticky="ew"
        )

    def setup_advanced_toolboxes(self):
        self.advanced_toggle_text = ctk.StringVar(value="Advanced Options  +")
        self.advanced_toggle_label = ctk.CTkLabel(
            self.root,
            textvariable=self.advanced_toggle_text,
            font=("Helvetica", 18),
            cursor="hand2",  # Change cursor to indicate clickable
        )
        self.advanced_toggle_label.grid(row=11, column=0, pady=10, padx=10, sticky="w")
        self.advanced_toggle_label.bind("<Button-1>", self.toggle_advanced_options)

    def setup_output_textbox(self):
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
        self.text.grid(row=10, column=0, columnspan=2, pady=12, padx=10, sticky="ew")

    def setup_progressbar(self):
        self.progress_bar = ctk.CTkProgressBar(self.root, mode="indeterminate")
        self.progress_bar.grid(
            row=9, column=0, columnspan=2, pady=12, padx=10, sticky="ew"
        )

    def setup_msa_button(self):
        self.msa_button = ctk.CTkButton(
            self.root,
            text="Run MSA",
            command=self.click_run_button,
            font=("Helvetica", 18),
        )
        self.msa_button.grid(
            row=8, column=0, columnspan=2, pady=12, padx=10, sticky="ew"
        )

    def setup_output_folder(self):
        self.browse_button_output_folder = ctk.CTkButton(
            self.root,
            text="Choose Output Folder",
            command=self.browse_output_folder,
            font=("Helvetica", 18),
        )
        self.browse_button_output_folder.grid(
            row=7, column=0, pady=12, padx=10, sticky="w"
        )

        # Entry widget to display the file path
        self.output_folder_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.output_folder_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.output_folder_entry.grid(row=7, column=1, padx=10, sticky="e")

    def setup_optional_checkboxes(self):
        self.run_iterative_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Run Iterative",
            variable=self.run_iterative_var,
            font=("Helvetica", 18),
            command=self.run_iterative_event,
        )
        self.run_iterative_checkbox.grid(
            row=5, column=0, columnspan=1, pady=12, padx=10, sticky="ew"
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
            row=5, column=1, columnspan=1, pady=12, padx=10, sticky="ew"
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
            row=6, column=0, columnspan=2, pady=12, padx=10, sticky="ew"
        )

        self.add_rob_if_not_present_checkbox = ctk.CTkSwitch(
            self.root,
            onvalue=1,
            offvalue=0,
            text="Add ROB if not present",
            variable=self.add_rob_if_not_present_var,
            font=("Helvetica", 18),
            command=self.add_rob_if_not_present_event,
        )

        self.add_rob_if_not_present_checkbox.grid(
            row=6, column=1, columnspan=2, pady=12, padx=10, sticky="ew"
        )

    def add_rob_if_not_present_event(self):
        if self.add_rob_if_not_present_var.get() == 0:
            self.run_iterative_var.set(0)

    def run_iterative_event(self):
        if self.run_iterative_var.get() == 1:
            self.add_rob_if_not_present_var.set(1)

    def setup_ml_models(self):
        self.ml_model_label = ctk.CTkLabel(
            self.root, text="Machine Learning Model: ", font=("Helvetica", 18)
        )
        self.ml_model_label.grid(row=4, column=0, padx=10, sticky="w")

        self.ml_model_combobox = ctk.CTkComboBox(
            self.root,
            values=ml_models.list_of_ml_models,
            variable=self.ml_model_var,
            width=300,
            font=("Helvetica", 18),
            command=self.check_if_full_msa_selected,
        )
        self.ml_model_combobox.grid(row=4, column=1, padx=10, sticky="e")

    def check_if_full_msa_selected(self, choice):
        if choice == ml_models.list_of_ml_models[-1]:
            self.advanced_options.full_msa_var.set(1)
        else:
            self.advanced_options.full_msa_var.set(0)

    def setup_voxels_file(self):
        self.browse_button_voxels_file = ctk.CTkButton(
            self.root,
            text="Upload Voxels File (Optional)",
            command=self.browse_voxels_file,
            font=("Helvetica", 18),
        )
        self.browse_button_voxels_file.grid(
            row=2, column=0, pady=12, padx=10, sticky="w"
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

    def setup_score_file(self):
        self.browse_button_score_file = ctk.CTkButton(
            self.root,
            text="Upload Score File",
            command=self.browse_score_file,
            font=("Helvetica", 18),
        )
        self.browse_button_score_file.grid(
            row=1, column=0, pady=12, padx=10, sticky="w"
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

    def setup_roi_file(self):
        self.browse_button_data_file = ctk.CTkButton(
            self.root,
            text="Upload Lesion Data",
            command=self.browse_data_file,
            font=("Helvetica", 18),
        )
        self.browse_button_data_file.grid(row=0, column=0, pady=12, padx=10, sticky="w")

        # Entry widget to display the file path
        self.data_file_entry = ctk.CTkEntry(
            self.root,
            textvariable=self.data_file_path,
            state="readonly",
            width=200,
            font=("Helvetica", 18),
        )
        self.data_file_entry.grid(row=0, column=1, padx=10, sticky="e")

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
            self.root.geometry("600x700")
        else:
            self.advanced_toggle_text.set(
                "Advanced Options  -"
            )  # Show minus symbol when visible
            self.advanced_options.frame.grid()
            self.root.geometry("600x850")

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
        self.text.insert("end", "Preparing Data\n")
        self.msa_button.configure(state="disabled")
        msa = MSA(
            data_file_path=self.data_file_path.get(),
            voxels_file_path=self.voxels_file_path.get(),
            score_file_path=self.score_file_path.get(),
            model_name=self.ml_model_var.get(),
            progress_bar=self.progress_bar,
            root=self.root,
            binarize_data=bool(self.binarize_data_var.get()),
            run_interaction_2d=bool(self.run_network_interaction_2d_var.get()),
            is_score_performance=self.score_type_var.get() == "Performance",
            random_seed=self.advanced_options.random_seed_var.get(),
            num_permutation=self.advanced_options.num_permutation_var.get(),
            add_rob_if_not_present=bool(self.add_rob_if_not_present_var.get()),
            full_msa=bool(self.advanced_options.full_msa_var.get()),
        )
        msa.prepare_data()
        self.text.insert("end", "Training Model\n")

        msa.train_model()

        if self.run_iterative_var.get():
            self.text.insert("end", "Running Iterative MSA\n")
            msa.run_iterative_msa()
        else:
            msa.run_msa()
            self.text.insert("end", "Running MSA\n")

        msa.plot_msa(bool(self.run_iterative_var.get()))

        if self.run_network_interaction_2d_var.get():
            self.text.insert("end", "Running Network Interaction\n")
            msa.run_interaction_2d()
            msa.plot_network_interaction()

        if self.run_iterative_var.get():
            msa.save_iterative(self.output_folder_path.get())
        else:
            msa.save(self.output_folder_path.get())

        self.text.insert("end", "Saved Results\n")


class AdvancedOptions:
    def __init__(self, parent: GUI):
        self.parent = parent
        self.frame = ctk.CTkFrame(parent.root)
        self.random_seed_var = ctk.IntVar(value=2810)
        self.num_permutation_var = ctk.IntVar(value=1000)
        self.full_msa_var = ctk.IntVar(value=0)
        self.create_widgets()

    def create_widgets(self):
        self.frame.grid_columnconfigure((0, 1), weight=1)
        # Example advanced option: Number of Iterations for Randomized Search
        self.setup_random_seed_field()
        self.setup_num_permutation_field()
        self.setup_full_msa_checkbox()

    def setup_random_seed_field(self):
        self.random_seed_label = ctk.CTkLabel(
            self.frame, text="Random Seed: ", font=("Helvetica", 18)
        )
        self.random_seed_label.grid(row=0, column=0, padx=10, pady=12, sticky="w")

        self.random_seed_entry = ctk.CTkEntry(
            self.frame,
            width=100,
            font=("Helvetica", 18),
            textvariable=self.random_seed_var,
        )
        self.random_seed_entry.grid(row=0, column=1, padx=10, pady=12, sticky="e")

    def setup_num_permutation_field(self):
        self.num_permutation_label = ctk.CTkLabel(
            self.frame, text="Num Permutation: ", font=("Helvetica", 18)
        )
        self.num_permutation_label.grid(row=1, column=0, padx=10, pady=12, sticky="w")

        self.num_permutation_entry = ctk.CTkEntry(
            self.frame,
            width=100,
            font=("Helvetica", 18),
            textvariable=self.num_permutation_var,
        )
        self.num_permutation_entry.grid(row=1, column=1, padx=10, pady=12, sticky="e")

    def setup_full_msa_checkbox(self):
        self.full_msa_checkbox = ctk.CTkSwitch(
            self.frame,
            onvalue=1,
            offvalue=0,
            text="Perform Full MSA",
            variable=self.full_msa_var,
            font=("Helvetica", 18),
            command=self.full_msa_event,
        )

        self.full_msa_checkbox.grid(
            row=2, column=0, columnspan=2, pady=12, padx=10, sticky="ew"
        )

    def full_msa_event(self):
        if self.full_msa_var.get() == 1:
            self.parent.ml_model_var.set(ml_models.list_of_ml_models[-1])
        else:
            self.parent.ml_model_var.set(ml_models.list_of_ml_models[2])
