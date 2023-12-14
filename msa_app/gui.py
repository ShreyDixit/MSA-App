import sys
import customtkinter as ctk
from tkinter import filedialog
from msa_app import ml_models
from msa_app.msa import MSA

ctk.set_default_color_theme("dark-blue")

class GUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.geometry('900x400')
        self.root.title()
        self.file_path = ctk.StringVar()
        self.create_widget()
        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")

    def create_widget(self):
         # Function to handle the file upload
        self.root.grid_columnconfigure((0, 1), weight=1)

        self.browse_button = ctk.CTkButton(self.root, text="Upload File", command=self.browse_file)
        self.browse_button.grid(row=0, column=0, pady=20, sticky = "w")

        # Entry widget to display the file path
        self.file_entry = ctk.CTkEntry(self.root, textvariable=self.file_path, state='readonly', width=200)
        self.file_entry.grid(row=0, column=1, padx=10, sticky = "e")

        self.y_column_type = ctk.StringVar(value="NIHSS Score")
        self.y_column_label = ctk.CTkComboBox(self.root, values= ["NIHSS Score", "Performance"], variable=self.y_column_type)
        self.y_column_label.grid(row=1, column=0, padx=10, pady=10, sticky = "w")

        self.y_column = ctk.StringVar()  # set initial value
        self.y_column_entry = ctk.CTkEntry(self.root, textvariable=self.y_column)
        self.y_column_entry.grid(row=1, column=1, padx=10, pady=10, sticky = "e")

        self.ml_model_label = ctk.CTkLabel(self.root, text="Machine Learning Model: ")
        self.ml_model_label.grid(row=2, column=0, padx=10, sticky = "w")

        self.ml_model = ctk.StringVar()  # set initial value
        self.ml_model_combobox = ctk.CTkComboBox(self.root, values= list(ml_models.models.keys()), variable=self.ml_model, width=200)
        self.ml_model_combobox.grid(row=2, column=1, padx=10, sticky = "e")

        # Submit button
        self.submit_button = ctk.CTkButton(self.root, text="Submit", command=self.submit_job)
        self.submit_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        self.text = ctk.CTkTextbox(self.root, height=100)
        self.text.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path.set(file_path)

    def submit_job(self):
        msa = MSA(self.file_path.get(), self.y_column.get(), self.y_column_type.get(), self.ml_model.get())
        msa.prepare_data()
        print("Prepared Data")
        msa.train_model()
        print("Trained Model")
        msa.run_msa()
        print("Finished Running MSA")
        msa.save()
        print("Saving File")
        msa.plot()

class TextRedirector(object):
    def __init__(self, widget: ctk.CTkTextbox, tag="stdout"):
        self.widget = widget
        self.tag = tag
        self.current_line = ""

    def write(self, string: str, end='/n', flush=False):
        self.widget.configure(state="normal")
        string = string.replace("\u2588", "|")
        if isinstance(string, str):
            if string == "\r":
                pass
            elif string.startswith("\r"):
                # For string starting with \r, overwrite the current line
                self.widget.delete(self.current_line, "end")
                self.current_line = "end - 1 lines"
                string = string[1:]  # Remove the \r from the string
                self.widget.insert("end", string, (self.tag,))
            else:
                # For other cases, handle end characters accordingly
                self.current_line = "end"
                self.widget.insert(self.current_line, f"\n{string}", (self.tag,))
        self.widget.configure(state="disabled")
        if flush:
            self.flush()

    def flush(self):
        # Perform any actions necessary to flush the output
        self.widget.update_idletasks()
        # Optionally, you might do other flush-related actions here
