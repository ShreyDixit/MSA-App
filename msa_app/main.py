import customtkinter as ctk
from msa_app.gui import GUI

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

if __name__ == "__main__":
    root = ctk.CTk()
    app = GUI(root)
    root.mainloop()
