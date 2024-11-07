# Libraries
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox


def select_folder_and_options():
    """
    Create GUI to select folder and set options for processing.
    """

    def on_submit():

        # Store all GUI options in a dictionary
        gui_options['idfolder'] = idfolder
        gui_options['fraction_frames'] = slider_fraction_frames.get()
        gui_options['num_processes'] = num_processes_scale.get()
        gui_options['use_gpu'] = var_use_gpu.get()
        gui_options['run_mediapipe'] = var_run_mediapipe.get()
        gui_options['save_images_mp'] = var_save_images_mp.get()
        gui_options['save_video_mp'] = var_save_video_mp.get()
        gui_options['hand_confidence'] = slider_handconf.get()
        gui_options['pose_confidence'] = slider_poseconf.get()
        gui_options['run_refine_labels'] = var_refine_labels.get()
        gui_options['save_images_refine'] = var_save_images_refine.get()
        gui_options['save_video_refine'] = var_save_video_refine.get()
        gui_options['run_triangulation'] = var_triangulation.get()
        gui_options['save_images_triangulation'] = var_save_images_triangulation.get()
        gui_options['save_video_triangulation'] = var_save_video_triangulation.get()

        if not gui_options['idfolder']:
            messagebox.showerror("Error", "No folder selected!")
        else:
            root.quit()  # Close the options window

    # Create a tkinter root window
    root = tk.Tk()
    root.title("Options for Processing")

    # Initialize a dictionary to hold the GUI options
    gui_options = {}

    # Set window size and center it on the screen
    window_width = 700
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_x = int((screen_width / 2) - (window_width / 2))
    position_y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    # Default options
    var_run_mediapipe = tk.BooleanVar(value=True)
    var_save_images_mp = tk.BooleanVar(value=False)
    var_save_video_mp = tk.BooleanVar(value=False)
    var_use_gpu = tk.BooleanVar(value=True)
    var_refine_labels = tk.BooleanVar(value=True)
    var_save_images_refine = tk.BooleanVar(value=False)
    var_save_video_refine = tk.BooleanVar(value=False)
    var_triangulation = tk.BooleanVar(value=True)
    var_save_images_triangulation = tk.BooleanVar(value=False)
    var_save_video_triangulation = tk.BooleanVar(value=False)

    # Select Folder
    def select_folder():
        nonlocal idfolder
        idfolder = filedialog.askdirectory(initialdir=str(Path(os.getcwd())))
        folder_label.config(text="Folder: " + idfolder)

    # Section 1: General Settings (Recording folder, GPU, Parallel processing)
    chk_general_options = tk.Label(root, text="General Settings", font=("Arial", 12, "bold"))
    chk_general_options.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    # Select Folder
    idfolder = ""  # Initialize folder path
    btn_select_folder = tk.Button(root, text="Select Folder", command=select_folder)
    btn_select_folder.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    folder_label = tk.Label(root, text="Folder: Not selected", anchor='w', wraplength=450)
    folder_label.grid(row=1, column=1, padx=10, pady=5, columnspan=3, sticky="w")

    # Fraction of frames to process
    slider_fraction_frames = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                      label="Fraction of recordings to process")
    slider_fraction_frames.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    slider_fraction_frames.set(1.0)

    # Add a slider to select the number of parallel processes
    num_cpus = os.cpu_count()
    num_processes_scale = tk.Scale(root, from_=1, to=num_cpus, orient=tk.HORIZONTAL,
                                   label="Number of parallel processes")
    num_processes_scale.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    num_processes_scale.set(num_cpus)

    # GPU
    chk_use_gpu = tk.Checkbutton(root, text="GPU Processing", variable=var_use_gpu)
    chk_use_gpu.grid(row=4, column=0, padx=10, pady=5, sticky="w")

    # Section 2: Mediapipe
    chk_run_mediapipe = tk.Checkbutton(root, text="Run Mediapipe", font=("Arial", 12, "bold"), variable=var_run_mediapipe)
    chk_run_mediapipe.grid(row=5, column=0, padx=10, pady=(10, 0), sticky="w")

    chk_save_images_mp = tk.Checkbutton(root, text="Save Images", variable=var_save_images_mp)
    chk_save_images_mp.grid(row=6, column=0, padx=5, pady=5, sticky="w")

    chk_save_video_mp = tk.Checkbutton(root, text="Save Video", variable=var_save_video_mp)
    chk_save_video_mp.grid(row=6, column=1, padx=5, pady=5, sticky="w")

    slider_handconf = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                               label="Minimum hand detection & tracking confidence")
    slider_handconf.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    slider_handconf.set(0.9)

    slider_poseconf = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                               label="Minimum pose detection & tracking confidence")
    slider_poseconf.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    slider_poseconf.set(0.9)

    # Section 3: Refine Labels
    chk_run_refinelabels = tk.Checkbutton(root, text="Refine Labels", font=("Arial", 12, "bold"), variable=var_refine_labels)
    chk_run_refinelabels.grid(row=9, column=0, padx=10, pady=(10, 0), sticky="w")

    chk_save_images_refine = tk.Checkbutton(root, text="Save Images", variable=var_save_images_refine)
    chk_save_images_refine.grid(row=10, column=0, padx=5, pady=5, sticky="w")

    chk_save_video_refine = tk.Checkbutton(root, text="Save Video", variable=var_save_video_refine)
    chk_save_video_refine.grid(row=10, column=1, padx=5, pady=5, sticky="w")

    # Section 4: Triangulation
    chk_run_refinelabels = tk.Checkbutton(root, text="Triangulation", font=("Arial", 12, "bold"), variable=var_triangulation)
    chk_run_refinelabels.grid(row=11, column=0, padx=10, pady=(10, 0), sticky="w")

    chk_save_images_triangulation = tk.Checkbutton(root, text="Save Images", variable=var_save_images_triangulation)
    chk_save_images_triangulation.grid(row=12, column=0, padx=5, pady=5, sticky="w")

    chk_save_video_triangulation = tk.Checkbutton(root, text="Save Video", variable=var_save_video_triangulation)
    chk_save_video_triangulation.grid(row=12, column=1, padx=5, pady=5, sticky="w")

    # Run button
    btn_submit = tk.Button(root, text="GO", command=on_submit)
    btn_submit.grid(row=13, column=1, padx=10, pady=10)

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    root.mainloop()
    return gui_options


if __name__ == '__main__':

    # Get the GUI options
    gui_options = select_folder_and_options()
    gui_options_json = json.dumps(gui_options)  # To pass as arguments into subprocesses

    # Check folder selected
    idfolder = gui_options['idfolder']
    if not idfolder:
        print("No folder selected. Exiting.")
        sys.exit()

    # Run mediapipe
    if gui_options['run_mediapipe']:

        print('Running Mediapipe.')
        subprocess.run(['python', 'labels2d.py', gui_options_json])

    # Run refine labels
    if gui_options['run_refine_labels']:

        print('Refining labels.')
        subprocess.run(['python', 'labelsrefine.py', gui_options_json])

    # Run triangulation
    if gui_options['run_triangulation']:

        print('Triangulating.')
        subprocess.run(['python', 'triangulation.py', gui_options_json])


