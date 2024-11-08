# Libraries
import json
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, MULTIPLE, Toplevel, Scrollbar
import glob


def select_folder_and_options():
    """
    Create GUI to select folder and set options for processing.
    """
    # Initialize nonlocal variables in outer scope
    idfolders = []
    main_folder = None
    num_cameras = 0

    # Select Folder
    def select_folder():
        nonlocal main_folder
        main_folder = filedialog.askdirectory(initialdir=str(Path(os.getcwd())))
        if not main_folder:
            return  # Exit if no folder is selected

        # Check number of cameras
        nonlocal num_cameras
        num_cameras = len(glob.glob(os.path.join(main_folder, 'calibration', '*.yaml')))

        # Update the scale range for num_processes_scale based on num_cameras
        max_processes = min(os.cpu_count(), num_cameras)
        num_processes_scale.configure(to=max_processes)
        num_processes_scale.set(max_processes)  # Set default to max_processes

        videos_folder = Path(main_folder) / 'videos'
        subfolders = [f.name for f in videos_folder.iterdir() if f.is_dir()]

        # Create the subfolder selection window
        subfolder_window = Toplevel(root)
        subfolder_window.title("Select Recordings")

        # Calculate position for subfolder_window
        main_x, main_y = root.winfo_x(), root.winfo_y()
        main_width = root.winfo_width()
        subfolder_window.geometry(f"+{main_x + main_width + 10}+{main_y}")

        folder_label.config(text="Selected Folder: " + str(main_folder))

        listbox = Listbox(subfolder_window, selectmode=MULTIPLE)
        for folder in subfolders:
            listbox.insert("end", folder)

        scrollbar = Scrollbar(subfolder_window)
        scrollbar.pack(side="right", fill="y")
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        listbox.pack()

        def save_selection():
            selected_indices = listbox.curselection()
            # Update the outer scope idfolders without reassigning
            idfolders.clear()
            idfolders.extend([str(videos_folder / subfolders[i]) for i in selected_indices])
            subfolder_window.destroy()

        confirm_button = tk.Button(subfolder_window, text="Select", command=save_selection)
        confirm_button.pack()

    def on_submit():
        gui_options['idfolders'] = idfolders
        gui_options['main_folder'] = main_folder
        gui_options['fraction_frames'] = slider_fraction_frames.get()
        gui_options['num_processes'] = num_processes_scale.get()
        gui_options['use_gpu'] = var_use_gpu.get()
        gui_options['run_mediapipe'] = var_run_mediapipe.get()
        gui_options['save_images_mp'] = var_save_images_mp.get()
        gui_options['save_video_mp'] = var_save_video_mp.get()
        gui_options['hand_confidence'] = slider_handconf.get()
        gui_options['pose_confidence'] = slider_poseconf.get()
        gui_options['run_triangulation'] = var_triangulation.get()
        gui_options['save_images_triangulation'] = var_save_images_triangulation.get()
        gui_options['save_video_triangulation'] = var_save_video_triangulation.get()

        if not gui_options['idfolders']:
            messagebox.showerror("Error", "No folder selected!")

        gui_options_json = json.dumps(gui_options)

        if gui_options['run_mediapipe']:
            print('Running Mediapipe.')
            subprocess.run(['python', 'labels2d.py', gui_options_json])

        if gui_options['run_triangulation']:
            print('Triangulating.')
            subprocess.run(['python', 'triangulaterefine.py', gui_options_json])


    def quit_application():
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Options for Processing")

    gui_options = {}

    window_width, window_height = 700, 750  # Adjusted height for quit button
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    position_x, position_y = (screen_width - window_width) // 2, (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    var_run_mediapipe = tk.BooleanVar(value=True)
    var_save_images_mp = tk.BooleanVar(value=False)
    var_save_video_mp = tk.BooleanVar(value=False)
    var_use_gpu = tk.BooleanVar(value=False)
    var_triangulation = tk.BooleanVar(value=True)
    var_save_images_triangulation = tk.BooleanVar(value=False)
    var_save_video_triangulation = tk.BooleanVar(value=False)

    chk_general_options = tk.Label(root, text="General Settings", font=("Arial", 12, "bold"))
    chk_general_options.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    btn_select_folder = tk.Button(root, text="Select Folder", command=select_folder)
    btn_select_folder.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    folder_label = tk.Label(root, text="Folder: Not selected", anchor='w', wraplength=450)
    folder_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    slider_fraction_frames = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                                      label="Fraction of recordings to process")
    slider_fraction_frames.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    slider_fraction_frames.set(1.0)

    # Initialize num_processes_scale and set it to update based on the number of cameras
    num_cpus = os.cpu_count()
    num_processes_scale = tk.Scale(root, from_=1, to=num_cpus, orient=tk.HORIZONTAL,
                                   label="Number of parallel processes")
    num_processes_scale.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    num_processes_scale.set(num_cpus)

    chk_use_gpu = tk.Checkbutton(root, text="GPU Processing", variable=var_use_gpu)
    chk_use_gpu.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    chk_run_mediapipe = tk.Checkbutton(root, text="Run Mediapipe", font=("Arial", 12, "bold"),
                                       variable=var_run_mediapipe)
    chk_run_mediapipe.grid(row=5, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

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

    chk_triangulation = tk.Checkbutton(root, text="Triangulation", font=("Arial", 12, "bold"),
                                       variable=var_triangulation)
    chk_triangulation.grid(row=11, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")

    chk_save_images_triangulation = tk.Checkbutton(root, text="Save Images", variable=var_save_images_triangulation)
    chk_save_images_triangulation.grid(row=12, column=0, padx=5, pady=5, sticky="w")
    chk_save_video_triangulation = tk.Checkbutton(root, text="Save Video", variable=var_save_video_triangulation)
    chk_save_video_triangulation.grid(row=12, column=1, padx=5, pady=5, sticky="w")

    # Run button
    btn_submit = tk.Button(root, text="GO", command=on_submit)
    btn_submit.grid(row=13, column=0, columnspan=2, padx=10, pady=10)

    # Add QUIT button at the bottom
    btn_quit = tk.Button(root, text="QUIT", command=quit_application)
    btn_quit.grid(row=14, column=0, columnspan=2, padx=10, pady=10)

    # Configure only two columns for better centering
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()
    return gui_options

if __name__ == '__main__':
    gui_options = select_folder_and_options()
