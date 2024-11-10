import json
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, MULTIPLE, Toplevel, Scrollbar
from pathlib import Path
import sys
import importlib.resources as pkg_resources  # For accessing package resources
from athena.labels2d import main as labels2d_main

package_name = 'athena'

def select_folder_and_options(root):
    """
    Launches a GUI for selecting the main folder and setting processing options.
    """
    # Initialize variables accessible within nested functions
    idfolders = []       # List of selected recording folders
    main_folder = None   # Path to the main folder
    num_cameras = 0      # Number of cameras determined from calibration files

    def select_folder():
        """
        Opens a dialog to select the main folder and allows selection of recordings.
        """
        nonlocal main_folder
        try:
            # Use the main root window as the parent
            main_folder = filedialog.askdirectory(
                title="Select Main Folder",
                parent=root
            )

            if not main_folder:
                return  # Exit if no folder is selected

            # Update the folder label in the main window
            folder_label.config(text="Selected Folder: " + str(main_folder))

            # Update the number of cameras based on calibration files
            nonlocal num_cameras
            calibration_folder = os.path.join(main_folder, 'calibration')
            if os.path.exists(calibration_folder):
                num_cameras = len(glob.glob(os.path.join(calibration_folder, '*.yaml')))
            else:
                num_cameras = 0

            if num_cameras == 0:
                messagebox.showerror("Error", "No calibration files found in 'calibration' folder.")
                return  # Exit if no calibration files found

            # Adjust the number of processes based on the number of cameras
            max_processes = min(os.cpu_count(), num_cameras)
            if max_processes < 1:
                max_processes = 1  # Ensure at least one process
            num_processes_scale.configure(to=max_processes)
            num_processes_scale.set(max_processes)

            # Get the list of subfolders (recordings) in the videos directory
            videos_folder = Path(main_folder) / 'videos'
            if not videos_folder.exists():
                messagebox.showerror("Error", f"No 'videos' folder found in {main_folder}.")
                return  # Exit if videos folder not found

            subfolders = [f.name for f in videos_folder.iterdir() if f.is_dir()]
            if not subfolders:
                messagebox.showerror("Error", f"No subfolders found in 'videos' folder.")
                return  # Exit if no subfolders in videos folder

            # Create a new window for selecting recordings
            subfolder_window = Toplevel(root)
            subfolder_window.title("Select Recordings")

            # Center the subfolder window on the screen
            subfolder_window.update_idletasks()
            window_width = subfolder_window.winfo_width()
            window_height = subfolder_window.winfo_height()
            screen_width = subfolder_window.winfo_screenwidth()
            screen_height = subfolder_window.winfo_screenheight()
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 2) - (window_height // 2)
            subfolder_window.geometry(f'+{x}+{y}')

            # Create a Listbox for selecting subfolders
            listbox = Listbox(subfolder_window, selectmode=MULTIPLE)
            for folder in subfolders:
                listbox.insert("end", folder)

            # Add a scrollbar to the Listbox
            scrollbar = Scrollbar(subfolder_window)
            scrollbar.pack(side="right", fill="y")
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            listbox.pack()

            def save_selection():
                """
                Saves the selected subfolders to 'idfolders' and closes the selection window.
                """
                selected_indices = listbox.curselection()
                idfolders.clear()
                idfolders.extend([str(videos_folder / subfolders[i]) for i in selected_indices])
                subfolder_window.destroy()

            # Button to confirm selection of recordings
            confirm_button = tk.Button(subfolder_window, text="Select", command=save_selection)
            confirm_button.pack()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            print(f"An error occurred in select_folder: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def on_submit():
        """
        Gathers all user-selected options and runs the processing scripts.
        """
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
        gui_options['hand_centroid_lfc'] = slider_hand_lfc.get()
        gui_options['all_landmarks_lfc'] = slider_all_lfc.get()

        if not gui_options['idfolders']:
            messagebox.showerror("Error", "No recordings selected!")
            return  # Prevent further execution if no folders are selected

        gui_options_json = json.dumps(gui_options)

        # Execute the processing scripts based on user selections
        try:
            if gui_options['run_mediapipe']:
                print('Running Mediapipe.')
                # Import the module and call its main function
                labels2d_main(gui_options_json)

            if gui_options['run_triangulation']:
                print('Triangulating.')
                # Import the module and call its main function
                from athena.triangulaterefine import main as triangulate_main
                triangulate_main(gui_options_json)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing: {e}")
            print(f"An error occurred in on_submit: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return  # Exit the function if an error occurs

        # GUI remains open after processing

    def quit_application():
        """
        Exits the application gracefully by closing the main window.
        """
        root.quit()
        root.destroy()

    def update_save_images_mp(*args):
        """
        Ensures that 'Save Images' is checked if 'Save Video' is checked in Mediapipe options.
        """
        if var_save_video_mp.get():
            var_save_images_mp.set(True)
        elif not var_save_images_mp.get() and var_save_video_mp.get():
            var_save_video_mp.set(False)

    def update_save_video_mp(*args):
        """
        Ensures that 'Save Video' is unchecked if 'Save Images' is unchecked in Mediapipe options.
        """
        if not var_save_images_mp.get() and var_save_video_mp.get():
            var_save_video_mp.set(False)

    def update_save_images_triangulation(*args):
        """
        Ensures that 'Save Images' is checked if 'Save Video' is checked in Triangulation options.
        """
        if var_save_video_triangulation.get():
            var_save_images_triangulation.set(True)
        elif not var_save_images_triangulation.get() and var_save_video_triangulation.get():
            var_save_video_triangulation.set(False)

    def update_save_video_triangulation(*args):
        """
        Ensures that 'Save Video' is unchecked if 'Save Images' is unchecked in Triangulation options.
        """
        if not var_save_images_triangulation.get() and var_save_video_triangulation.get():
            var_save_video_triangulation.set(False)

    # Now set up the GUI components
    gui_options = {}  # Dictionary to hold GUI options

    # Set window size and center it on the screen
    window_width, window_height = 600, 650
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    position_x, position_y = (screen_width - window_width) // 2, (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    # Variables for storing GUI state
    var_run_mediapipe = tk.BooleanVar(value=True)
    var_save_images_mp = tk.BooleanVar(value=False)
    var_save_video_mp = tk.BooleanVar(value=False)
    var_use_gpu = tk.BooleanVar(value=False)
    var_triangulation = tk.BooleanVar(value=True)
    var_save_images_triangulation = tk.BooleanVar(value=False)
    var_save_video_triangulation = tk.BooleanVar(value=False)

    # Trace changes to 'Save Video' and 'Save Images' variables
    var_save_video_mp.trace_add('write', update_save_images_mp)
    var_save_images_mp.trace_add('write', update_save_video_mp)

    var_save_video_triangulation.trace_add('write', update_save_images_triangulation)
    var_save_images_triangulation.trace_add('write', update_save_video_triangulation)

    # General settings frame
    frame_general = tk.LabelFrame(root, text="General Settings", padx=10, pady=10)
    frame_general.pack(fill="x", padx=10, pady=5)

    # Button to select the main folder
    btn_select_folder = tk.Button(frame_general, text="Select Folder", command=select_folder)
    btn_select_folder.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    # Label to display the selected folder
    folder_label = tk.Label(frame_general, text="Folder: Not selected", anchor='w', wraplength=450)
    folder_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

    # Run Mediapipe frame
    frame_mediapipe = tk.LabelFrame(root, text="Run Mediapipe", padx=10, pady=10)
    frame_mediapipe.pack(fill="x", padx=10, pady=5)

    # Checkbox to run Mediapipe
    chk_run_mediapipe = tk.Checkbutton(
        frame_mediapipe, text="Run Mediapipe", font=("Arial", 15, "bold"),
        variable=var_run_mediapipe
    )
    chk_run_mediapipe.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    # Configure column weights for equal resizing
    frame_mediapipe.columnconfigure(0, weight=1)
    frame_mediapipe.columnconfigure(1, weight=1)

    # Slider for fraction of frames to process
    slider_fraction_frames = tk.Scale(
        frame_mediapipe, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
        label="Fraction of frames to process"
    )
    slider_fraction_frames.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
    slider_fraction_frames.set(1.0)

    # Slider for number of parallel processes
    num_cpus = os.cpu_count()
    num_processes_scale = tk.Scale(
        frame_mediapipe, from_=1, to=num_cpus, orient=tk.HORIZONTAL,
        label="Number of parallel processes"
    )
    num_processes_scale.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
    num_processes_scale.set(num_cpus)

    # Checkbox for GPU processing
    chk_use_gpu = tk.Checkbutton(frame_mediapipe, text="GPU Processing", variable=var_use_gpu)
    chk_use_gpu.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    # Slider for hand detection confidence
    slider_handconf = tk.Scale(
        frame_mediapipe, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
        label="Minimum hand detection confidence"
    )
    slider_handconf.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
    slider_handconf.set(0.9)

    # Slider for pose detection confidence
    slider_poseconf = tk.Scale(
        frame_mediapipe, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
        label="Minimum pose detection confidence"
    )
    slider_poseconf.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
    slider_poseconf.set(0.9)

    # Checkboxes for saving images and videos during Mediapipe processing
    chk_save_images_mp = tk.Checkbutton(frame_mediapipe, text="Save Images", variable=var_save_images_mp)
    chk_save_images_mp.grid(row=4, column=0, padx=5, pady=5, sticky="w")
    chk_save_video_mp = tk.Checkbutton(frame_mediapipe, text="Save Video", variable=var_save_video_mp)
    chk_save_video_mp.grid(row=4, column=1, padx=5, pady=5, sticky="w")

    # Triangulation and Refinement frame
    frame_triangulation = tk.LabelFrame(root, text="Triangulation and Refinement", padx=10, pady=10)
    frame_triangulation.pack(fill="x", padx=10, pady=5)

    # Checkbox to run triangulation
    chk_triangulation = tk.Checkbutton(
        frame_triangulation, text="Run Triangulation and Refinement", font=("Arial", 15, "bold"),
        variable=var_triangulation
    )
    chk_triangulation.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

    # Configure column weights for equal resizing
    frame_triangulation.columnconfigure(0, weight=1)
    frame_triangulation.columnconfigure(1, weight=1)

    # Hand centroid low-frequency cutoff
    slider_hand_lfc = tk.Scale(
        frame_triangulation, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL,
        label="Hand centroid: low-freq cutoff (Hz)"
    )
    slider_hand_lfc.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    slider_hand_lfc.set(5)

    # All landmark low-frequency cutoff
    slider_all_lfc = tk.Scale(
        frame_triangulation, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL,
        label="All points: low-freq cutoff (Hz)"
    )
    slider_all_lfc.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    slider_all_lfc.set(20)

    # Checkboxes for saving images and videos during triangulation
    chk_save_images_triangulation = tk.Checkbutton(
        frame_triangulation, text="Save Images", variable=var_save_images_triangulation
    )
    chk_save_images_triangulation.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    chk_save_video_triangulation = tk.Checkbutton(
        frame_triangulation, text="Save Video", variable=var_save_video_triangulation
    )
    chk_save_video_triangulation.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    # Button to start processing
    btn_submit = tk.Button(root, text="GO", command=on_submit)
    btn_submit.pack(pady=10)

    # Button to quit the application
    btn_quit = tk.Button(root, text="QUIT", command=quit_application)
    btn_quit.pack(pady=5)

def main():
    # Create the root window
    root = tk.Tk()
    root.title("ATHENA: Automatically Tracking Hands Expertly with No Annotations")
    # Withdraw the root window
    root.withdraw()

    # Create the splash screen on top of the root window
    splash = Toplevel(root)
    splash.overrideredirect(True)

    # Load the logo image using importlib.resources
    try:
        with pkg_resources.path(package_name, 'logo.png') as logo_path:
            logo_image = tk.PhotoImage(file=str(logo_path))
    except Exception as e:
        messagebox.showerror("Error", f"Unable to load logo image: {e}")
        print(f"Error loading logo image: {e}", file=sys.stderr)
        logo_image = None  # Or set a default image

    if logo_image:
        window_width = logo_image.width()
        window_height = logo_image.height()
    else:
        window_width, window_height = 400, 300  # Default size if logo not loaded

    # Get the screen width and height
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    # Calculate position x and y coordinates
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    splash.geometry(f'{window_width}x{window_height}+{x}+{y}')

    if logo_image:
        label = tk.Label(splash, image=logo_image)
        label.pack()
        # Keep a reference
        label.image = logo_image
    else:
        label = tk.Label(splash, text="ATHENA")
        label.pack(expand=True)

    # After duration milliseconds, destroy splash and continue
    def close_splash():
        splash.destroy()
        # Deiconify the root window
        root.deiconify()
        select_folder_and_options(root)

    splash.after(2000, close_splash)
    # Start the event loop
    root.mainloop()

if __name__ == '__main__':
    main()