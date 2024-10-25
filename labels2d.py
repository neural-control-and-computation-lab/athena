# Libraries
import cv2 as cv
import glob
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk  # For the progress bar
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, PoseLandmarker, HandLandmarkerOptions, PoseLandmarkerOptions, RunningMode
import av

def createvideo(image_folder, extension, fs, output_folder, video_name):
    """
    Compiling a set of images into a video in sequential order.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get the list of images and sort them by the frame number
    images = [img for img in os.listdir(image_folder) if img.endswith(extension)]

    if not images:
        print(f"No images found in {image_folder}.")
        return

    # Sort the images based on their filenames (assuming they contain frame numbers)
    images.sort()

    # Read the first image to get the frame dimensions
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Set the codec and create the video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(os.path.join(output_folder, video_name), fourcc, fs, (width, height))

    # Write each image to the video file
    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    # Release the video writer
    video.release()
    cv.destroyAllWindows()


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 20  # pixels
    FONT_SIZE = 3
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize both
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx][0]  # Access the first item in the handedness list

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness.category_name}",  # Now correctly accessing category_name
                   (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image


def readcalibration(calfilepathway):
    """
    Outputs camera calibration parameters.

    :param calibrationfiles: Pathway containing camera calibration parameters within individual yaml files.
    :return: Extrinsic, intrinsic and distortion coefficients.
    """

    extrinsics = []
    intrinsics = []
    dist_coeffs = []

    for cam in range(len(calfilepathway)):

        # Grab camera calibration parameters
        cam_yaml = cv.FileStorage(calfilepathway[cam], cv.FILE_STORAGE_READ)
        cam_int = cam_yaml.getNode("intrinsicMatrix").mat()
        cam_dist = cam_yaml.getNode("distortionCoefficients").mat()
        cam_rotn = cam_yaml.getNode("R").mat().transpose()
        cam_transln = cam_yaml.getNode("T").mat()
        cam_transform = transformationmatrix(cam_rotn, cam_transln)

        # Store calibration parameters
        extrinsics.append(cam_transform)
        intrinsics.append(cam_int.transpose())
        dist_coeffs.append(cam_dist.reshape(-1))

    return extrinsics, intrinsics, dist_coeffs


def run_mediapipe(input_streams, gui_options, cam_mats_intrinsic, cam_dist_coeffs, outdir_images_trial, display_width=450, display_height=360):
    # Extract options from the gui_options dictionary
    save_images = gui_options['save_images']
    monitor_images = gui_options['monitor_images']
    use_gpu = gui_options['use_gpu']
    process_to_frame = gui_options['slider_value']
    fps_label = gui_options['fps_label']  # Get the FPS label from the dictionary
    progress_bar = gui_options['progress_bar']  # Get the progress bar from the dictionary
    root = gui_options['root']  # Get the root Tkinter window

    # Load HandLandmarker and PoseLandmarker models
    hand_model_path = 'hand_landmarker.task'
    pose_model_path = 'pose_landmarker_full.task'

    # Set GPU delegate based on user selection
    delegate = mp.tasks.BaseOptions.Delegate.GPU if use_gpu else mp.tasks.BaseOptions.Delegate.CPU

    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_hands=2
    )
    pose_options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO
    )

    # Create a list of PyAV containers based on input_streams
    containers = [av.open(stream) for stream in input_streams]
    video_streams = [container.streams.video[0] for container in containers]

    # Get FPS and total frames from the first video stream
    fps = video_streams[0].average_rate
    if fps.denominator != 0:
        fps = fps.numerator / fps.denominator
    else:
        fps = 30  # Default FPS if not available
    total_frames = video_streams[0].frames

    # Initialize HandLandmarker and PoseLandmarker for each camera
    hand_landmarkers = [HandLandmarker.create_from_options(hand_options) for _ in containers]
    pose_landmarkers = [PoseLandmarker.create_from_options(pose_options) for _ in containers]

    # Containers for detected key points for each camera
    kpts_cam_l = [[] for _ in containers]
    kpts_cam_r = [[] for _ in containers]
    kpts_body = [[] for _ in containers]

    # Containers for detected keypoints (world coordinates) for each camera
    kpts_cam_l_world = [[] for _ in containers]
    kpts_cam_r_world = [[] for _ in containers]
    kpts_body_world = [[] for _ in containers]

    # Containers for handedness confidence scores
    handscore = [[] for _ in containers]

    # Define expected lengths
    num_hand_keypoints = 21
    num_body_keypoints = 33

    # Initialize frame number for each camera
    framenums = [0] * len(containers)

    # Initialize a counter for forced updates
    frame_counter = 0

    # Track start time for FPS calculation
    start_time = time.time()

    # Precompute undistortion maps
    frame_width, frame_height = None, None
    undistort_maps = []

    # Get frame dimensions and undistort maps from the first frame
    for cam, container in enumerate(containers):
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frame_array = frame.to_ndarray(format='rgb24')
                frame_height, frame_width = frame_array.shape[:2]
                # Set up undistort maps
                map1, map2 = cv.initUndistortRectifyMap(cam_mats_intrinsic[cam], cam_dist_coeffs[cam], None, None, (frame_width, frame_height), cv.CV_16SC2)
                undistort_maps.append((map1, map2))
                break  # Only need one frame
            break  # Only need one packet

    # Reset the containers to start from the beginning
    containers = [av.open(stream) for stream in input_streams]
    frame_iters = [container.decode(video=0) for container in containers]

    max_frames = int(process_to_frame * total_frames)

    while framenums[0] < max_frames:
        frames = []
        for frame_iter in frame_iters:
            try:
                frame = next(frame_iter)
                frames.append(frame)
            except StopIteration:
                frames.append(None)
        if any(frame is None for frame in frames):
            break

        # Process each frame
        for cam, frame in enumerate(frames):
            if frame is None:
                continue
            # Convert PyAV frame to NumPy array in RGB format
            frame_array = frame.to_ndarray(format='rgb24')

            # Undistort image using precomputed maps
            map1, map2 = undistort_maps[cam]
            frame_array = cv.remap(frame_array, map1, map2, interpolation=cv.INTER_LINEAR)

            # Convert to RGBA if using GPU
            if use_gpu:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv.cvtColor(frame_array, cv.COLOR_RGB2RGBA))
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)

            timestamp_ms = int(framenums[cam] * 1000 / fps)

            # Hand Landmarks detection
            hand_results = hand_landmarkers[cam].detect_for_video(mp_image, timestamp_ms)
            frame_keypoints_l = []
            frame_keypoints_r = []
            frame_keypoints_body = []
            frame_keypoints_l_world = []
            frame_keypoints_r_world = []
            frame_keypoints_body_world = []
            frame_handscore = [-1, -1]  # Default -1 (not detected)

            if hand_results.hand_landmarks:
                for hand_landmarks, hand_world_landmarks, handedness_list in zip(
                    hand_results.hand_landmarks, hand_results.hand_world_landmarks, hand_results.handedness):
                    handedness = handedness_list[0]

                    # Process Left and Right hands separately
                    if handedness.category_name == 'Left':
                        frame_keypoints_l = [[int(frame_array.shape[1] * hand_landmark.x),
                                              int(frame_array.shape[0] * hand_landmark.y),
                                              hand_landmark.z,
                                              hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_landmarks]
                        frame_keypoints_l_world = [[int(frame_array.shape[1] * hand_landmark.x),
                                                    int(frame_array.shape[0] * hand_landmark.y),
                                                    hand_landmark.z,
                                                    hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_world_landmarks]
                        frame_handscore[0] = handedness.score
                    else:
                        frame_keypoints_r = [[int(frame_array.shape[1] * hand_landmark.x),
                                              int(frame_array.shape[0] * hand_landmark.y),
                                              hand_landmark.z,
                                              hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_landmarks]
                        frame_keypoints_r_world = [[int(frame_array.shape[1] * hand_landmark.x),
                                                    int(frame_array.shape[0] * hand_landmark.y),
                                                    hand_landmark.z,
                                                    hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_world_landmarks]
                        frame_handscore[1] = handedness.score

                    # Draw hand landmarks on the image
                    frame_array = draw_hand_landmarks_on_image(frame_array, hand_results)

            # Pose Landmarks detection
            pose_results = pose_landmarkers[cam].detect_for_video(mp_image, timestamp_ms)

            if pose_results.pose_landmarks:
                for pose_landmarks, pose_world_landmarks in zip(
                    pose_results.pose_landmarks, pose_results.pose_world_landmarks):
                    frame_keypoints_body = [[int(body_landmark.x * frame_array.shape[1]),
                                             int(body_landmark.y * frame_array.shape[0]),
                                             body_landmark.z,
                                             body_landmark.visibility, body_landmark.presence] for body_landmark in pose_landmarks]
                    frame_keypoints_body_world = [[int(body_landmark.x * frame_array.shape[1]),
                                                   int(body_landmark.y * frame_array.shape[0]),
                                                   body_landmark.z,
                                                   body_landmark.visibility, body_landmark.presence] for body_landmark in pose_world_landmarks]

                    # Draw pose landmarks on the image
                    frame_array = draw_pose_landmarks_on_image(frame_array, pose_results)

            # Ensure correct number of keypoints by padding
            if len(frame_keypoints_l) < num_hand_keypoints:
                frame_keypoints_l += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l))
                frame_keypoints_l_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l_world))
            if len(frame_keypoints_r) < num_hand_keypoints:
                frame_keypoints_r += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r))
                frame_keypoints_r_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r_world))
            if len(frame_keypoints_body) < num_body_keypoints:
                frame_keypoints_body += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body))
                frame_keypoints_body_world += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body_world))

            # Append keypoints
            kpts_cam_l[cam].append(frame_keypoints_l)
            kpts_cam_r[cam].append(frame_keypoints_r)
            kpts_body[cam].append(frame_keypoints_body)
            kpts_cam_l_world[cam].append(frame_keypoints_l_world)
            kpts_cam_r_world[cam].append(frame_keypoints_r_world)
            kpts_body_world[cam].append(frame_keypoints_body_world)

            # Handedness confidence
            handscore[cam].append(frame_handscore)

            # Resize the frame
            resized_frame = cv.resize(frame_array, (display_width, display_height))

            # Display if monitoring
            if monitor_images:
                window_x = (cam % 4) * (display_width + 2)  # Adjust for 4 columns
                window_y = (cam // 4) * (display_height + 29)  # Adjust row position

                cv.imshow(f'cam{cam}', cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR))
                cv.moveWindow(f'cam{cam}', window_x, window_y)

            # Save images if needed
            if save_images:
                save_path = f'{outdir_images_trial}/cam{cam}/frame{framenums[cam]:04d}.png'
                result = cv.imwrite(save_path, cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR))
                if not result:
                    print(f"Failed to save frame {framenums[cam]:04d} for cam {cam} at {save_path}")

            # Increment frame number
            framenums[cam] += 1

        frame_counter += 1  # Increment the frame counter

        # Update FPS label and progress bar every 10 frames
        if frame_counter % 10 == 0:
            # FPS calculation
            elapsed_time = time.time() - start_time
            fps_value = framenums[0] / elapsed_time if elapsed_time > 0 else 0
            fps_value = fps_value * len(containers)  # Multiply by number of cameras
            fps_label.config(text=f"FPS: {fps_value:.2f}")

            # Update progress bar
            progress_value = (framenums[0] / max_frames) * 100
            progress_bar["value"] = progress_value

            # Force the GUI to update
            root.update_idletasks()
            root.update()

        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    # Release resources
    cv.destroyAllWindows()
    for hand_landmarker, pose_landmarker in zip(hand_landmarkers, pose_landmarkers):
        hand_landmarker.close()
        pose_landmarker.close()
    for container in containers:
        container.close()

    # Convert lists to NumPy arrays
    kpts_cam_l = np.array(kpts_cam_l)
    kpts_cam_r = np.array(kpts_cam_r)
    kpts_body = np.array(kpts_body)
    kpts_cam_l_world = np.array(kpts_cam_l_world)
    kpts_cam_r_world = np.array(kpts_cam_r_world)
    kpts_body_world = np.array(kpts_body_world)
    confidence_hand = np.array(handscore)

    return kpts_cam_l, kpts_cam_r, kpts_body, kpts_cam_l_world, kpts_cam_r_world, kpts_body_world, confidence_hand

def select_folder_and_options():
    """
    Create GUI to select folder, set options for saving images/videos, monitoring,
    and add a slider for selecting a value between 0 and 1, along with FPS display and progress bar.
    """
    def on_submit():
        # Store all GUI options in a dictionary
        gui_options['save_images'] = var_save_images.get()
        gui_options['save_video'] = var_save_video.get()
        gui_options['monitor_images'] = var_monitor_images.get()
        gui_options['use_gpu'] = var_use_gpu.get()
        gui_options['slider_value'] = slider.get()  # Get the value from the slider
        gui_options['idfolder'] = idfolder
        gui_options['fps_label'] = fps_label  # Add FPS label to the dictionary
        gui_options['progress_bar'] = progress_bar  # Add progress bar to the dictionary
        gui_options['root'] = root
        if not gui_options['idfolder']:
            messagebox.showerror("Error", "No folder selected!")
        else:
            root.quit()  # Close the window

    # Create a tkinter root window
    root = tk.Tk()
    root.title("Options for Processing")

    # Initialize a dictionary to hold the GUI options
    gui_options = {}

    # Set window size and center it on the screen
    window_width = 500
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_x = int((screen_width / 2) - (window_width / 2))
    position_y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    # Select folder button and folder label
    def select_folder():
        nonlocal idfolder
        idfolder = filedialog.askdirectory(initialdir=str(Path(os.getcwd())))
        folder_label.config(text="Folder: " + idfolder)

    idfolder = ""  # Initialize folder path
    btn_select_folder = tk.Button(root, text="Select Folder", command=select_folder)
    btn_select_folder.grid(row=0, column=0, padx=10, pady=10)

    folder_label = tk.Label(root, text="Folder: Not selected", anchor='w', wraplength=450)
    folder_label.grid(row=0, column=1, padx=10, pady=10, columnspan=3, sticky="w")

    # Checkbox for saving images, videos, and monitoring images
    var_save_images = tk.BooleanVar(value=False)
    var_save_video = tk.BooleanVar(value=False)
    var_monitor_images = tk.BooleanVar(value=False)
    var_use_gpu = tk.BooleanVar(value=True)

    chk_save_images = tk.Checkbutton(root, text="Save Images", variable=var_save_images)
    chk_save_images.grid(row=1, column=0, padx=10, pady=5)

    chk_save_video = tk.Checkbutton(root, text="Save Video", variable=var_save_video)
    chk_save_video.grid(row=1, column=1, padx=10, pady=5)

    chk_monitor_images = tk.Checkbutton(root, text="Monitor Images", variable=var_monitor_images)
    chk_monitor_images.grid(row=1, column=2, padx=10, pady=5)

    chk_use_gpu = tk.Checkbutton(root, text="GPU Processing", variable=var_use_gpu)
    chk_use_gpu.grid(row=2, column=1, padx=10, pady=5)

    slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,
                      label="Fraction of recordings to process")
    slider.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    slider.set(1.0)

    progress_label = tk.Label(root, text="Progress:")
    progress_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", style="TProgressbar")
    progress_bar.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    progress_bar["value"] = 0
    progress_bar["maximum"] = 100

    fps_label = tk.Label(root, text="FPS: 0")
    fps_label.grid(row=7, column=1, padx=10, pady=5)

    btn_submit = tk.Button(root, text="GO", command=on_submit)
    btn_submit.grid(row=4, column=1, padx=10, pady=20)

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    root.mainloop()

    return gui_options


def transformationmatrix(R, t):
    """
    Create a 4x4 transformation matrix based on a rotation vector and translation vector.

    :param R: 3x3 rotation matrix.
    :param t: translation vector.
    :return: 4x4 transformation matrix.
    """

    T = np.concatenate((R, t.reshape(3, 1)), axis=1)
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

if __name__ == '__main__':
    start = time.time()

    gui_options = select_folder_and_options()
    idfolder = gui_options['idfolder']

    if idfolder:
        id = os.path.basename(os.path.normpath(idfolder))

        trialfolders = sorted(glob.glob(os.path.join(idfolder, 'videos/*Recording*')))
        outdir_images = os.path.join(idfolder, 'images/')
        outdir_video = os.path.join(idfolder, 'videos_processed/')
        outdir_data2d = os.path.join(idfolder, 'landmarks/')

        print(f"Selected Folder: {idfolder}")
        print(f"Save Images: {gui_options['save_images']}")
        print(f"Save Video: {gui_options['save_video']}")
        print(f"Use GPU: {gui_options['use_gpu']}")

        if gui_options['save_video'] and not gui_options['save_images']:
            print("Cannot save video without saving images. Adjusting settings.")
            gui_options['save_video'] = False  # Adjust the setting in gui_options

        # Gather camera calibration parameters
        calfiles = glob.glob(os.path.join(idfolder, 'calibration', '*.yaml'))
        cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)

        for trial in tqdm(trialfolders):
            trialname = os.path.basename(trial)
            print(f"Processing trial: {trialname}")

            vidnames = sorted(glob.glob(os.path.join(trial, '*.avi')))
            ncams = len(vidnames)

            outdir_images_trial = os.path.join(outdir_images, trialname)
            outdir_video_trial = os.path.join(outdir_video, trialname)
            outdir_data2d_trial = os.path.join(outdir_data2d, trialname)

            os.makedirs(outdir_data2d_trial, exist_ok=True)
            if gui_options['save_images']:
                os.makedirs(outdir_images_trial, exist_ok=True)
                for cam in range(ncams):
                    os.makedirs(os.path.join(outdir_images_trial, f'cam{cam}'), exist_ok=True)

            # Call run_mediapipe with additional parameters
            kpts_cam_l, kpts_cam_r, kpts_body, kpts_cam_l_world, kpts_cam_r_world, kpts_body_world, confidence_hand = (
                run_mediapipe(vidnames, gui_options, cam_mats_intrinsic, cam_dist_coeffs, outdir_images_trial))

            # Save the results
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_left.npy'), kpts_cam_l)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_right.npy'), kpts_cam_r)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_body.npy'), kpts_body)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_left.npy'), kpts_cam_l_world)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_right.npy'), kpts_cam_r_world)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_body.npy'), kpts_body_world)
            np.save(os.path.join(outdir_data2d_trial, f'{trialname}_handedness_score.npy'), confidence_hand)

            if gui_options['save_images'] and gui_options['save_video']:
                os.makedirs(outdir_video_trial, exist_ok=True)
                for cam in range(ncams):
                    imagefolder = os.path.join(outdir_images_trial, f'cam{cam}')
                    createvideo(image_folder=imagefolder, extension='.png', fs=60,
                                output_folder=outdir_video_trial, video_name=f'cam{cam}.mp4')

    end = time.time()
    print(f'Time to run code: {end - start:.2f} seconds')