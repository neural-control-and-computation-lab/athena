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
    images.sort(key=lambda img: int(img.split('frame')[-1].split('.')[0]))

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


def run_mediapipe(input_streams, gui_options, display_width=450, display_height=360):
    # Extract options from the gui_options dictionary
    save_images = gui_options['save_images']
    monitor_images = gui_options['monitor_images']
    use_gpu = gui_options['use_gpu']
    process_to_frame = gui_options['slider_value']
    fps_label = gui_options['fps_label']  # Get the FPS label from the dictionary
    progress_bar = gui_options['progress_bar']  # Get the progress bar from the dictionary

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

    # Create a list of cameras based on input_streams
    caps = [cv.VideoCapture(stream) for stream in input_streams]
    fps = caps[0].get(cv.CAP_PROP_FPS)  # Assume all streams have the same frame rate
    total_frames = int(caps[0].get(cv.CAP_PROP_FRAME_COUNT))

    # Initialize HandLandmarker and PoseLandmarker for each camera
    hand_landmarkers = [HandLandmarker.create_from_options(hand_options) for _ in caps]
    pose_landmarkers = [PoseLandmarker.create_from_options(pose_options) for _ in caps]

    # Containers for detected key points for each camera
    kpts_cam_l = [[] for _ in caps]
    kpts_cam_r = [[] for _ in caps]
    kpts_body = [[] for _ in caps]

    # Containers for detected keypoints (world coordinates) for each camera
    kpts_cam_l_world = [[] for _ in caps]
    kpts_cam_r_world = [[] for _ in caps]
    kpts_body_world = [[] for _ in caps]

    # Containers for handedness confidence scores
    handscore = [[] for _ in caps]

    # Define expected lengths
    num_hand_keypoints = 21
    num_body_keypoints = 33

    # Initialize frame number for each camera
    framenums = [0] * len(caps)

    # Initialize a counter for forced updates
    frame_counter = 0

    # Track start time for FPS calculation
    start_time = time.time()

    while framenums[0] <= int(process_to_frame*total_frames):
        frames = [cap.read() for cap in caps]
        if not all(ret for ret, _ in frames):
            break

        # Process each frame for hand and pose landmarks
        for cam, (_, frame) in enumerate(frames):

            # Undistort image
            frame = cv.undistort(frame, cam_mats_intrinsic[cam], cam_dist_coeffs[cam])

            # Convert frame from BGR to RGB as required by Mediapipe
            if use_gpu:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv.cvtColor(frame, cv.COLOR_BGR2RGBA))
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            timestamp_ms = int(framenums[cam] * 1000 / fps)

            # Hand Landmarks detection with the camera's specific detector
            hand_results = hand_landmarkers[cam].detect_for_video(mp_image, timestamp_ms)
            frame_keypoints_l = []
            frame_keypoints_r = []
            frame_keypoints_body = []
            frame_keypoints_l_world = []
            frame_keypoints_r_world = []
            frame_keypoints_body_world = []
            frame_handscore = [-1, -1]  # Default -1 (not detected)

            if hand_results.hand_landmarks:
                # Iterate over all detected hands
                for hand_landmarks, hand_world_landmarks, handedness_list in zip(hand_results.hand_landmarks, hand_results.hand_world_landmarks, hand_results.handedness):
                    handedness = handedness_list[0]

                    # Loop over each hand landmark (21 landmarks per hand)
                    if handedness.category_name == 'Left':  # Check if the hand is labeled as 'Left'
                        frame_keypoints_l = [[int(frame.shape[1] * hand_landmark.x),
                                              int(frame.shape[0] * hand_landmark.y),
                                              hand_landmark.z,
                                              hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_landmarks]
                        frame_keypoints_l_world = [[int(frame.shape[1] * hand_landmark.x),
                                                    int(frame.shape[0] * hand_landmark.y),
                                                    hand_landmark.z,
                                                    hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_world_landmarks]
                        frame_handscore[0] = handedness.score

                    else:  # Right hand
                        frame_keypoints_r = [[int(frame.shape[1] * hand_landmark.x),
                                              int(frame.shape[0] * hand_landmark.y),
                                              hand_landmark.z,
                                              hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_landmarks]
                        frame_keypoints_r_world = [[int(frame.shape[1] * hand_landmark.x),
                                                    int(frame.shape[0] * hand_landmark.y),
                                                    hand_landmark.z,
                                                    hand_landmark.visibility, hand_landmark.presence] for hand_landmark in hand_world_landmarks]
                        frame_handscore[1] = handedness.score

                    # Draw hand landmarks on the image
                    frame = draw_hand_landmarks_on_image(frame, hand_results)

            # Pose Landmarks detection with the camera's specific detector
            pose_results = pose_landmarkers[cam].detect_for_video(mp_image, timestamp_ms)

            if pose_results.pose_landmarks:
                # Iterate through each pose in the list (even if it's a single pose)
                for pose_landmarks, pose_world_landmarks in zip(pose_results.pose_landmarks, pose_results.pose_world_landmarks):
                    frame_keypoints_body = [[int(body_landmark.x * frame.shape[1]),
                                             int(body_landmark.y * frame.shape[0]),
                                             body_landmark.z,
                                             body_landmark.visibility, body_landmark.presence] for body_landmark in pose_landmarks]
                    frame_keypoints_body_world = [[int(body_landmark.x * frame.shape[1]),
                                                   int(body_landmark.y * frame.shape[0]),
                                                   body_landmark.z,
                                                   body_landmark.visibility, body_landmark.presence] for body_landmark in pose_world_landmarks]

                    # Draw pose landmarks on the image
                    frame = draw_pose_landmarks_on_image(frame, pose_results)

            # Ensure each list has the correct number of keypoints by padding
            if len(frame_keypoints_l) < num_hand_keypoints:
                frame_keypoints_l += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l))
                frame_keypoints_l_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l_world))
            if len(frame_keypoints_r) < num_hand_keypoints:
                frame_keypoints_r += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r))
                frame_keypoints_r_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r_world))
            if len(frame_keypoints_body) < num_body_keypoints:
                frame_keypoints_body += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body))
                frame_keypoints_body_world += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body_world))

            # Append padded key points for each hand (left and right) and body
            kpts_cam_l[cam].append(frame_keypoints_l)
            kpts_cam_r[cam].append(frame_keypoints_r)
            kpts_body[cam].append(frame_keypoints_body)
            kpts_cam_l_world[cam].append(frame_keypoints_l_world)
            kpts_cam_r_world[cam].append(frame_keypoints_r_world)
            kpts_body_world[cam].append(frame_keypoints_body_world)

            # Handedness confidence
            handscore[cam].append(frame_handscore)

            frame_counter += 1  # Increment the frame counter
            # Only update FPS label and progress bar every 10 frames
            if frame_counter % 10 == 0:
                # FPS calculation
                elapsed_time = time.time() - start_time
                fps_value = framenums[0] / elapsed_time if elapsed_time > 0 else 0
                fps_value = fps_value * len(caps)  # Multiply by number of cameras to get true FPS
                fps_label.config(text=f"FPS: {fps_value:.2f}")  # Update the FPS label

                # Update progress bar
                progress_value = (framenums[0] / int(process_to_frame*total_frames)) * 100
                progress_bar["value"] = progress_value

                # Force the GUI to update (but only periodically)
                gui_options['root'].update_idletasks()  # Update the FPS label and progress bar
                gui_options['root'].update()  # Allow the GUI to process events

            # Resize the frame
            resized_frame = cv.resize(frame, (display_width, display_height))

            # display if monitoring
            if monitor_images:

                # Set a grid layout for displaying each camera feed
                window_x = (cam % 4) * (display_width + 2)  # Adjust for 4 columns
                window_y = (cam // 4) * (display_height + 29)  # Adjust row position

                cv.imshow(f'cam{cam}', resized_frame)
                cv.moveWindow(f'cam{cam}', window_x, window_y)

            # Save images if needed
            if save_images:
                save_path = f'{outdir_images_trial}/cam{cam}/frame{framenums[cam]:04d}.png'
                result = cv.imwrite(save_path, resized_frame)
                if not result:
                    print(f"Failed to save frame {framenums[cam]:04d} for cam {cam} in PNG format at {save_path}")

            # Increment the frame number for the current camera
            framenums[cam] += 1

        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    # Release resources
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    # Close all HandLandmarker and PoseLandmarker instances
    for hand_landmarker, pose_landmarker in zip(hand_landmarkers, pose_landmarkers):
        hand_landmarker.close()
        pose_landmarker.close()

    # Convert lists to NumPy arrays (should now be consistent in shape)
    kpts_cam_l = np.array(kpts_cam_l)  # ncams x nframes x 21 landmarks x 5 outputs (xyz, presence, visibility)
    kpts_cam_r = np.array(kpts_cam_r)
    kpts_body = np.array(kpts_body)  # ncams x nframes x 33 landmarks x 5 outputs (xyz, presence, visibility)
    kpts_cam_l_world = np.array(kpts_cam_l_world)
    kpts_cam_r_world = np.array(kpts_cam_r_world)
    kpts_body_world = np.array(kpts_body_world)
    confidence_hand = np.array(handscore)  # ncams x nframes x 2 hands

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
            save_video = False

    # Gather camera calibration parameters
    calfiles = glob.glob(idfolder + '/calibration/*.yaml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)

    for trial in tqdm(trialfolders):
        trialname = os.path.basename(trial)
        print(trialname)

        vidnames = sorted(glob.glob(os.path.join(trial, '*.avi')))
        ncams = len(vidnames)

        outdir_images_trial = os.path.join(outdir_images, trialname)
        outdir_video_trial = os.path.join(outdir_video, trialname)
        outdir_data2d_trial = os.path.join(outdir_data2d, trialname)

        if gui_options['save_images']:
            os.makedirs(outdir_images_trial, exist_ok=True)
            os.makedirs(outdir_data2d_trial, exist_ok=True)
            for cam in range(ncams):
                os.makedirs(os.path.join(outdir_images_trial, f'cam{cam}'), exist_ok=True)

        kpts_cam_l, kpts_cam_r, kpts_body, kpts_cam_l_world, kpts_cam_r_world, kpts_body_world, confidence_hand = (
            run_mediapipe(vidnames, gui_options))

        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_left'), kpts_cam_l)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_right'), kpts_cam_r)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dlandmarks_body'), kpts_body)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_left'), kpts_cam_l_world)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_right'), kpts_cam_r_world)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_2Dworldlandmarks_body'), kpts_body_world)
        np.save(os.path.join(outdir_data2d_trial, f'{trialname}_handedness_score'), confidence_hand)

        if gui_options['save_images'] and gui_options['save_video']:
            os.makedirs(outdir_video_trial, exist_ok=True)
            for cam in range(ncams):
                imagefolder = os.path.join(outdir_images_trial, f'cam{cam}')
                createvideo(image_folder=imagefolder, extension='.png', fs=60,
                            output_folder=outdir_video_trial, video_name=f'cam{cam}.mp4')

    end = time.time()
    print(f'Time to run code: {end - start} seconds')