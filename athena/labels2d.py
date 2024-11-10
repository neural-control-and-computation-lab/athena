import os
import glob
import json
import time
import threading
import tkinter as tk
import tkinter.ttk as ttk  # For the progress bar
import av
import cv2 as cv
import numpy as np
import mediapipe as mp
import concurrent.futures
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    PoseLandmarker,
    HandLandmarkerOptions,
    PoseLandmarkerOptions,
    RunningMode
)
from multiprocessing import Manager, set_start_method
import importlib.resources as pkg_resources
from athena import models  # Replace with the actual package path

hand_model_path = pkg_resources.path(models, "hand_landmarker.task")
pose_model_path = pkg_resources.path(models, "pose_landmarker_full.task")


def createvideo(image_folder, extension, fps, output_folder, video_name):
    """
    Compiles a set of images into a video in sequential order.

    Parameters:
        image_folder (str): The directory containing the images.
        extension (str): The file extension of the images (e.g., '.png', '.jpg').
        fps (float): Frames per second for the output video.
        output_folder (str): The directory where the output video will be saved.
        video_name (str): The filename of the output video.

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get the list of images and sort them by the frame number
    images = [img for img in os.listdir(image_folder) if img.endswith(extension)]
    if not images:
        print(f"No images found in {image_folder}.")
        return

    images.sort()
    # Read the first image to get the frame dimensions
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv.imread(first_frame_path)
    if frame is None:
        print(f"Failed to read the first image at {first_frame_path}.")
        return
    height, width, layers = frame.shape

    # Set the codec and create the video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_folder, video_name)
    video = cv.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each image to the video file
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        video.write(frame)

    # Release the video writer
    video.release()
    cv.destroyAllWindows()


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """
    Draws pose landmarks on an image.

    Parameters:
        rgb_image (np.ndarray): The input RGB image.
        detection_result (PoseLandmarkerResult): The result object containing pose landmarks.

    Returns:
        np.ndarray: The image with pose landmarks drawn.
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for pose_landmarks in pose_landmarks_list:
        # Convert pose landmarks to protobuf format for drawing
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in pose_landmarks
        ])

        # Draw the pose landmarks on the image
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    """
    Draws hand landmarks and handedness on an image.

    Parameters:
        rgb_image (np.ndarray): The input RGB image.
        detection_result (HandLandmarkerResult): The result object containing hand landmarks and handedness.

    Returns:
        np.ndarray: The image with hand landmarks and handedness drawn.
    """
    MARGIN = 20  # pixels
    FONT_SIZE = 3
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize both
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        handedness = handedness[0]  # Access the first item in the handedness list

        # Convert hand landmarks to protobuf format for drawing
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in hand_landmarks
        ])

        # Draw the hand landmarks
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Get the top-left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image
        cv.putText(
            annotated_image,
            f"{handedness.category_name}",
            (text_x, text_y),
            cv.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA
        )

    return annotated_image


def readcalibration(calibration_files):
    """
    Reads camera calibration parameters from YAML files.

    Parameters:
        calibration_files (list of str): List of file paths to the camera calibration YAML files.

    Returns:
        tuple: A tuple containing:
            - extrinsics (list of np.ndarray): List of 4x4 extrinsic matrices for each camera.
            - intrinsics (list of np.ndarray): List of 3x3 intrinsic matrices for each camera.
            - dist_coeffs (list of np.ndarray): List of distortion coefficients for each camera.
    """
    extrinsics = []
    intrinsics = []
    dist_coeffs = []

    for cam_file in calibration_files:
        # Grab camera calibration parameters
        cam_yaml = cv.FileStorage(cam_file, cv.FILE_STORAGE_READ)
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


def transformationmatrix(R, t):
    """
    Creates a 4x4 homogeneous transformation matrix from rotation and translation.

    Parameters:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-element translation vector.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    """
    T = np.concatenate((R, t.reshape(3, 1)), axis=1)
    T = np.vstack((T, [0, 0, 0, 1]))
    return T


def process_camera(cam, input_stream, gui_options, cam_mats_intrinsic, cam_dist_coeffs, undistort_map, display_width,
                   display_height, progress_queue):
    """
    Processes a single camera stream and saves keypoints directly to disk.

    Parameters:
        cam (int): Camera index.
        input_stream (str): Path to the input video file.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        undistort_map (tuple): Precomputed undistortion maps for the camera (map1, map2).
        display_width (int): Width for displaying frames.
        display_height (int): Height for displaying frames.
        progress_queue (multiprocessing.Queue): Queue for communicating progress.

    Returns:
        int: The camera index.
    """
    print(f"Starting processing for camera {cam}")

    # Extract options from the gui_options dictionary
    save_images = gui_options['save_images_mp']
    use_gpu = gui_options['use_gpu']
    process_to_frame = gui_options['fraction_frames']
    outdir_images_trial = gui_options['outdir_images_trial']
    outdir_data2d_trial = gui_options['outdir_data2d_trial']
    hand_confidence = gui_options['hand_confidence']
    pose_confidence = gui_options['pose_confidence']

    # Paths for saving data
    data_save_path = os.path.join(outdir_data2d_trial, f'cam{cam}')
    os.makedirs(data_save_path, exist_ok=True)

    # Initialize file paths for saving keypoints
    kpts_cam_l_file = os.path.join(data_save_path, '2Dlandmarks_left.npy')
    kpts_cam_r_file = os.path.join(data_save_path, '2Dlandmarks_right.npy')
    kpts_body_file = os.path.join(data_save_path, '2Dlandmarks_body.npy')
    kpts_cam_l_world_file = os.path.join(data_save_path, '2Dworldlandmarks_left.npy')
    kpts_cam_r_world_file = os.path.join(data_save_path, '2Dworldlandmarks_right.npy')
    kpts_body_world_file = os.path.join(data_save_path, '2Dworldlandmarks_body.npy')
    confidence_hand_file = os.path.join(data_save_path, 'handedness_score.npy')

    # Prepare lists to store keypoints
    kpts_cam_l = []
    kpts_cam_r = []
    kpts_body = []
    kpts_cam_l_world = []
    kpts_cam_r_world = []
    kpts_body_world = []
    handscore = []

    # Set GPU delegate based on user selection
    delegate = mp.tasks.BaseOptions.Delegate.GPU if use_gpu else mp.tasks.BaseOptions.Delegate.CPU

    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=hand_confidence,
        min_hand_presence_confidence=hand_confidence,
        min_tracking_confidence=hand_confidence
    )
    pose_options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=pose_confidence,
        min_pose_presence_confidence=pose_confidence,
        min_tracking_confidence=pose_confidence
    )

    # Create PyAV container and video stream
    container = av.open(input_stream)
    video_stream = container.streams.video[0]

    # Get video FPS and total frames
    if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
        fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
    else:
        fps = 30.0  # Default FPS if not available

    total_frames = video_stream.frames
    if total_frames == 0:
        # Estimate total frames if not available
        duration_s = container.duration / av.time_base
        total_frames = int(duration_s * fps)

    # Initialize HandLandmarker and PoseLandmarker for this camera
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Define expected lengths
    num_hand_keypoints = 21
    num_body_keypoints = 33

    # Start time for processing FPS calculation
    start_time = time.time()
    last_fps_time = start_time
    frames_since_last_fps = 0

    # Initialize frame number
    framenum = 0
    max_frames = int(process_to_frame * total_frames)

    prev_timestamp_ms = -1  # Initialize previous timestamp

    # Use frame iterator directly
    frame_iter = container.decode(video=0)
    for frame in frame_iter:
        if framenum >= max_frames:
            break

        # Convert PyAV frame to NumPy array in RGB format
        frame_array = frame.to_ndarray(format='rgb24')

        # Undistort image using precomputed maps
        map1, map2 = undistort_map
        frame_array = cv.remap(frame_array, map1, map2, interpolation=cv.INTER_LINEAR)

        # Convert to RGBA if using GPU
        if use_gpu:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGBA,
                data=cv.cvtColor(frame_array, cv.COLOR_RGB2RGBA)
            )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)

        # Calculate timestamp for MediaPipe in milliseconds
        timestamp_ms = int(framenum * 1000 / fps)

        # Ensure timestamps are strictly increasing
        if timestamp_ms <= prev_timestamp_ms:
            timestamp_ms = prev_timestamp_ms + 1  # Increment by 1 ms
        prev_timestamp_ms = timestamp_ms

        # Hand Landmarks detection
        try:
            hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in hand_landmarker.detect_for_video: {e}")
            hand_results = mp.tasks.vision.HandLandmarkerResult([], [], [])

        # Pose Landmarks detection
        try:
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in pose_landmarker.detect_for_video: {e}")
            pose_results = mp.tasks.vision.PoseLandmarkerResult([], [])

        # Hand Landmarks processing
        frame_keypoints_l = []
        frame_keypoints_r = []
        frame_keypoints_l_world = []
        frame_keypoints_r_world = []
        frame_handscore = [-1, -1]  # Default -1 (not detected)

        if hand_results.hand_landmarks:
            for hand_landmarks, hand_world_landmarks, handedness_list in zip(
                hand_results.hand_landmarks,
                hand_results.hand_world_landmarks,
                hand_results.handedness
            ):
                handedness = handedness_list[0]

                # Process Left and Right hands separately
                if handedness.category_name == 'Left':
                    frame_keypoints_l = [[
                        int(frame_array.shape[1] * hand_landmark.x),
                        int(frame_array.shape[0] * hand_landmark.y),
                        hand_landmark.z,
                        hand_landmark.visibility,
                        hand_landmark.presence
                    ] for hand_landmark in hand_landmarks]
                    frame_keypoints_l_world = [[
                        hand_world_landmark.x,
                        hand_world_landmark.y,
                        hand_world_landmark.z,
                        hand_world_landmark.visibility,
                        hand_world_landmark.presence
                    ] for hand_world_landmark in hand_world_landmarks]
                    frame_handscore[0] = handedness.score
                else:
                    frame_keypoints_r = [[
                        int(frame_array.shape[1] * hand_landmark.x),
                        int(frame_array.shape[0] * hand_landmark.y),
                        hand_landmark.z,
                        hand_landmark.visibility,
                        hand_landmark.presence
                    ] for hand_landmark in hand_landmarks]
                    frame_keypoints_r_world = [[
                        hand_world_landmark.x,
                        hand_world_landmark.y,
                        hand_world_landmark.z,
                        hand_world_landmark.visibility,
                        hand_world_landmark.presence
                    ] for hand_world_landmark in hand_world_landmarks]
                    frame_handscore[1] = handedness.score

                # Draw hand landmarks on the image
                frame_array = draw_hand_landmarks_on_image(frame_array, hand_results)

        # Pose Landmarks processing
        frame_keypoints_body = []
        frame_keypoints_body_world = []
        if pose_results.pose_landmarks:
            for pose_landmarks, pose_world_landmarks in zip(
                pose_results.pose_landmarks,
                pose_results.pose_world_landmarks
            ):
                frame_keypoints_body = [[
                    int(body_landmark.x * frame_array.shape[1]),
                    int(body_landmark.y * frame_array.shape[0]),
                    body_landmark.z,
                    body_landmark.visibility,
                    body_landmark.presence
                ] for body_landmark in pose_landmarks]
                frame_keypoints_body_world = [[
                    body_world_landmark.x,
                    body_world_landmark.y,
                    body_world_landmark.z,
                    body_world_landmark.visibility,
                    body_world_landmark.presence
                ] for body_world_landmark in pose_world_landmarks]

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
            frame_keypoints_body_world += [[-1, -1, -1, -1, -1]] * (
                num_body_keypoints - len(frame_keypoints_body_world)
            )

        # Append keypoints
        kpts_cam_l.append(frame_keypoints_l)
        kpts_cam_r.append(frame_keypoints_r)
        kpts_body.append(frame_keypoints_body)
        kpts_cam_l_world.append(frame_keypoints_l_world)
        kpts_cam_r_world.append(frame_keypoints_r_world)
        kpts_body_world.append(frame_keypoints_body_world)

        # Handedness confidence
        handscore.append(frame_handscore)

        # Resize the frame
        resized_frame = cv.resize(frame_array, (display_width, display_height))

        # Save images if needed
        if save_images:
            save_path = os.path.join(outdir_images_trial, f'cam{cam}', f'frame{framenum:04d}.png')
            result = cv.imwrite(save_path, cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR))
            if not result:
                print(f"Failed to save frame {framenum:04d} for cam {cam} at {save_path}")

        # Increment frame number
        framenum += 1
        frames_since_last_fps += 1

        # Send progress update every N frames
        if framenum % 10 == 0 or framenum == max_frames:
            progress = (framenum / max_frames) * 100
            progress_queue.put({'cam': cam, 'progress': progress})

        # Calculate processing FPS every N frames
        if frames_since_last_fps >= 10 or framenum == max_frames:
            current_time = time.time()
            elapsed_time = current_time - last_fps_time
            if elapsed_time > 0:
                processing_fps = frames_since_last_fps / elapsed_time
                progress_queue.put({'cam': cam, 'fps': processing_fps})
            last_fps_time = current_time
            frames_since_last_fps = 0

    # After processing all frames, convert lists to NumPy arrays and save to disk
    kpts_cam_l = np.array(kpts_cam_l)
    kpts_cam_r = np.array(kpts_cam_r)
    kpts_body = np.array(kpts_body)
    kpts_cam_l_world = np.array(kpts_cam_l_world)
    kpts_cam_r_world = np.array(kpts_cam_r_world)
    kpts_body_world = np.array(kpts_body_world)
    confidence_hand = np.array(handscore)

    # Save the results to disk
    np.save(kpts_cam_l_file, kpts_cam_l)
    np.save(kpts_cam_r_file, kpts_cam_r)
    np.save(kpts_body_file, kpts_body)
    np.save(kpts_cam_l_world_file, kpts_cam_l_world)
    np.save(kpts_cam_r_world_file, kpts_cam_r_world)
    np.save(kpts_body_world_file, kpts_body_world)
    np.save(confidence_hand_file, confidence_hand)

    # Release resources
    hand_landmarker.close()
    pose_landmarker.close()
    container.close()

    # Send a completion message for this camera
    progress_queue.put({'cam': cam, 'done': True})

    return cam


def run_mediapipe(input_streams, gui_options, cam_mats_intrinsic, cam_dist_coeffs, outdir_images_trial,
                  outdir_data2d_trial, trialname, display_width=450, display_height=360, progress_queue=None):
    """
    Processes multiple camera streams in parallel using multiprocessing.

    Parameters:
        input_streams (list): List of input video file paths.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        outdir_images_trial (str): Output directory for images.
        outdir_data2d_trial (str): Output directory for data.
        trialname (str): Name of the trial.
        display_width (int, optional): Width for displaying frames. Defaults to 450.
        display_height (int, optional): Height for displaying frames. Defaults to 360.
        progress_queue (multiprocessing.Queue, optional): Queue for communicating progress.

    Returns:
        None
    """
    num_processes = gui_options.get('num_processes', os.cpu_count())

    # Precompute undistortion maps
    undistort_maps = []
    frame_width, frame_height = None, None

    # Get frame dimensions and undistort maps from the first frame of each camera
    for cam, input_stream in enumerate(input_streams):
        container = av.open(input_stream)
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frame_array = frame.to_ndarray(format='rgb24')
                frame_height, frame_width = frame_array.shape[:2]
                # Set up undistort maps
                map1, map2 = cv.initUndistortRectifyMap(
                    cam_mats_intrinsic[cam],
                    cam_dist_coeffs[cam],
                    None,
                    cam_mats_intrinsic[cam],
                    (frame_width, frame_height),
                    cv.CV_16SC2
                )
                undistort_maps.append((map1, map2))
                break  # Only need one frame
            break  # Only need one packet
        container.close()

    # Update gui_options with additional information
    gui_options['outdir_images_trial'] = outdir_images_trial
    gui_options['outdir_data2d_trial'] = outdir_data2d_trial
    gui_options['trialname'] = trialname

    # Create a copy of gui_options without GUI elements
    gui_options_no_gui = gui_options.copy()
    # Remove any GUI elements from gui_options if they are present
    gui_elements = ['fps_label', 'progress_bar', 'root', 'fps_value', 'progress_var']
    for key in gui_elements:
        gui_options_no_gui.pop(key, None)  # Safely remove if present

    # Use ProcessPoolExecutor to process cameras in parallel
    total_cameras = len(input_streams)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for cam, input_stream in enumerate(input_streams):
            futures.append(executor.submit(
                process_camera,
                cam,
                input_stream,
                gui_options_no_gui,
                cam_mats_intrinsic,
                cam_dist_coeffs,
                undistort_maps[cam],
                display_width,
                display_height,
                progress_queue  # Pass the progress_queue to child processes
            ))

        # Wait for all processes to complete
        for future in concurrent.futures.as_completed(futures):
            cam = future.result()
            print(f"Camera {cam} processing complete.")


def main(gui_options_json):
    # Set the multiprocessing start method to 'spawn'
    set_start_method('spawn')

    gui_options = json.loads(gui_options_json)

    # Get the GUI options
    idfolders = gui_options['idfolders']
    main_folder = gui_options['main_folder']

    # Create the main root window for progress
    progress_root = tk.Tk()
    progress_root.title("Processing Progress")
    progress_root.attributes("-topmost", True)

    window_width = 500  # Width of the window
    window_height = 100  # Height of the window
    screen_width = progress_root.winfo_screenwidth()
    screen_height = progress_root.winfo_screenheight()
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    progress_root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    # Add progress bar and FPS label to progress_root
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        progress_root,
        orient="horizontal",
        mode="determinate",
        variable=progress_var,
        maximum=100
    )
    progress_bar.pack(pady=10, padx=10, fill=tk.X)

    fps_value = tk.DoubleVar()
    fps_label = tk.Label(progress_root, text="Avg FPS: 0")
    fps_label.pack(pady=5)

    # Create a Manager for shared objects
    manager = Manager()
    progress_queue = manager.Queue()
    cam_progress = manager.dict()  # For tracking progress per camera
    cam_fps = manager.dict()       # For tracking FPS per camera

    def update_progress():
        """
        Updates the progress bar and FPS label based on messages from the progress queue.
        """
        try:
            # Try to get messages from the queue without blocking
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                if 'progress' in progress and 'cam' in progress:
                    cam = progress['cam']
                    cam_progress[cam] = progress['progress']
                    # Calculate total progress
                    total_progress = sum(cam_progress.values()) / (len(cam_progress) * 100) * 100
                    progress_var.set(total_progress)
                    progress_bar["value"] = progress_var.get()
                if 'fps' in progress and 'cam' in progress:
                    cam = progress['cam']
                    fps = progress['fps']
                    cam_fps[cam] = fps
                    # Calculate average FPS
                    avg_fps = sum(cam_fps.values()) / len(cam_fps)
                    fps_value.set(avg_fps)
                    fps_label.config(text=f"Avg FPS: {fps_value.get():.2f}")
                if 'done' in progress:
                    if progress.get('cam') is not None:
                        cam = progress['cam']
                        cam_progress[cam] = 100
                    else:
                        fps_label.config(text="Processing Complete")
                        # Instead of quitting immediately, set a flag
                        update_progress.processing_done = True
        except Exception as e:
            print(f"Error in update_progress: {e}")
        if not update_progress.processing_done or not progress_queue.empty():
            # Schedule the function to run again after 100 milliseconds
            progress_root.after(100, update_progress)
        else:
            # All processing is done, and the queue is empty; now we can quit
            progress_root.quit()

    # Initialize the processing_done flag
    update_progress.processing_done = False

    def process_videos():
        """
        Processes the videos for each trial and updates the progress queue.
        """
        if idfolders:
            trialfolders = sorted(idfolders)
            outdir_images = os.path.join(main_folder, 'images/')
            outdir_video = os.path.join(main_folder, 'videos_processed/')
            outdir_data2d = os.path.join(main_folder, 'landmarks/')

            print(f"Selected Folder: {main_folder}")
            print(f"Save Images: {gui_options['save_images_mp']}")
            print(f"Save Video: {gui_options['save_video_mp']}")
            print(f"Use GPU: {gui_options['use_gpu']}")

            if gui_options['save_video_mp'] and not gui_options['save_images_mp']:
                print("Cannot save video without saving images. Adjusting settings.")
                gui_options['save_video_mp'] = False  # Adjust the setting in gui_options

            # Gather camera calibration parameters
            calfiles = sorted(glob.glob(os.path.join(main_folder, 'calibration', '*.yaml')))
            cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)

            total_trials = len(trialfolders)
            processed_trials = 0

            for trial in trialfolders:
                trialname = os.path.basename(trial)
                print(f"Processing trial: {trialname}")

                vidnames = sorted(glob.glob(os.path.join(trial, '*.avi')))
                ncams = len(vidnames)

                container = av.open(vidnames[0])
                video_stream = container.streams.video[0]
                # Get video FPS and total frames
                if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
                    fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
                else:
                    fps = 30.0  # Default FPS if not available
                container.close()

                outdir_images_trial = os.path.join(outdir_images, trialname)
                outdir_video_trial = os.path.join(outdir_video, trialname)
                outdir_data2d_trial = os.path.join(outdir_data2d, trialname)

                os.makedirs(outdir_data2d_trial, exist_ok=True)
                if gui_options['save_images_mp']:
                    os.makedirs(outdir_images_trial, exist_ok=True)
                    for cam in range(ncams):
                        os.makedirs(os.path.join(outdir_images_trial, f'cam{cam}'), exist_ok=True)

                # Initialize cam_progress and cam_fps for each camera
                for cam in range(ncams):
                    cam_progress[cam] = 0.0
                    cam_fps[cam] = 0.0

                # Call run_mediapipe with progress_queue
                run_mediapipe(
                    vidnames,
                    gui_options,
                    cam_mats_intrinsic,
                    cam_dist_coeffs,
                    outdir_images_trial,
                    outdir_data2d_trial,
                    trialname,
                    progress_queue=progress_queue
                )

                if gui_options['save_images_mp'] and gui_options['save_video_mp']:
                    os.makedirs(outdir_video_trial, exist_ok=True)
                    for cam in range(ncams):
                        imagefolder = os.path.join(outdir_images_trial, f'cam{cam}')
                        createvideo(
                            image_folder=imagefolder,
                            extension='.png',
                            fps=fps,
                            output_folder=outdir_video_trial,
                            video_name=f'cam{cam}.mp4'
                        )

                # Update progress per trial
                processed_trials += 1
                total_progress = (processed_trials / total_trials) * 100
                progress_queue.put({'progress': total_progress})

            # When done, put 'done' in the queue
            print("Processing complete, putting 'done' into queue")
            progress_queue.put({'done': True})

    # Start the processing in a separate thread
    threading.Thread(target=process_videos).start()

    # Start updating the progress window
    update_progress()

    # Start the Tkinter main loop for the progress window
    progress_root.mainloop()