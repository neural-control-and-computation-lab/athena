# Libraries
import cv2 as cv
import glob
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def createvideo(image_folder, extension, fs, output_folder, video_name):
    """
    Compiling a set of images into a video.
    Code adapted from here: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python

    :param image_folder: Pathway containing all images.
    :param extension: Extension of images.
    :param fs: Sampling rate.
    :param output_folder: Pathway of output video.
    :param video_name: Name of output video.
    """

    # Create output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Gather and read images
    images = [img for img in os.listdir(image_folder) if img.endswith(extension)]
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Write video
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Had codec as 0 before - Anaconda gave warning (though videos saved)
    video = cv.VideoWriter(output_folder + '/' + video_name, fourcc, fs, (width, height))
    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    # Clear windows
    cv.destroyAllWindows()
    video.release()


def run_mediapipe(input_streams, save_images=None):

    # Create a list of cameras based on input_streams
    caps = [cv.VideoCapture(stream) for stream in input_streams]

    # Set camera resolution
    for cap in caps:
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(3, height)
        cap.set(4, width)

    # Create hand key points detector objects for each camera
    nhands = 2
    hands = [mp.solutions.hands.Hands(min_detection_confidence=0.50, max_num_hands=nhands, min_tracking_confidence=0.50)
             for cap in caps]
    pose = [mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.50) for cap in caps]

    # Containers for detected key points for each camera
    kpts_cam_l = [[] for cap in caps]
    kpts_cam_r = [[] for cap in caps]
    kpts_body = [[] for cap in caps]

    # Initialize frame number
    framenum = 0

    while True:

        # Read frames from videos
        frames = [cap.read() for cap in caps]

        # If wasn't able to read, break
        if not all(ret for ret, _ in frames):
            break

        # Convert frames from BGR to RGB
        for cam, (_, frame) in enumerate(frames):
            frames[cam] = (True, cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # To improve performance, optionally mark the image as not writeable to pass by reference
        for cam, (_, frame) in enumerate(frames):
            frames[cam] = (True, frame.copy())
            frame.flags.writeable = False

        # Mediapipe output
        results = [hands[cam].process(frame) for cam, (_, frame) in enumerate(frames)]
        results_pose = [pose[cam].process(frame) for cam, (_, frame) in enumerate(frames)]

        # Access 2D hand landmarks (pixel coordinates) if detected (otherwise [-1, -1])
        for cam, (ret, frame) in enumerate(frames):

            if results[cam].multi_hand_landmarks:

                nhands_detected = len(results[cam].multi_hand_landmarks)

                if nhands_detected == 1:
                    hand_landmarks = results[cam].multi_hand_landmarks[0]
                    frame_keypoints = []
                    for p in range(21):
                        pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
                        pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
                        kpts = [pxl_x, pxl_y]
                        frame_keypoints.append(kpts)

                    # Check handedness and append key points
                    # Note: handedness is based on mirror image, so if detected is left then actually a right hand
                    handedness = results[cam].multi_handedness[0].classification[0].label
                    if handedness == 'Left':
                        kpts_cam_r[cam].append(frame_keypoints)
                        kpts_cam_l[cam].append([[-1, -1]] * 21)
                    else:
                        kpts_cam_r[cam].append([[-1, -1]] * 21)
                        kpts_cam_l[cam].append(frame_keypoints)

                elif nhands_detected == 2:

                    frame_keypoints_l = []
                    frame_keypoints_r = []
                    handedness = results[cam].multi_handedness[0].classification[0].label  # Of the first detected hand

                    # Check handedness
                    if handedness == 'Left':
                        hand_landmarks_r = results[cam].multi_hand_landmarks[0]
                        hand_landmarks_l = results[cam].multi_hand_landmarks[1]
                    else:  # Assuming the other is a right hand (reasonable assumption unless multiple people in frame)
                        hand_landmarks_r = results[cam].multi_hand_landmarks[1]
                        hand_landmarks_l = results[cam].multi_hand_landmarks[0]

                    for p in range(21):
                        pxl_x_r = int(round(frame.shape[1] * hand_landmarks_r.landmark[p].x))
                        pxl_y_r = int(round(frame.shape[0] * hand_landmarks_r.landmark[p].y))
                        kpts_r = [pxl_x_r, pxl_y_r]
                        frame_keypoints_r.append(kpts_r)

                        pxl_x_l = int(round(frame.shape[1] * hand_landmarks_l.landmark[p].x))
                        pxl_y_l = int(round(frame.shape[0] * hand_landmarks_l.landmark[p].y))
                        kpts_l = [pxl_x_l, pxl_y_l]
                        frame_keypoints_l.append(kpts_l)

                    # Append key points
                    kpts_cam_r[cam].append(frame_keypoints_r)
                    kpts_cam_l[cam].append(frame_keypoints_l)

                # In rare circumstances, mediapipe will detect more than max # hands detected (> 2 here).
                # I fill with blanks to err on safe side and interpolate after since these are rare cases (1 frame in
                # 1 video out of 1040 recorded videos), and tasks here are quite slow so interpolation works fine.
                # Alternative #1: Pick from the detected hands.
                # Alternative #2: Set static_image_mode=True (treats images as sequence of [possibly unrelated] images)
                # https://stackoverflow.com/questions/76648549/mediapipe-max-num-hands-not-working-properly
                elif nhands_detected > 2:
                    print('WARNING: More than 2 hands detected. Frame #: ' + str(framenum) + '; Cam #: ' + str(cam))
                    kpts_cam_r[cam].append([[-1, -1]] * 21)
                    kpts_cam_l[cam].append([[-1, -1]] * 21)

            else:
                kpts_cam_r[cam].append([[-1, -1]] * 21)
                kpts_cam_l[cam].append([[-1, -1]] * 21)

            if results_pose[cam].pose_landmarks:
                frame_keypoints_body = []
                for i, body_landmark in enumerate(results_pose[cam].pose_landmarks.landmark):
                    pxl_x_body = body_landmark.x * frame.shape[1]
                    pxl_y_body = body_landmark.y * frame.shape[0]
                    pxl_x_body = int(round(pxl_x_body))
                    pxl_y_body = int(round(pxl_y_body))
                    frame_keypoints_body.append([pxl_x_body, pxl_y_body])
                kpts_body[cam].append(frame_keypoints_body)
            else:
                kpts_body[cam].append([[-1, -1]] * 33)

            # Draw hand landmarks
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            if results[cam].multi_hand_landmarks:
                for hand_landmarks in results[cam].multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            if results_pose[cam].pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, results_pose[cam].pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Display and save images
            # cv.imshow(f'cam{cam}', frame)
            if save_images is True:
                cv.imwrite(outdir_images + trialname + '/cam' + str(cam) + '/' + 'frame' + f'{framenum:04d}' + '.png', frame)

        k = cv.waitKey(1)
        if k & 0xFF == 27:  # ESC key
            break

        # Increment frame number
        print(framenum)
        framenum += 1

    # Clear windows
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    # Return 2D hand landmarks
    return np.array(kpts_cam_l), np.array(kpts_cam_r), np.array(kpts_body)


# Run code
if __name__ == '__main__':

    # Counter
    start = time.time()

    # Define working directory
    wdir = Path(os.getcwd())

    # Create a tkinter root window (it won't be displayed)
    root = tk.Tk()
    root.withdraw()

    # Open a dialog box to select participant's folder
    idfolder = filedialog.askdirectory(initialdir=str(wdir))
    id = os.path.basename(os.path.normpath(idfolder))

    # Gather trial folders
    trialfolders = sorted(glob.glob(idfolder + '/videos/*Recording*'))

    # Options for image and video saving (can increase code run time substantially)
    saveimages = True   # Takes up a lot of space, so caution if saving
    savevideo = True    # Can only be done if images saved

    # Output directories
    outdir_images = idfolder + '/images/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data2d = idfolder + '/landmarks/'

    # Make output directories if they do not exist
    if saveimages is True:
        if not os.path.exists(outdir_images):
            os.mkdir(outdir_images)
        if savevideo is True:
            if not os.path.exists(outdir_video):
                os.mkdir(outdir_video)
    if not os.path.exists(outdir_data2d):
        os.mkdir(outdir_data2d)

    for trial in tqdm(trialfolders):

        # Identify trial name
        trialname = os.path.basename(trial)
        print(trialname)

        # Gather trial videos
        vidnames = sorted(glob.glob(trial + '/*.avi'))
        ncams = len(vidnames)

        # Create sub folders for given trial (for each camera) for storing labelled images
        if saveimages is True:
            if not os.path.exists(outdir_images + trialname):
                os.mkdir(outdir_images + trialname)
                for cam in range(ncams):
                    os.mkdir(outdir_images + trialname + '/cam' + str(cam))

        # Obtain 2D landmarks from each camera
        kpts_caml, kpts_camr, kpts_cambody = run_mediapipe(vidnames, save_images=saveimages)

        # Save 2D landmarks as np array (ncameras x nframes x 21 landmarks x 2dimension)
        np.save(outdir_data2d + trialname + '_2Dlandmarks_left', kpts_caml)
        np.save(outdir_data2d + trialname + '_2Dlandmarks_right', kpts_camr)
        np.save(outdir_data2d + trialname + '_2Dlandmarks_body', kpts_cambody)

        # Create video from 2D labelled images
        if saveimages and savevideo is True:

            # Output directory for given trial
            if not os.path.exists(outdir_video + trialname):
                os.mkdir(outdir_video + trialname)

            # Save video
            for cam in range(ncams):
                imagefolder = outdir_images + trialname + '/cam' + str(cam)
                print('Saving video.')
                createvideo(image_folder=imagefolder, extension='.png', fs=60,
                            output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
