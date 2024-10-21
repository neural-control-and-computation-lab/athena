# Libraries
import cv2 as cv
import glob
from labels2d import createvideo
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from labels2d import readcalibration
from triangulation import triangulate_simple


def hex2bgr(hexcode):
    """
    Converts hexadecimal code to BGR (OpenCV reverses the RGB).
    Adapted from here: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python

    :param hexcode: Hexadecimal code
    :return: BGR tuple
    """
    h = hexcode.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    bgr = rgb[::-1]

    return bgr


def undistort_points(points, matrix, dist):
    points = points.reshape(-1, 1, 2)
    out = cv.undistortPoints(points, matrix, dist)
    return out


def project_3d_to_2d(X_world, intrinsic_matrix, extrinsic_matrix):

    # Transform 3D point to camera coordinates
    X_camera = np.dot(extrinsic_matrix, X_world)

    # Project onto the image plane using the intrinsic matrix
    X_image_homogeneous = np.dot(intrinsic_matrix, X_camera[:3])  # Skip the homogeneous coordinate (4th value)

    # Normalize the homogeneous coordinates to get 2D point
    u = X_image_homogeneous[0] / X_image_homogeneous[2]
    v = X_image_homogeneous[1] / X_image_homogeneous[2]

    return np.array([u, v])


def switch_hands(data2d):
    data_2d_switched = data2d.copy()

    # Step #1:
    for cam in range(ncams):

        rwrist = data2d[cam, :, 16, :]
        lwrist = data2d[cam, :, 15, :]
        rhand = data2d[cam, :, 33, :]
        lhand = data2d[cam, :, 54, :]

        norm_rvsr = np.linalg.norm(rwrist - rhand, axis=-1)
        norm_rvsl = np.linalg.norm(rwrist - lhand, axis=-1)
        norm_lvsr = np.linalg.norm(lwrist - rhand, axis=-1)
        norm_lvsl = np.linalg.norm(lwrist - lhand, axis=-1)

        # Condition where hands are switched and both hands present
        condition1 = norm_rvsr > norm_rvsl
        condition2 = rhand[:, 0] != -1
        condition3 = lhand[:, 0] != -1

        # Condition where left hand is mislabelled as right hand and only 1 hand present
        condition4 = norm_lvsr < norm_rvsl
        condition5 = lhand[:, 0] == -1
        condition6 = norm_rvsr > norm_lvsr  # Need this to prevent where RH is only hand and correctly there
        combined_condition = (condition1 & condition2 & condition3) | (condition4 & condition5 & condition6)

        for i, flag in enumerate(combined_condition):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp

    # Step #2: Output 3D points
    data_2d = data_2d_switched.copy()
    data_2d[data_2d == -1] = np.nan
    data_2d = data_2d.reshape((ncams, -1, 2))
    data_2d_undistort = np.empty(data_2d.shape)
    for cam in range(ncams):
        data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                  cam_dist_coeffs[cam]).reshape(len(data_2d[cam]), 2)

    lhand = np.empty((nframes, 3))
    lhand[:] = np.nan
    rhand = np.empty((nframes, 3))
    rhand[:] = np.nan
    data_2d_undistort = data_2d_undistort.reshape((ncams, nframes, nlandmarks, 2))
    handestimate = np.empty((ncams, nframes, 2, 2))
    handestimate[:] = np.nan

    for frame in range(nframes):

        # Identify cams where left hand and left wrist were tracked
        sub_lh = data_2d_undistort[:, frame, 54, :]
        good_lh = ~np.isnan(sub_lh[:, 0])
        sub_lw = data_2d_undistort[:, frame, 15, :]
        good_lw = ~np.isnan(sub_lw[:, 0])
        good_lhlw = good_lh & good_lw

        # Identify cams where right hand and right wrist were tracked
        sub_rh = data_2d_undistort[:, frame, 33, :]
        good_rh = ~np.isnan(sub_rh[:, 0])
        sub_rw = data_2d_undistort[:, frame, 16, :]
        good_rw = ~np.isnan(sub_rw[:, 0])
        good_rhrw = good_rh & good_rw

        # Require at least 2 cameras to have picked up both hand and wrist
        if np.sum(good_lhlw) >= 2:
            lhand[frame] = triangulate_simple(sub_lh[good_lhlw], cam_mats_extrinsic[good_lhlw])

        if np.sum(good_rhrw) >= 2:
            rhand[frame] = triangulate_simple(sub_rh[good_rhrw], cam_mats_extrinsic[good_rhrw])

        for cam in range(ncams):
            lhand_world = np.append(lhand[frame], 1)
            rhand_world = np.append(rhand[frame], 1)
            handestimate[cam, frame, 0, :] = project_3d_to_2d(rhand_world, cam_mats_intrinsic[cam],
                                                              cam_mats_extrinsic[cam])
            handestimate[cam, frame, 1, :] = project_3d_to_2d(lhand_world, cam_mats_intrinsic[cam],
                                                              cam_mats_extrinsic[cam])

    # Step 3
    data_2d_switched[data_2d_switched == -1] = -9999
    for cam in range(ncams):

        rhand = data_2d_switched[cam, :, 33, :]
        lhand = data_2d_switched[cam, :, 54, :]
        rhand_est = handestimate[cam, :, 0, :]
        lhand_est = handestimate[cam, :, 1, :]

        norm_rvsrest = np.linalg.norm(rhand - rhand_est, axis=-1)
        norm_lvsrest = np.linalg.norm(lhand - rhand_est, axis=-1)
        norm_rvslest = np.linalg.norm(rhand - lhand_est, axis=-1)
        norm_lvslest = np.linalg.norm(lhand - lhand_est, axis=-1)

        c1 = norm_lvsrest < norm_rvsrest
        c2 = norm_lvsrest < norm_lvslest

        c3 = norm_rvslest < norm_lvslest
        c4 = norm_rvslest < norm_rvsrest

        c5 = (c1 & c2) | (c3 & c4)

        for i, flag in enumerate(c5):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp
    data_2d_switched[data_2d_switched == -9999] = -1

    return data_2d_switched


def visualizelabels(input_streams, data, display_width=450, display_height=360):
    """
    Draws 2D hand landmarks on videos.

    :param input_streams: List of videos
    :param data: 2D hand landmarks.
    """

    # Create a list of cameras based on input_streams
    caps = [cv.VideoCapture(stream) for stream in input_streams]

    # Set camera resolution
    for cap in caps:
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(3, height)
        cap.set(4, width)

    # Creating links for each digit
    colors = ['#009988', '#009988', '#009988', '#009988',
              '#EE7733', '#EE7733', '#EE7733', '#EE7733',
              '#DDDDDD', '#DDDDDD', '#DDDDDD', '#DDDDDD', '#DDDDDD',
              '#009988', '#009988',
              '#EE7733', '#EE7733',
              '#FDE7EF', '#FDE7EF', '#FDE7EF', '#FDE7EF',
              '#F589B1', '#F589B1', '#F589B1', '#F589B1',
              '#ED2B72', '#ED2B72', '#ED2B72', '#ED2B72',
              '#A50E45', '#A50E45', '#A50E45', '#A50E45',
              '#47061D', '#47061D', '#47061D', '#47061D',
              '#E5F6FF', '#E5F6FF', '#E5F6FF', '#E5F6FF',
              '#80D1FF', '#80D1FF', '#80D1FF', '#80D1FF',
              '#1AACFF', '#1AACFF', '#1AACFF', '#1AACFF',
              '#0072B3', '#0072B3', '#0072B3', '#0072B3',
              '#00314D', '#00314D', '#00314D', '#00314D']

    links = [[0, 1], [1, 2], [2, 3], [3, 7],  # left face
             [0, 4], [4, 5], [5, 6], [6, 8],  # right face
             [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],  # trunk
             [11, 13], [13, 15],  # left arm
             [12, 14], [14, 16],  # right arm
             [33, 34], [34, 35], [35, 36], [36, 37],  # right thumb
             [33, 38], [38, 39], [39, 40], [40, 41],
             [33, 42], [42, 43], [43, 44], [44, 45],
             [33, 46], [46, 47], [47, 48], [48, 49],
             [33, 50], [50, 51], [51, 52], [52, 53],
             [54, 55], [55, 56], [56, 57], [57, 58],  # left thumb
             [54, 59], [59, 60], [60, 61], [61, 62],
             [54, 63], [63, 64], [64, 65], [65, 66],
             [54, 67], [67, 68], [68, 69], [69, 70],
             [54, 71], [71, 72], [72, 73], [73, 74]]

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

        # Access 2D hand landmarks (pixel coordinates) if detected (otherwise [-1, -1])
        for cam, (ret, frame) in enumerate(frames):

            # Draw hand landmarks
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            for number, link in enumerate(links):
                start = link[0]
                end = link[1]
                if np.isnan(data[cam, framenum, start, 0]) or np.isnan(data[cam, framenum, end, 0]):
                    continue
                posn_start = (int(data[cam, framenum, start, 0]), int(data[cam, framenum, start, 1]))
                posn_end = (int(data[cam, framenum, end, 0]), int(data[cam, framenum, end, 1]))
                cv.line(frame, posn_start, posn_end, hex2bgr(colors[number]), 2)

            for landmark in range(21):
                if np.isnan(data[cam, framenum, landmark, 0]):
                    continue
                posn = (int(data[cam, framenum, landmark, 0]), int(data[cam, framenum, landmark, 1]))
                cv.circle(frame, posn, 3, (0, 0, 0), thickness=1)

            # Resize the frame
            resized_frame = cv.resize(frame, (display_width, display_height))

            # Display and save images
            # cv.imshow(f'cam{cam}', resized_frame)
            cv.imwrite(outdir_images_refined + trialname + '/cam' + str(cam) + '/' + 'frame' + f'{framenum:04d}' + '.png', resized_frame)

        k = cv.waitKey(10)
        if k & 0xFF == 27:  # ESC key
            break

        # Increment frame number
        framenum += 1

    # Clear windows
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


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
    id = os.path.split(os.path.split(idfolder)[0])[1]
    visit = os.path.basename(os.path.normpath(idfolder))
    print(id + '; ' + visit)

    # Identify trials
    trials = sorted(glob.glob(idfolder + '/landmarks/*'))

    # Gather camera calibration parameters
    calfiles = glob.glob(idfolder + '/calibration/*.yaml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)
    ncams = len(calfiles)

    # Output directories
    outdir_images_refined = idfolder + '/imagesrefined/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data2d = idfolder + '/landmarks/'

    # Make output directories
    os.makedirs(outdir_images_refined, exist_ok=True)
    os.makedirs(outdir_video, exist_ok=True)

    for trial in tqdm(trials):

        # Identify trial name
        trialname = os.path.basename(trial)
        print(trialname)

        # Load keypoint data
        data_2d_right = np.load(glob.glob(trial + '/*2Dlandmarks_right.npy')[0]).astype(float)
        data_2d_left = np.load(glob.glob(trial + '/*2Dlandmarks_left.npy')[0]).astype(float)
        data_2d_body = np.load(glob.glob(trial + '/*2Dlandmarks_body.npy')[0]).astype(float)

        # Isolate keypoint data
        data_2d_combined = np.concatenate((data_2d_body[:, :, :, :2], data_2d_right[:, :, :, :2], data_2d_left[:, :, :, :2]), axis=2)

        # Video parameters
        nframes = data_2d_combined.shape[1]
        nlandmarks = data_2d_combined.shape[2]

        # Output directories for the specific trial (for visualizations)
        os.makedirs(outdir_images_refined + trialname, exist_ok=True)
        os.makedirs(outdir_video + trialname, exist_ok=True)
        for cam in range(ncams):
            os.makedirs(outdir_images_refined + trialname + '/cam' + str(cam), exist_ok=True)

        # Save refined labels
        data_2d_refined = switch_hands(data_2d_combined)
        np.save(outdir_data2d + trialname + '/' + trialname + '_2Dlandmarksrefined', data_2d_refined)

        # Output visualizations
        # Refined 2D labels
        vidnames = sorted(glob.glob(idfolder + '/videos/' + trialname + '/*.avi'))
        visualizelabels(vidnames, data=data_2d_refined)
        for cam in range(ncams):
            imagefolder = outdir_images_refined + trialname + '/cam' + str(cam)
            print('Saving video.')
            createvideo(image_folder=imagefolder, extension='.png', fs=60,
                        output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '_refined.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
