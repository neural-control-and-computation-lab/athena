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


def visualizelabels(input_streams, data):
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

            # Display and save images
            # cv.imshow(f'cam{cam}', frame)
            cv.imwrite(outdir_images_refined + trialname + '/cam' + str(cam) + '/' + 'frame' + f'{framenum:04d}' + '.png', frame)

        k = cv.waitKey(10)
        if k & 0xFF == 27:  # ESC key
            break

        # Increment frame number
        print(framenum)
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

    # Gather 2D hand locations from all trials
    trialdata_right = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_right.npy'))
    trialdata_left = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_left.npy'))
    trialdata_body = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_body.npy'))

    # Output directories
    outdir_images = idfolder + '/images/'
    outdir_images_refined = idfolder + '/imagesrefined/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data2d = idfolder + '/landmarks/'

    # Make output directories if they do not exist (landmarks folder should already exist)
    if not os.path.exists(outdir_images):
        os.mkdir(outdir_images)
    if not os.path.exists(outdir_images_refined):
        os.mkdir(outdir_images_refined)
    if not os.path.exists(outdir_video):
        os.mkdir(outdir_video)

    for trialright, trialleft, trialbody in tqdm(zip(trialdata_right, trialdata_left, trialdata_body)):

        # Identify trial name
        filename = os.path.basename(trialright)
        fileparts = filename.split('_2Dlandmarks_right.npy')
        trialname = fileparts[0]
        print(trialname)

        # Load 2D hand location data
        data_2d_right = np.load(trialright).astype(float)
        data_2d_left = np.load(trialleft).astype(float)
        data_2d_body = np.load(trialbody).astype(float)
        data_2d_combined = np.concatenate((data_2d_body, data_2d_right, data_2d_left), axis=2)

        # Number of cameras
        ncams = np.shape(data_2d_right)[0]

        # Output directories for the specific trial (for visualizations)
        outdir_images_trialfolder = outdir_images_refined + str(trialname) + '/data3d/'
        if not os.path.exists(outdir_images + str(trialname)):
            os.mkdir(outdir_images + str(trialname))
            for cam in range(ncams):
                os.mkdir(outdir_images + trialname + '/cam' + str(cam))
        if not os.path.exists(outdir_images_refined + str(trialname)):
            os.mkdir(outdir_images_refined + str(trialname))
            for cam in range(ncams):
                os.mkdir(outdir_images_refined + trialname + '/cam' + str(cam))
        if not os.path.exists(outdir_images_trialfolder):
            os.mkdir(outdir_images_trialfolder)
        outdir_video_trialfolder = outdir_video + str(trialname)
        if not os.path.exists(outdir_video_trialfolder):
            os.mkdir(outdir_video_trialfolder)

        # Copy data to avoid overwriting issues
        test = data_2d_combined.copy()

        # Fix hand switching (temp - need to vectorize and make into function)
        for cam in range(ncams):
            rwrist = data_2d_combined[cam, :, 16, :]
            lwrist = data_2d_combined[cam, :, 15, :]
            rhand = data_2d_combined[cam, :, 33, :]
            lhand = data_2d_combined[cam, :, 54, :]

            norm_rvsr = np.linalg.norm(rwrist - rhand, axis=-1)
            norm_rvsl = np.linalg.norm(rwrist - lhand, axis=-1)
            norm_lvsr = np.linalg.norm(lwrist - rhand, axis=-1)
            norm_lvsl = np.linalg.norm(lwrist - lhand, axis=-1)

            # Missing frames
            rhand_missing = np.where(rhand[:, 0] == -1, 0, 1)
            lhand_missing = np.where(lhand[:, 0] == -1, 0, 1)

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
                    temp = np.copy(test[cam, i, 33:54, :])
                    test[cam, i, 33:54, :] = test[cam, i, 54:75, :]
                    test[cam, i, 54:75, :] = temp

        # Save refined labels
        np.save(outdir_data2d + trialname + '_2Dlandmarks', test)

        # Output visualizations
        # Refined 2D labels (note, these are from the unsynced data, so cam frames may be 1-3 frames off)
        vidnames = sorted(glob.glob(idfolder + '/videos/' + trialname + '/*.avi'))
        visualizelabels(vidnames, data=test)
        for cam in range(ncams):
            imagefolder = outdir_images_refined + trialname + '/cam' + str(cam)
            print('Saving video.')
            createvideo(image_folder=imagefolder, extension='.png', fs=60,
                        output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '_refined.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
