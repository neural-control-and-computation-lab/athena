# Libraries
import glob
from labels2d import createvideo
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from labels2d import readcalibration


def triangulate_simple(points, camera_mats):
    """
    Triangulates undistorted 2D landmark locations from each camera to a set of 3D points in global space.

    Code from here: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py

    :param points: 2D camear landmark locations.
    :param camera_mats: Camera extrinsic matrices.
    :return: 3D points.
    """

    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d


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


def visualize_3d(p3ds, save_path=None):
    """
    Visualized 3D points in 3D space and saves images if filename given.

    Code adapted from here: https://github.com/TemugeB/bodypose3d/blob/main/show_3d_pose.py

    :param p3ds: 3D points
    :param save_path: Filename of saved images.
    """

    colours = ['#FDE7EF', '#FDE7EF', '#FDE7EF', '#FDE7EF',
               '#F589B1', '#F589B1', '#F589B1', '#F589B1',
               '#ED2B72', '#ED2B72', '#ED2B72', '#ED2B72',
               '#A50E45', '#A50E45', '#A50E45', '#A50E45',
               '#47061D', '#47061D', '#47061D', '#47061D',
               '#E5F6FF', '#E5F6FF', '#E5F6FF', '#E5F6FF',
               '#80D1FF', '#80D1FF', '#80D1FF', '#80D1FF',
               '#1AACFF', '#1AACFF', '#1AACFF', '#1AACFF',
               '#0072B3', '#0072B3', '#0072B3', '#0072B3',
               '#00314D', '#00314D', '#00314D', '#00314D']
    links = [[33, 34], [34, 35], [35, 36], [36, 37],  # right thumb
             [33, 38], [38, 39], [39, 40], [40, 41],
             [33, 42], [42, 43], [43, 44], [44, 45],
             [33, 46], [46, 47], [47, 48], [48, 49],
             [33, 50], [50, 51], [51, 52], [52, 53],
             [54, 55], [55, 56], [56, 57], [57, 58],  # left thumb
             [54, 59], [59, 60], [60, 61], [61, 62],
             [54, 63], [63, 64], [64, 65], [65, 66],
             [54, 67], [67, 68], [68, 69], [69, 70],
             [54, 71], [71, 72], [72, 73], [73, 74]]
    nframes = np.shape(p3ds)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum in range(nframes):

        # Skip frames
        if framenum % 2 == 0:
            continue

        for linknum, link in enumerate(links):
            ax.plot(xs=[p3ds[framenum, link[0], 0], p3ds[framenum, link[1], 0]],
                    ys=[p3ds[framenum, link[0], 1], p3ds[framenum, link[1], 1]],
                    zs=[p3ds[framenum, link[0], 2], p3ds[framenum, link[1], 2]],
                    linewidth=5, c=colours[linknum], alpha=0.7)

        for i in range(33, 75):
            ax.scatter(xs=p3ds[framenum, i:i + 1, 0], ys=p3ds[framenum, i:i + 1, 1], zs=p3ds[framenum, i:i + 1, 2],
                       marker='o', s=40, lw=2, c='white', edgecolors='black', alpha=0.7)

        # Axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Axis limits and labels
        ax.set_xlim3d([-1300, -300])
        ax.set_ylim3d([-1500, -500])
        ax.set_zlim3d([-2400, -1400])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(-67, -42)
        ax.set_axis_off()

        if save_path is not None:
            plt.savefig(save_path.format(framenum), dpi=100)
        else:
            plt.pause(0.1)
        ax.cla()

    if save_path is None:
        plt.show()

    plt.close(fig)


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

    # Gather camera calibration parameters
    calfiles = glob.glob(idfolder + '/calibration/*.yaml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)
    ncams = len(calfiles)

    # Gather 2D data
    trialdata = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks.npy'))

    # Output directories
    outdir_images_refined = idfolder + '/imagesrefined/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data3d = idfolder + '/landmarks/'

    for trial in tqdm(trialdata):

        # Identify trial name
        filename = os.path.basename(trial)
        fileparts = filename.split('_2Dlandmarks.npy')
        trialname = fileparts[0]
        print(trialname)

        # Load 2D hand location data and reshape
        data_2d = np.load(trial).astype(float)
        nlandmarks = np.shape(data_2d)[2]
        data_2d = data_2d.reshape((ncams, -1, 2))

        # Check # of cameras
        if ncams != data_2d.shape[0]:
            print('Number of cameras in calibration parameters does not match 2D data.')
            quit()

        # Outputting 3D points
        # Code adapted from aniposelib: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py
        npoints = data_2d.shape[1]  # nframes x nlandmarks
        data3d = np.empty((npoints, 3))
        data3d[:] = np.nan
        print('Triangulating.')
        for point in range(npoints):

            subp = data_2d[:, point, :]

            # Check how many cameras picked up the landmark for the given frame
            good = ~np.isnan(subp[:, 0])

            # Require at least 2 cameras to have picked up a landmark to triangulate, otherwise keep as nan
            if np.sum(good) >= 2:
                data3d[point] = triangulate_simple(subp[good], np.array(cam_mats_extrinsic)[good])

        # Reshaping to nframes x nlandmarks x 3-dimension
        data3d = data3d.reshape((int(len(data3d)/nlandmarks), nlandmarks, 3))

        # Save 3D landmarks as np array
        np.save(outdir_data3d + trialname + '_3Dlandmarks', data3d)

        # Output directories for the specific trial (for visualizations)
        outdir_images_trialfolder = outdir_images_refined + str(trialname) + '/data3d/'
        if not os.path.exists(outdir_images_trialfolder):
            os.mkdir(outdir_images_trialfolder)
        outdir_video_trialfolder = outdir_video + str(trialname)
        if not os.path.exists(outdir_video_trialfolder):
            os.mkdir(outdir_video_trialfolder)

        # Output 3D visualizations
        print('Saving video.')
        visualize_3d(data3d, save_path=outdir_images_trialfolder + 'frame_{:04d}.png')
        createvideo(image_folder=outdir_images_trialfolder, extension='.png', fs=30,
                    output_folder=outdir_video_trialfolder, video_name='data3d.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')