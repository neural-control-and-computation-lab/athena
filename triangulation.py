# Libraries
import cv2 as cv
import glob
from itertools import combinations
from labels2d import createvideo
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def readcalibration(calfilepathway):
    """
    Outputs camera calibration parameters separate .yaml file.

    :param calibrationfiles: Pathway containing all yaml files.
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
        cam_rotn = cam_yaml.getNode("R").mat()
        cam_transln = cam_yaml.getNode("T").mat()
        cam_transform = transformationmatrix(cam_rotn, cam_transln)

        # Store calibration parameters
        extrinsics.append(cam_transform)
        intrinsics.append(cam_int)
        dist_coeffs.append(cam_dist.reshape(-1))

    return extrinsics, intrinsics, dist_coeffs


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


def undistort_points(points, matrix, dist):
    """
    Undistorts 2D pixel points based on camera intrinsics and distortion coefficients.

    Code from here: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py

    :param points: 2D pixel points of landmark locations.
    :param matrix: Intrinsic camera parameters.
    :param dist: Distortion coefficients.
    :return: Undistorted 2D points of landmark locations.
    """

    points = points.reshape(-1, 1, 2)
    out = cv.undistortPoints(points, matrix, dist)
    return out


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

    # Creating links for each digit
    thumb = [[0, 1], [1, 2], [2, 3], [3, 4]]
    index = [[0, 5], [5, 6], [6, 7], [7, 8]]
    middle = [[0, 9], [9, 10], [10, 11], [11, 12]]
    ring = [[0, 13], [13, 14], [14, 15], [15, 16]]
    little = [[0, 17], [17, 18], [18, 19], [19, 20]]
    body = [thumb, index, middle, ring, little]
    colors = ['#AAAAAA', '#EE3377', '#EE7733', '#009988', '#0077BB']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine axis ranges (ignoring first and last second)
    axis_min = np.min(p3ds[30:-30], axis=(0, 1))
    axis_max = np.max(p3ds[30:-30], axis=(0, 1))
    axisrange = axis_max - axis_min
    max_axisrange = max(axisrange)
    max_axisrange = (math.ceil(max_axisrange / 100.00) * 100)

    for framenum, kpts3d in enumerate(p3ds):

        # Skip frames
        # if framenum % 3 == 0:
        #     continue

        # Drawing links
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]], ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                        zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]], linewidth=5, c=part_color, alpha=0.7)

        # Drawing joints
        for i in range(21):
            ax.scatter(xs=kpts3d[i:i + 1, 0], ys=kpts3d[i:i + 1, 1], zs=kpts3d[i:i + 1, 2],
                       marker='o', s=40, lw=2, c='white', edgecolors='black', alpha=0.7)

        # Axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Axis limits
        ax.set_xlim3d([axis_min[0], axis_min[0]+max_axisrange])
        ax.set_xlabel('X')
        ax.set_ylim3d([axis_min[1], axis_min[1]+max_axisrange])
        ax.set_ylabel('Y')
        ax.set_zlim3d([axis_min[2], axis_min[2]+max_axisrange])
        ax.set_zlabel('Z')
        ax.view_init(-71, -73)

        # Remove background
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

    # Gather all combination of cameras
    camlist = np.arange(ncams)
    cam_combos = []
    for cam in range(2, ncams+1):
        combos = list(combinations(camlist, cam))
        cam_combos.extend(combos)
    ncombos = len(cam_combos)

    # Gather 2D data
    trialdata = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_body.npy'))  ### NEED TO FIX INPUT HERE ###

    # Output directories
    outdir_images = idfolder + '/images/'
    outdir_images_refined = idfolder + '/imagesrefined/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data2d = idfolder + '/landmarks/'
    outdir_data3d = idfolder + '/landmarks/'

    for trial in tqdm(trialdata):

        # Identify trial name
        filename = os.path.basename(trial)
        fileparts = filename.split('_2Dlandmarks_body.npy')  ### NEED TO FIX INPUT HERE ###
        trialname = fileparts[0]
        print(trialname)

        # Load 2D hand location data
        data_2d = np.load(trial).astype(float)

        # Check # of cameras
        if ncams != data_2d.shape[0]:
            print('Number of cameras in calibration parameters does not match 2D data.')
            quit()

        # Undistort 2D points based on camera intrinsics and distortion coefficients
        # Output is ncams x (nframes x 21 landmarks) x 2-dimension
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                      cam_dist_coeffs[cam]).reshape(len(data_2d[cam]), 2)

        # Outputting 3D points
        # Code adapted from aniposelib: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py
        npoints = data_2d_undistort.shape[1]  # nframes x 21
        for combo in cam_combos:
            data3d = np.empty((npoints, 3))
            data3d[:] = np.nan
            cam_mats_extrinsic_sub = [cam_mats_extrinsic[cam] for cam in combo]

            for point in range(npoints):

                # Selecting only from the specific camera combinations for the given frame/landmark point
                subp = data_2d_undistort[combo, point, :]

                # Check how many cameras picked up the landmark for the given frame
                good = ~np.isnan(subp[:, 0])

                # Require at least 2 cameras to have picked up a landmark to triangulate, otherwise keep as nan
                if np.sum(good) >= 2:
                    data3d[point] = triangulate_simple(subp[good], np.array(cam_mats_extrinsic_sub)[good])

            # Reshaping to nframes x 21 landmarks x 3-dimension
            data3d = data3d.reshape((int(len(data3d) / 21), 21, 3))

            # Save 3D landmarks as np array
            if len(combo) == ncams:  # Using all cameras
                data3d_use = data3d.copy()

            camcombo_str = ''.join(map(str, combo))
            outdir_data3d_subset = outdir_data3d + '3d/' + camcombo_str + '/'
            if not os.path.exists(outdir_data3d_subset):
                os.mkdir(outdir_data3d_subset)
            np.save(outdir_data3d_subset + trialname + '_3Dlandmarks', data3d)

        # Missing data
        missing = np.count_nonzero(np.isnan(data3d_use))/3
        print('Frames missing: ' + str(missing))

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

        # Output visualizations
        # 3D datapoints
        visualize_3d(data3d_use, save_path=outdir_images_trialfolder + 'frame_{:04d}.png')
        createvideo(image_folder=outdir_images_trialfolder, extension='.png', fs=30,
                    output_folder=outdir_video_trialfolder, video_name='data3d.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')