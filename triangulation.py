# Libraries
import cv2 as cv
import glob
import json
from labels2d import createvideo, readcalibration
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm


def undistort_points(points, matrix, dist):
    points = points.reshape(-1, 1, 2)
    out = cv.undistortPoints(points, matrix, dist)
    return out


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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plt.ion()

    for framenum in range(nframes):

        # Skip frames
        # if framenum % 2 == 0:
        #     continue

        for linknum, link in enumerate(links):
            ax.plot(xs=[p3ds[framenum, link[0], 0], p3ds[framenum, link[1], 0]],
                    ys=[p3ds[framenum, link[0], 1], p3ds[framenum, link[1], 1]],
                    zs=[p3ds[framenum, link[0], 2], p3ds[framenum, link[1], 2]],
                    linewidth=5, c=colours[linknum], alpha=0.7)

        for i in range(33, 75):
            ax.scatter(xs=p3ds[framenum, i:i + 1, 0], ys=p3ds[framenum, i:i + 1, 1], zs=p3ds[framenum, i:i + 1, 2],
                       marker='o', s=10, lw=1, c='white', edgecolors='black', alpha=0.7)

        # Axis ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        # Axis limits and labels
        ax.set_xlim3d([-400, 400])
        ax.set_ylim3d([-400, 400])
        ax.set_zlim3d([600, 1400])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(-60, -50)

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

    # Convert gui options back to dictionary
    gui_options_json = sys.argv[1]
    gui_options = json.loads(gui_options_json)

    # Open a dialog box to select participant's folder
    idfolders = gui_options['idfolders']
    main_folder = gui_options['main_folder']

    # Gather camera calibration parameters
    calfiles = sorted(glob.glob(main_folder + '/calibration/*.yaml'))
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)
    ncams = len(calfiles)

    # Identify trials
    trials = []
    for i in range(len(idfolders)):
        trials.append(main_folder + '/landmarks/' + os.path.basename(idfolders[i]))
    trials = sorted(trials)

    # Output directories
    outdir_images_refined = main_folder + '/imagesrefined/'
    outdir_video = main_folder + '/videos_processed/'
    outdir_data3d = main_folder + '/landmarks/'

    # Make output directories
    os.makedirs(outdir_images_refined, exist_ok=True)
    os.makedirs(outdir_video, exist_ok=True)

    for trial in tqdm(trials):
        # Identify trial name
        trialname = os.path.basename(trial)
        print(f"Processing trial: {trialname}")

        # Load 2D hand location data and reshape
        data_2d = np.load(glob.glob(trial + '/*2Dlandmarksrefined.npy')[0]).astype(float)
        nframes = np.shape(data_2d)[1]
        nlandmarks = np.shape(data_2d)[2]
        data_2d = data_2d.reshape((ncams, -1, 2))

        # Check # of cameras
        if ncams != data_2d.shape[0]:
            print('Number of cameras in calibration parameters does not match 2D data.')
            quit()

        # Replace missing data with nans
        nancondition = (data_2d[:, :, 0] == -1) & (data_2d[:, :, 1] == -1)  # Replacing missing data with nans
        data_2d[nancondition, :] = np.nan

        # Undistort 2D points based on camera intrinsics and distortion coefficients
        # Output is ncams x (nframes x nlandmarks) x 2-dimension
        # As we already undistort image at beginning, I am using dist coefficients of [0, 0, 0, 0, 0]
        # The undistort_points will help normalize the coordinates to the camera intrinsics
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                      np.array([0, 0, 0, 0, 0])).reshape(len(data_2d[cam]), 2)

        # Outputting 3D points
        # Code adapted from aniposelib: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py
        npoints = data_2d_undistort.shape[1]  # nframes x nlandmarks
        data3d = np.empty((npoints, 3))
        data3d[:] = np.nan
        for point in range(npoints):

            subp = data_2d_undistort[:, point, :]

            # Check how many cameras picked up the landmark for the given frame
            good = ~np.isnan(subp[:, 0])

            # Require at least 2 cameras to have picked up a landmark to triangulate, otherwise keep as nan
            if np.sum(good) >= 2:
                data3d[point] = triangulate_simple(subp[good], cam_mats_extrinsic[good])

        # Reshaping to nframes x nlandmarks x 3-dimension
        data3d = data3d.reshape((int(len(data3d)/nlandmarks), nlandmarks, 3))

        # Save 3D landmarks as np array
        np.save(outdir_data3d + trialname + '/' + trialname + '_3Dlandmarks', data3d)

        # Output directories for the specific trial (for visualizations)
        outdir_images_trialfolder = outdir_images_refined + str(trialname) + '/data3d/'
        outdir_video_trialfolder = outdir_video + str(trialname)
        os.makedirs(outdir_images_trialfolder, exist_ok=True)
        os.makedirs(outdir_video_trialfolder, exist_ok=True)

        # Output 3D visualizations
        if gui_options['save_images_triangulation']:
            print('Saving images.')
            visualize_3d(data3d, save_path=outdir_images_trialfolder + 'frame_{:04d}.png')
        if gui_options['save_video_triangulation']:
            print('Saving video.')
            createvideo(image_folder=outdir_images_trialfolder, extension='.png', fs=100,
                        output_folder=outdir_video_trialfolder, video_name='data3d.mp4')
