# Libraries
import cv2 as cv
import glob
import json
from labels2d import createvideo, readcalibration
import numpy as np
import os
from scipy.interpolate import splev, splrep
from scipy import signal
import sys
from tqdm import tqdm
from triangulation import triangulate_simple, undistort_points
import matplotlib.pyplot as plt



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


def nan_helper(y):
    """
    https://github.com/lambdaloop/anipose/blob/master/anipose/filter_pose.py
    :param y:
    :return:
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def project_3d_to_2d(X_world, intrinsic_matrix, extrinsic_matrix):
    # Transform 3D point to camera coordinates
    X_camera = np.dot(extrinsic_matrix, X_world)

    # Project onto the image plane using the intrinsic matrix
    X_image_homogeneous = np.dot(intrinsic_matrix, X_camera[:3])  # Skip the homogeneous coordinate (4th value)

    # Normalize the homogeneous coordinates to get 2D point
    u = X_image_homogeneous[0] / X_image_homogeneous[2]
    v = X_image_homogeneous[1] / X_image_homogeneous[2]

    return np.array([u, v])


def smooth2d(data2d, kernel, pixeldif):
    """
    Applies a median filter to the data and calculates difference between original and filtered signal.
    If difference between signals is beyond certain threshold, a spline is fit to the original data.

    Adapted from here:
    https://github.com/lambdaloop/anipose/blob/master/anipose/filter_pose.py
    :param data2d: 2D data.
    :param kernel: kernel size for median filter.
    :param pixeldif: Threshold length of pixels difference to warrant spline smooth.
    :return: Spline fit 2D data.
    """

    # Empty array for storing median filtered signal
    data_2d_mfilt = np.empty(data2d.shape)

    # Applying median filter
    for camera in range(ncams):
        for landmark in range(21):
            data_2d_mfilt[camera, :, landmark, 0] = signal.medfilt(data2d[camera, :, landmark, 0], kernel_size=kernel)
            data_2d_mfilt[camera, :, landmark, 1] = signal.medfilt(data2d[camera, :, landmark, 1], kernel_size=kernel)

    # Calculating difference between original and filtered signal
    errx = data2d[:, :, :, 0] - data_2d_mfilt[:, :, :, 0]
    erry = data2d[:, :, :, 1] - data_2d_mfilt[:, :, :, 1]
    err = np.sqrt((errx ** 2) + (erry ** 2))

    # Applying spline fit to replace extraneous data points
    data_2d_spline = np.empty(data2d.shape)

    for camera in range(ncams):
        for landmark in range(21):
            x = data2d[camera, :, landmark, 0]
            y = data2d[camera, :, landmark, 1]
            err_sub = err[camera, :, landmark]
            bad = np.zeros(err_sub.shape, dtype='bool')
            bad[err_sub >= pixeldif] = True

            # Ignore first and last few data points
            bad[:kernel] = False
            bad[-kernel:] = False

            pos = np.array([x, y]).T
            posi = np.copy(pos)
            posi[bad] = np.nan

            for i in range(posi.shape[1]):
                vals = posi[:, i]
                nans, ix = nan_helper(vals)

                # More than 1 data point missing, more than 80% data there
                if np.sum(nans) > 0 and np.mean(~nans) > 0.80:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans] = splev(ix(nans), spline)

                data_2d_spline[camera, :, landmark, i] = vals

    return data_2d_spline


def switch_hands(data2d):

    # Store switched hand data
    data_2d_switched = data2d.copy()

    # Part A: Switch hands based on right and left hands being closest to the respective side arms
    for cam in range(ncams):

        # Wrist (body pose model) and hand (hand pose model) locations
        rwrist = data2d[cam, :, 16, :]
        lwrist = data2d[cam, :, 15, :]
        rhand = data2d[cam, :, 33, :]
        lhand = data2d[cam, :, 54, :]

        # 2D distance between wrist and hands
        norm_rvsr = np.linalg.norm(rwrist - rhand, axis=-1)
        norm_rvsl = np.linalg.norm(rwrist - lhand, axis=-1)
        norm_lvsr = np.linalg.norm(lwrist - rhand, axis=-1)
        norm_lvsl = np.linalg.norm(lwrist - lhand, axis=-1)

        # Present rhand and lhand
        c1 = (rhand[:, 0] != -1) & (rhand[:, 1] != -1)
        c2 = (lhand[:, 0] != -1) & (lhand[:, 1] != -1)

        # Present rarm and larm
        c3 = (rwrist[:, 0] != -1) & (rwrist[:, 1] != -1)
        c4 = (lwrist[:, 0] != -1) & (lwrist[:, 1] != -1)

        # Hands are switched (2 hands and 2 arms present)
        c5 = norm_rvsr > norm_rvsl
        c6 = norm_lvsl > norm_lvsr
        condition1a = c1 & c2 & c3 & c4 & c5 & c6

        # Hands are switched (2 hands and left arm present)
        c7 = norm_lvsl > norm_lvsr
        condition2a = c1 & c2 & ~c3 & c4 & c7

        # Hands are switched (2 hands and right arm present)
        c8 = norm_rvsr > norm_rvsl
        condition3a = c1 & c2 & c3 & ~c4 & c8

        # Hands are switched (left hand and 2 arms present)
        c9 = norm_lvsl > norm_rvsl
        condition4a = ~c1 & c2 & c3 & c4 & c9

        # Hands are switched (right hand and 2 arms present)
        c10 = norm_rvsr > norm_lvsr
        condition5a = c1 & ~c2 & c3 & c4 & c10

        # If any of the conditions above are met, then switch hands
        combined_condition_a = condition1a | condition2a | condition3a | condition4a | condition5a

        for i, flag in enumerate(combined_condition_a):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp

    # Part B: Use estimated 2D projections to further detect hand switching
    # Estimate 3D locations of the right & left hands and project to 2D for each cam
    # Undistort and normalize 2D coordinates (distortion coefficients set to 0 as image undistorted already)
    data_2d = data_2d_switched.copy()
    nancondition = (data_2d[:, :, :, 0] == -1) & (data_2d[:, :, :, 1] == -1)  # Replacing missing data with nans
    data_2d[nancondition, :] = np.nan
    data_2d = data_2d.reshape((ncams, -1, 2))
    data_2d_undistort = np.empty(data_2d.shape)
    for cam in range(ncams):
        data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                  np.array([0, 0, 0, 0, 0])).reshape(len(data_2d[cam]), 2)
    data_2d_undistort = data_2d_undistort.reshape((ncams, nframes, nlandmarks, 2))

    # Pre-allocate storage of locations for left and right hands
    lhand = np.empty((nframes, 3))  # 3D location of left hand
    lhand[:] = np.nan
    rhand = np.empty((nframes, 3))  # 3D location of right hand
    rhand[:] = np.nan
    handestimate = np.empty((ncams, nframes, 2, 2))  # 2D projections of left and right hand
    handestimate[:] = np.nan

    # 3D triangulation and project back to 2D
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

        # Require at least 2 cameras to have picked up both hand and wrist for triangulation
        if np.sum(good_lh) >= 2:
            lhand[frame] = triangulate_simple(sub_lh[good_lh], cam_mats_extrinsic[good_lh])

        if np.sum(good_rh) >= 2:
            rhand[frame] = triangulate_simple(sub_rh[good_rh], cam_mats_extrinsic[good_rh])

        # Project back to 2D
        for cam in range(ncams):
            lhand_world = np.append(lhand[frame], 1)
            rhand_world = np.append(rhand[frame], 1)
            handestimate[cam, frame, 0, :] = project_3d_to_2d(rhand_world, cam_mats_intrinsic[cam],
                                                              cam_mats_extrinsic[cam])
            handestimate[cam, frame, 1, :] = project_3d_to_2d(lhand_world, cam_mats_intrinsic[cam],
                                                              cam_mats_extrinsic[cam])

    # Use estimated 2D projections to further detect hand switching
    nancondition = (data_2d_switched[:, :, :, 0] == -1) & (data_2d_switched[:, :, :, 1] == -1)
    data_2d_switched[nancondition, :] = -9999
    for cam in range(ncams):

        # Obtaining locations of right and left hands
        rhand = data_2d_switched[cam, :, 33, :]
        lhand = data_2d_switched[cam, :, 54, :]
        rhand_est = handestimate[cam, :, 0, :]
        lhand_est = handestimate[cam, :, 1, :]

        # Calculating differences between mediapipe predicted and 2D projected hand locations
        norm_rvsrest = np.linalg.norm(rhand - rhand_est, axis=-1)
        norm_lvsrest = np.linalg.norm(lhand - rhand_est, axis=-1)
        norm_rvslest = np.linalg.norm(rhand - lhand_est, axis=-1)
        norm_lvslest = np.linalg.norm(lhand - lhand_est, axis=-1)

        # Condition where MP predicted left hand is closer to estimated right hand
        c1 = norm_lvsrest < norm_rvsrest
        c2 = norm_lvsrest < norm_lvslest
        condition1b = c1 & c2

        # Condition where MP predicted right hand is closer to estimated left hand
        c3 = norm_rvslest < norm_lvslest
        c4 = norm_rvslest < norm_rvsrest
        condition2b = c3 & c4

        # If any of the conditions above are met, then switch hands
        combined_condition_b = condition1b | condition2b

        for i, flag in enumerate(combined_condition_b):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp

    nancondition = (data_2d_switched[:, :, :, 0] == -9999) & (data_2d_switched[:, :, :, 1] == -9999)
    data_2d_switched[nancondition, :] = -1

    return data_2d_switched

#TODO: parallelize visualize labels
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

        # If frame number of video is more than frames of landmarks, quit
        if framenum > data.shape[1]-1:
            break

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

    # Identify trials
    trials = []
    for i in range(len(idfolders)):
        trials.append(main_folder + '/landmarks/' + os.path.basename(idfolders[i]))
    trials = sorted(trials)

    # Gather camera calibration parameters
    calfiles = glob.glob(main_folder + '/calibration/*.yaml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)
    ncams = len(calfiles)

    # Output directories
    outdir_images_refined = main_folder + '/imagesrefined/'
    outdir_video = main_folder + '/videos_processed/'
    outdir_data2d = main_folder + '/landmarks/'
    outdir_data3d = main_folder + '/landmarks/'

    # Make output directories
    os.makedirs(outdir_images_refined, exist_ok=True)
    os.makedirs(outdir_video, exist_ok=True)

    for trial in tqdm(trials):
        # Identify trial name
        trialname = os.path.basename(trial)
        print(f"Processing trial: {trialname}")

        # Load keypoint data
        data_2d_right = []
        data_2d_left = []
        data_2d_body = []
        landmarkfiles = sorted([d for d in glob.glob(trial + '/*') if os.path.isdir(d)])
        for cam in range(ncams):
            data_2d_right.append(np.load(glob.glob(landmarkfiles[cam] + '/*2Dlandmarks_right.npy')[0]).astype(float))
            data_2d_left.append(np.load(glob.glob(landmarkfiles[cam] + '/*2Dlandmarks_left.npy')[0]).astype(float))
            data_2d_body.append(np.load(glob.glob(landmarkfiles[cam] + '/*2Dlandmarks_body.npy')[0]).astype(float))
        data_2d_right = np.stack(data_2d_right)
        data_2d_left = np.stack(data_2d_left)
        data_2d_body = np.stack(data_2d_body)

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

        # Switch hands and smooth
        data_2d_refined = switch_hands(data_2d_combined)
        #data_2d_refined = smooth2d(data_2d_refined, kernel=7, pixeldif=20)
        np.save(outdir_data2d + trialname + '/' + trialname + '_2Dlandmarksrefined', data_2d_refined)

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

        # Project back to 2D
        data3d_homogeneous = np.hstack([data3d, np.ones((data3d.shape[0], 1))])
        data_2d_new = np.zeros((ncams, data3d.shape[0], 2))
        for cam in range(ncams):
            data_2d_new[cam, :, :] = project_3d_to_2d(data3d_homogeneous.transpose(), cam_mats_intrinsic[cam], cam_mats_extrinsic[cam]).transpose()
        data_2d_new = data_2d_new.reshape((ncams, int(len(data3d) / nlandmarks), nlandmarks, 2))

        # Reshaping to nframes x nlandmarks x 3-dimension
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))

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


        # Output visualizations
        vidnames = sorted(glob.glob(main_folder + '/videos/' + trialname + '/*.avi'))
        if gui_options['save_images_refine']:
            print('Saving refined images.')
            visualizelabels(vidnames, data=data_2d_refined)

        if gui_options['save_video_refine']:
            print('Saving refined videos.')
            for cam in range(ncams):
                imagefolder = outdir_images_refined + trialname + '/cam' + str(cam)
                createvideo(image_folder=imagefolder, extension='.png', fs=100,
                            output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '_refined.mp4')
