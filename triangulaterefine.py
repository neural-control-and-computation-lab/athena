# Libraries
import cv2 as cv
import sys
import glob
import json
from labels2d import createvideo, readcalibration
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import av
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


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


def project_3d_to_2d(X_world, intrinsic_matrix, extrinsic_matrix):
    # Transform 3D point to camera coordinates
    X_camera = np.dot(extrinsic_matrix, X_world)

    # Project onto the image plane using the intrinsic matrix
    X_image_homogeneous = np.dot(intrinsic_matrix, X_camera[:3])  # Skip the homogeneous coordinate (4th value)

    # Normalize the homogeneous coordinates to get 2D point
    u = X_image_homogeneous[0] / X_image_homogeneous[2]
    v = X_image_homogeneous[1] / X_image_homogeneous[2]

    return np.array([u, v])


def calculate_bone_lengths(data3d, links):
    """
    Calculate median bone lengths for each link.
    :param data3d: 3D data array [frames, landmarks, coordinates].
    :param links: List of links defined as pairs of landmark indices.
    :return: Dictionary of median lengths for each link.
    """
    bone_lengths = {}
    for link in links:
        distances = np.linalg.norm(data3d[:, link[0], :] - data3d[:, link[1], :], axis=-1)
        bone_lengths[tuple(link)] = np.nanmedian(distances)
    return bone_lengths


def smooth3d(data3d, sigma=1, iterations=5, threshold_factor=0.1):
    """
    Applies iterative smoothing and bone-length constraint enforcement.

    :param data3d: 3D data array [frames, landmarks, coordinates].
    :param sigma: Standard deviation for Gaussian kernel.
    :param iterations: Number of smoothing and constraint enforcement iterations.
    :param threshold_factor: Allowed deviation from the median bone length.
    :return: Smoothed and adjusted 3D data.
    """
    # Define the bone links
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

    # Calculate initial bone lengths for reference
    bone_lengths = calculate_bone_lengths(data3d, links)

    data3d_smoothed = data3d.copy()

    for it in range(iterations):
        # Apply NaN-aware Gaussian smoothing over time
        for coord in range(3):
            data = data3d_smoothed[:, :, coord]
            data3d_smoothed[:, :, coord] = nan_gaussian_filter1d(
                data, sigma=sigma, axis=0)

        # Enforce bone-length constraints
        for frame in range(data3d_smoothed.shape[0]):
            for link in links:
                p1_idx, p2_idx = link
                point1 = data3d_smoothed[frame, p1_idx]
                point2 = data3d_smoothed[frame, p2_idx]
                current_length = np.linalg.norm(point2 - point1)
                target_length = bone_lengths[tuple(link)]

                # Skip if invalid
                if current_length == 0 or np.isnan(current_length) or target_length == 0:
                    continue

                # Scale if deviation is beyond threshold
                deviation = current_length - target_length
                if abs(deviation) > threshold_factor * target_length:
                    scaling_factor = target_length / current_length
                    midpoint = (point1 + point2) / 2

                    if not np.isnan(scaling_factor) and np.isfinite(scaling_factor):
                        # Scale both points relative to the midpoint
                        data3d_smoothed[frame, p1_idx] = midpoint + (point1 - midpoint) * scaling_factor
                        data3d_smoothed[frame, p2_idx] = midpoint + (point2 - midpoint) * scaling_factor

    return data3d_smoothed

def nan_gaussian_filter1d(arr, sigma, axis=0):
    """
    Apply Gaussian filter to an array with NaN values along a specified axis.

    :param arr: Input array with NaNs.
    :param sigma: Standard deviation for Gaussian kernel.
    :param axis: Axis along which to apply the filter.
    :return: Smoothed array.
    """
    # Create an array of weights where data is not NaN
    weights = (~np.isnan(arr)).astype(float)
    # Replace NaNs with zero for convolution
    arr_filled = np.nan_to_num(arr)
    # Apply Gaussian filter to the data and weights
    filtered_data = gaussian_filter1d(arr_filled * weights, sigma=sigma, axis=axis, mode='nearest')
    filtered_weights = gaussian_filter1d(weights, sigma=sigma, axis=axis, mode='nearest')
    # Avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed_arr = filtered_data / filtered_weights
    smoothed_arr[filtered_weights == 0] = np.nan
    return smoothed_arr


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


def process_camera(cam, input_stream, data, display_width, display_height, outdir_images_refined, trialname):
    """
    Process a single camera stream for all frames, drawing 2D hand landmarks and saving images.
    """
    try:
        # Open video stream with PyAV
        container = av.open(input_stream)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'  # Set threading for PyAV

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

        links = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
                 [9, 10], [11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15],
                 [12, 14], [14, 16], [33, 34], [34, 35], [35, 36], [36, 37], [33, 38],
                 [38, 39], [39, 40], [40, 41], [33, 42], [42, 43], [43, 44], [44, 45],
                 [33, 46], [46, 47], [47, 48], [48, 49], [33, 50], [50, 51], [51, 52],
                 [52, 53], [54, 55], [55, 56], [56, 57], [57, 58], [54, 59], [59, 60],
                 [60, 61], [61, 62], [54, 63], [63, 64], [64, 65], [65, 66], [54, 67],
                 [67, 68], [68, 69], [69, 70], [54, 71], [71, 72], [72, 73], [73, 74]]

        # Loop through all frames in the video
        for framenum, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                # Convert frame to numpy array and BGR for OpenCV
                img = frame.to_ndarray(format="bgr24")

                # Draw landmarks if available for the current frame
                if framenum < data.shape[1] and not np.isnan(data[cam, framenum, :, 0]).all():
                    for number, link in enumerate(links):
                        start, end = link
                        if not np.isnan(data[cam, framenum, [start, end], 0]).any():
                            posn_start = tuple(data[cam, framenum, start, :2].astype(int))
                            posn_end = tuple(data[cam, framenum, end, :2].astype(int))
                            cv.line(img, posn_start, posn_end, hex2bgr(colors[number]), 2)

                    for landmark in range(21):
                        if not np.isnan(data[cam, framenum, landmark, 0]):
                            posn = tuple(data[cam, framenum, landmark, :2].astype(int))
                            cv.circle(img, posn, 3, (0, 0, 0), thickness=1)

                # Resize and save the processed frame
                resized_frame = cv.resize(img, (display_width, display_height))
                cv.imwrite(f"{outdir_images_refined}{trialname}/cam{cam}/frame{framenum:04d}.png", resized_frame)

    except Exception as e:
        print(f"Error processing camera {cam}, frame {framenum}: {e}")
    finally:
        # Ensure the container is closed after processing
        container.close()

def visualizelabels(input_streams, data, display_width=450, display_height=360, outdir_images_refined='', trialname=''):
    """
    Draws 2D hand landmarks on videos.
    :param input_streams: List of videos
    :param data: 2D hand landmarks.
    """
    # Limit the max_workers to avoid too many open files
    max_workers = min(len(input_streams), 8)  # Adjust based on system capacity

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cam in range(len(input_streams)):
            futures.append(executor.submit(
                process_camera,
                cam, input_streams[cam], data, display_width, display_height, outdir_images_refined, trialname
            ))

        # Wait for all threads to complete
        for future in futures:
            future.result()


def visualize_3d(p3ds, save_path=None):
    """
    Visualize 3D points in 3D space and saves images if filename given.
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
    ax.set_xlim3d([-400, 400])
    ax.set_ylim3d([-400, 400])
    ax.set_zlim3d([600, 1400])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(-60, -50)

    # Create line and scatter objects outside the loop
    lines = [ax.plot([], [], [], linewidth=5, color=colours[i], alpha=0.7)[0] for i in range(len(links))]
    scatter = ax.scatter([], [], [], marker='o', s=10, lw=1, c='white', edgecolors='black', alpha=0.7)

    for framenum in range(len(p3ds)):
        # Update lines for each link
        for linknum, (link, line) in enumerate(zip(links, lines)):
            line.set_data([p3ds[framenum, link[0], 0], p3ds[framenum, link[1], 0]],
                          [p3ds[framenum, link[0], 1], p3ds[framenum, link[1], 1]])
            line.set_3d_properties([p3ds[framenum, link[0], 2], p3ds[framenum, link[1], 2]])

        # Update scatter data for all points at once
        scatter._offsets3d = (p3ds[framenum, 33:75, 0],
                              p3ds[framenum, 33:75, 1],
                              p3ds[framenum, 33:75, 2])

        if save_path is not None:
            plt.savefig(save_path.format(framenum), dpi=100)
        else:
            plt.pause(0.01)

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
    calfiles = sorted(glob.glob(main_folder + '/calibration/*.yaml'))
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
        data_2d = switch_hands(data_2d_combined).reshape((ncams, -1, 2))

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
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))

        # Smooth
        data3d = smooth3d(data3d)

        # Re-flatten
        data3d = data3d.reshape(-1, 3)

        # Project back to 2D
        data3d_homogeneous = np.hstack([data3d, np.ones((data3d.shape[0], 1))])
        data_2d_new = np.zeros((ncams, data3d.shape[0], 2))
        for cam in range(ncams):
            data_2d_new[cam, :, :] = project_3d_to_2d(data3d_homogeneous.transpose(), cam_mats_intrinsic[cam], cam_mats_extrinsic[cam]).transpose()
        data_2d_new = data_2d_new.reshape((ncams, int(len(data3d) / nlandmarks), nlandmarks, 2))

        np.save(outdir_data2d + trialname + '/' + trialname + '_2Dlandmarksrefined', data_2d_new)

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
        if gui_options['save_images_triangulation']:
            print('Saving refined images.')
            visualizelabels(vidnames, outdir_images_refined=outdir_images_refined, trialname=trialname, data=data_2d_new)

        if gui_options['save_video_triangulation']:
            print('Saving refined videos.')
            for cam in range(ncams):
                imagefolder = outdir_images_refined + trialname + '/cam' + str(cam)
                createvideo(image_folder=imagefolder, extension='.png', fs=100,
                            output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '_refined.mp4')
