import sys
import os
import glob
import json
import cv2 as cv
import av
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import savgol_filter
from athena.labels2d import createvideo, readcalibration


def undistort_points(points, matrix, dist):
    """
    Undistorts a set of 2D points given the camera matrix and distortion coefficients.

    Parameters:
        points (np.ndarray): Input points to undistort. Shape should be (n_points, 2).
        matrix (np.ndarray): Camera intrinsic matrix.
        dist (np.ndarray): Distortion coefficients.

    Returns:
        np.ndarray: Undistorted points.
    """
    points = points.reshape(-1, 1, 2)
    out = cv.undistortPoints(points, matrix, dist)
    return out


def triangulate_simple(points, camera_mats):
    """
    Triangulates undistorted 2D landmark locations from each camera to 3D points in global space.

    Parameters:
        points (np.ndarray): Undistorted 2D landmark locations from each camera, shape (num_cams, 2).
        camera_mats (list of np.ndarray): List of camera extrinsic matrices, each of shape (3, 4).

    Returns:
        np.ndarray: Triangulated 3D point in world coordinates.
    """
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    _, _, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d


def hex2bgr(hexcode):
    """
    Converts a hexadecimal color code to a BGR tuple.

    Parameters:
        hexcode (str): Hexadecimal color code string, e.g., '#FF00FF'.

    Returns:
        tuple: BGR color tuple suitable for OpenCV functions.
    """
    h = hexcode.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    bgr = rgb[::-1]
    return bgr


def project_3d_to_2d(X_world, intrinsic_matrix, extrinsic_matrix):
    """
    Projects a 3D point in world coordinates to 2D image coordinates.

    Parameters:
        X_world (np.ndarray): 4-element array representing the 3D point in homogeneous coordinates (x, y, z, 1).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        extrinsic_matrix (np.ndarray): Camera extrinsic matrix (3x4).

    Returns:
        np.ndarray: 2-element array representing the 2D image coordinates (u, v).
    """
    # Transform 3D point to camera coordinates
    X_camera = np.dot(extrinsic_matrix, X_world)

    # Project onto the image plane using the intrinsic matrix
    X_image_homogeneous = np.dot(intrinsic_matrix, X_camera[:3])

    # Normalize the homogeneous coordinates to get 2D point
    u = X_image_homogeneous[0] / X_image_homogeneous[2]
    v = X_image_homogeneous[1] / X_image_homogeneous[2]

    return np.array([u, v])


def calculate_bone_lengths(data3d, links):
    """
    Calculate median bone lengths for each link.

    Parameters:
        data3d (np.ndarray): 3D data array [frames, landmarks, coordinates].
        links (list): List of links defined as pairs of landmark indices.

    Returns:
        dict: Dictionary of median lengths for each link.
    """
    bone_lengths = {}
    for link in links:
        distances = np.linalg.norm(data3d[:, link[0], :] - data3d[:, link[1], :], axis=-1)
        bone_lengths[tuple(link)] = np.nanmedian(distances)
    return bone_lengths


def restore_long_nan_runs(original_data, filtered_data, min_length=5):
    """
    Restores NaNs in the filtered_data where the original data had contiguous NaN runs longer than min_length frames.

    Parameters:
        original_data (np.ndarray): Original 1D data array with NaNs.
        filtered_data (np.ndarray): Filtered 1D data array.
        min_length (int): Minimum length of NaN runs to restore.

    Returns:
        np.ndarray: Filtered data with NaNs restored in appropriate places.
    """
    nan_mask = np.isnan(original_data)
    is_nan = np.concatenate(([0], nan_mask.view(np.int8), [0]))
    absdiff = np.abs(np.diff(is_nan))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    for start, end in ranges:
        run_length = end - start
        if run_length > min_length:
            filtered_data[start:end] = np.nan
    return filtered_data


def smooth3d(data3d, fps, centroid_frequency_cutoff=5, centroid_polyorder=3,
             point_frequency_cutoff=20, point_polyorder=3,
             iterations=3, threshold_factor=0.1):
    """
    Apply iterative smoothing to 3D data.

    This function applies iterative smoothing with frequency-based window lengths to hand centroids and points,
    restoring NaNs only for long missing data runs.

    Parameters:
        data3d (np.ndarray): 3D data array of shape [frames, landmarks, coordinates].
        fps (float): Frames per second of the data.
        centroid_frequency_cutoff (float): Desired cutoff frequency (Hz) for centroid smoothing.
        centroid_polyorder (int): Polynomial order for centroid smoothing.
        point_frequency_cutoff (float): Desired cutoff frequency (Hz) for point smoothing.
        point_polyorder (int): Polynomial order for point smoothing.
        iterations (int): Number of smoothing and constraint enforcement iterations.
        threshold_factor (float): Allowed deviation from the median bone length.

    Returns:
        np.ndarray: Smoothed and adjusted 3D data.
    """
    n_frames, n_landmarks, _ = data3d.shape

    # Determine window lengths based on desired cutoff frequencies
    centroid_window_length = int(fps / centroid_frequency_cutoff * 2 + 1)
    point_window_length = int(fps / point_frequency_cutoff * 2 + 1)

    # Ensure window lengths are odd and valid
    centroid_window_length = max(3, centroid_window_length | 1)
    centroid_window_length = min(centroid_window_length, n_frames | 1)
    point_window_length = max(3, point_window_length | 1)
    point_window_length = min(point_window_length, n_frames | 1)

    # Define the bone links
    links = [
        # Right hand
        [33, 34], [34, 35], [35, 36], [36, 37],
        [33, 38], [38, 39], [39, 40], [40, 41],
        [33, 42], [42, 43], [43, 44], [44, 45],
        [33, 46], [46, 47], [47, 48], [48, 49],
        [33, 50], [50, 51], [51, 52], [52, 53],
        # Left hand
        [54, 55], [55, 56], [56, 57], [57, 58],
        [54, 59], [59, 60], [60, 61], [61, 62],
        [54, 63], [63, 64], [64, 65], [65, 66],
        [54, 67], [67, 68], [68, 69], [69, 70],
        [54, 71], [71, 72], [72, 73], [73, 74],
    ]

    # Calculate initial bone lengths for reference
    bone_lengths = calculate_bone_lengths(data3d, links)
    data3d_smoothed = data3d.copy()

    for _ in range(iterations):
        # Aggressive low-pass filtering of hand centroids
        for hand_indices in [range(33, 54), range(54, 75)]:
            # Compute centroid for each frame
            centroids = np.nanmean(data3d_smoothed[:, hand_indices, :], axis=1)

            # Handle NaNs in centroids
            nan_mask = np.isnan(centroids).any(axis=1)
            valid_mask = ~nan_mask

            if np.sum(valid_mask) < centroid_window_length:
                smoothed_centroids = centroids.copy()
            else:
                smoothed_centroids = centroids.copy()
                for coord in range(3):
                    data = centroids[:, coord]
                    data_valid = data[valid_mask]
                    indices = np.arange(len(data))
                    data_interp = np.interp(indices, indices[valid_mask], data_valid)
                    data_filtered = savgol_filter(data_interp, window_length=centroid_window_length,
                                                  polyorder=centroid_polyorder)
                    data_filtered = restore_long_nan_runs(data, data_filtered, min_length=5)
                    smoothed_centroids[:, coord] = data_filtered

            # Adjust hand landmarks based on smoothed centroid
            for frame in range(n_frames):
                if np.isnan(smoothed_centroids[frame]).any():
                    continue
                original_centroid = np.nanmean(data3d_smoothed[frame, hand_indices, :], axis=0)
                if np.isnan(original_centroid).any():
                    continue
                shift = smoothed_centroids[frame] - original_centroid
                data3d_smoothed[frame, hand_indices, :] += shift

        # Less aggressive filtering of all points
        for coord in range(3):
            for landmark in range(n_landmarks):
                data = data3d_smoothed[:, landmark, coord]
                nan_mask = np.isnan(data)
                valid_mask = ~nan_mask
                if np.sum(valid_mask) < point_window_length:
                    continue
                data_valid = data[valid_mask]
                indices = np.arange(len(data))
                data_interp = np.interp(indices, indices[valid_mask], data_valid)
                data_filtered = savgol_filter(data_interp, window_length=point_window_length,
                                              polyorder=point_polyorder)
                data_filtered = restore_long_nan_runs(data, data_filtered, min_length=5)
                data3d_smoothed[:, landmark, coord] = data_filtered

        # Enforce bone-length constraints
        for frame in range(n_frames):
            for link in links:
                p1_idx, p2_idx = link
                point1 = data3d_smoothed[frame, p1_idx]
                point2 = data3d_smoothed[frame, p2_idx]
                current_length = np.linalg.norm(point2 - point1)
                target_length = bone_lengths[tuple(link)]

                if current_length == 0 or np.isnan(current_length) or target_length == 0:
                    continue

                deviation = current_length - target_length
                if abs(deviation) > threshold_factor * target_length:
                    scaling_factor = target_length / current_length
                    midpoint = (point1 + point2) / 2

                    if not np.isnan(scaling_factor) and np.isfinite(scaling_factor):
                        data3d_smoothed[frame, p1_idx] = midpoint + (point1 - midpoint) * scaling_factor
                        data3d_smoothed[frame, p2_idx] = midpoint + (point2 - midpoint) * scaling_factor

    return data3d_smoothed


def switch_hands(data2d, ncams, nframes, nlandmarks, cam_mats_intrinsic, cam_mats_extrinsic):
    """
    Detects and corrects swapped hand data in 2D landmarks.

    This function checks for situations where the left and right hand data might be swapped
    and corrects them based on proximity to wrists and projected 2D hand positions.

    Parameters:
        data2d (np.ndarray): 2D landmark data of shape (ncams, nframes, nlandmarks, 2).
        ncams (int): Number of cameras.
        nframes (int): Number of frames.
        nlandmarks (int): Number of landmarks.
        cam_mats_intrinsic (list): List of intrinsic camera matrices.
        cam_mats_extrinsic (list): List of extrinsic camera matrices.

    Returns:
        np.ndarray: Corrected 2D landmark data with hands switched where necessary.
    """
    data_2d_switched = data2d.copy()

    # Part A: Switch hands based on proximity to wrists
    for cam in range(ncams):
        # Wrist and hand locations
        rwrist = data2d[cam, :, 16, :]
        lwrist = data2d[cam, :, 15, :]
        rhand = data2d[cam, :, 33, :]
        lhand = data2d[cam, :, 54, :]

        # Calculate distances
        norm_rvsr = np.linalg.norm(rwrist - rhand, axis=-1)
        norm_rvsl = np.linalg.norm(rwrist - lhand, axis=-1)
        norm_lvsr = np.linalg.norm(lwrist - rhand, axis=-1)
        norm_lvsl = np.linalg.norm(lwrist - lhand, axis=-1)

        # Conditions
        c1 = ~np.isnan(rhand[:, 0])
        c2 = ~np.isnan(lhand[:, 0])
        c3 = ~np.isnan(rwrist[:, 0])
        c4 = ~np.isnan(lwrist[:, 0])
        c5 = norm_rvsr > norm_rvsl
        c6 = norm_lvsl > norm_lvsr
        condition1a = c1 & c2 & c3 & c4 & c5 & c6

        c7 = norm_lvsl > norm_lvsr
        condition2a = c1 & c2 & ~c3 & c4 & c7

        c8 = norm_rvsr > norm_rvsl
        condition3a = c1 & c2 & c3 & ~c4 & c8

        c9 = norm_lvsl > norm_rvsl
        condition4a = ~c1 & c2 & c3 & c4 & c9

        c10 = norm_rvsr > norm_lvsr
        condition5a = c1 & ~c2 & c3 & c4 & c10

        combined_condition_a = condition1a | condition2a | condition3a | condition4a | condition5a

        for i, flag in enumerate(combined_condition_a):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp

    # Part B: Use estimated 2D projections to further detect hand switching
    data_2d = data_2d_switched.copy()
    nancondition = (data_2d[:, :, :, 0] == -1) & (data_2d[:, :, :, 1] == -1)
    data_2d[nancondition] = np.nan
    data_2d = data_2d.reshape((ncams, -1, 2))
    data_2d_undistort = np.empty(data_2d.shape)
    for cam in range(ncams):
        data_2d_undistort[cam] = undistort_points(
            data_2d[cam].astype(float),
            cam_mats_intrinsic[cam],
            np.array([0, 0, 0, 0, 0])
        ).reshape(len(data_2d[cam]), 2)
    data_2d_undistort = data_2d_undistort.reshape((ncams, nframes, nlandmarks, 2))

    # Pre-allocate storage
    lhand_3d = np.empty((nframes, 3))
    lhand_3d[:] = np.nan
    rhand_3d = np.empty((nframes, 3))
    rhand_3d[:] = np.nan
    handestimate = np.empty((ncams, nframes, 2, 2))
    handestimate[:] = np.nan

    # 3D triangulation and project back to 2D
    for frame in range(nframes):
        sub_lh = data_2d_undistort[:, frame, 54, :]
        good_lh = ~np.isnan(sub_lh[:, 0])
        sub_rh = data_2d_undistort[:, frame, 33, :]
        good_rh = ~np.isnan(sub_rh[:, 0])

        if np.sum(good_lh) >= 2:
            lhand_3d[frame] = triangulate_simple(sub_lh[good_lh], cam_mats_extrinsic[good_lh])

        if np.sum(good_rh) >= 2:
            rhand_3d[frame] = triangulate_simple(sub_rh[good_rh], cam_mats_extrinsic[good_rh])

        # Project back to 2D
        for cam in range(ncams):
            if not np.isnan(rhand_3d[frame]).any():
                rhand_world = np.append(rhand_3d[frame], 1)
                handestimate[cam, frame, 0, :] = project_3d_to_2d(
                    rhand_world, cam_mats_intrinsic[cam], cam_mats_extrinsic[cam]
                )
            if not np.isnan(lhand_3d[frame]).any():
                lhand_world = np.append(lhand_3d[frame], 1)
                handestimate[cam, frame, 1, :] = project_3d_to_2d(
                    lhand_world, cam_mats_intrinsic[cam], cam_mats_extrinsic[cam]
                )

    # Detect hand switching
    nancondition = (data_2d_switched[:, :, :, 0] == -1) & (data_2d_switched[:, :, :, 1] == -1)
    data_2d_switched[nancondition] = -9999
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
        condition1b = c1 & c2

        c3 = norm_rvslest < norm_lvslest
        c4 = norm_rvslest < norm_rvsrest
        condition2b = c3 & c4

        combined_condition_b = condition1b | condition2b

        for i, flag in enumerate(combined_condition_b):
            if flag:
                temp = np.copy(data_2d_switched[cam, i, 33:54, :])
                data_2d_switched[cam, i, 33:54, :] = data_2d_switched[cam, i, 54:75, :]
                data_2d_switched[cam, i, 54:75, :] = temp

    nancondition = (data_2d_switched[:, :, :, 0] == -9999) & (data_2d_switched[:, :, :, 1] == -9999)
    data_2d_switched[nancondition] = -1

    return data_2d_switched


def process_camera(cam, input_stream, data, display_width, display_height, outdir_images_refined, trialname):
    """
    Process a single camera stream for all frames, drawing 2D hand landmarks and saving images.

    Parameters:
        cam (int): Camera index.
        input_stream (str): Path to the video file for the camera.
        data (np.ndarray): 2D landmarks data.
        display_width (int): Width to resize the output images.
        display_height (int): Height to resize the output images.
        outdir_images_refined (str): Output directory for refined images.
        trialname (str): Name of the trial.
    """
    try:
        container = av.open(input_stream)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        colors = [
            '#009988', '#009988', '#009988', '#009988',
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
            '#00314D', '#00314D', '#00314D', '#00314D'
        ]

        links = [
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10], [11, 12], [11, 23], [12, 24], [23, 24],
            [11, 13], [13, 15],
            [12, 14], [14, 16],
            [33, 34], [34, 35], [35, 36], [36, 37],
            [33, 38], [38, 39], [39, 40], [40, 41],
            [33, 42], [42, 43], [43, 44], [44, 45],
            [33, 46], [46, 47], [47, 48], [48, 49],
            [33, 50], [50, 51], [51, 52], [52, 53],
            [54, 55], [55, 56], [56, 57], [57, 58],
            [54, 59], [59, 60], [60, 61], [61, 62],
            [54, 63], [63, 64], [64, 65], [65, 66],
            [54, 67], [67, 68], [68, 69], [69, 70],
            [54, 71], [71, 72], [72, 73], [73, 74]
        ]

        for framenum, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")

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

                resized_frame = cv.resize(img, (display_width, display_height))
                output_path = os.path.join(outdir_images_refined, trialname, f'cam{cam}', f'frame{framenum:04d}.png')
                cv.imwrite(output_path, resized_frame)

    except Exception as e:
        print(f"Error processing camera {cam}, frame {framenum}: {e}")
    finally:
        container.close()


def visualizelabels(input_streams, data, display_width=450, display_height=360, outdir_images_refined='', trialname=''):
    """
    Draws 2D hand landmarks on videos.

    Parameters:
        input_streams (list): List of video file paths.
        data (np.ndarray): 2D hand landmarks.
        display_width (int, optional): Display width for resizing images. Defaults to 450.
        display_height (int, optional): Display height for resizing images. Defaults to 360.
        outdir_images_refined (str, optional): Output directory for refined images.
        trialname (str, optional): Name of the trial.
    """
    max_workers = min(len(input_streams), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cam in range(len(input_streams)):
            futures.append(executor.submit(
                process_camera,
                cam, input_streams[cam], data, display_width, display_height, outdir_images_refined, trialname
            ))

        for future in futures:
            future.result()


def visualize_3d(p3ds, save_path=None):
    """
    Visualize 3D points in 3D space and saves images if filename given.

    Parameters:
        p3ds (np.ndarray): 3D points, shape (n_frames, n_landmarks, 3).
        save_path (str, optional): If provided, saves the images to the specified path format.
    """
    colours = [
        '#FDE7EF', '#FDE7EF', '#FDE7EF', '#FDE7EF',
        '#F589B1', '#F589B1', '#F589B1', '#F589B1',
        '#ED2B72', '#ED2B72', '#ED2B72', '#ED2B72',
        '#A50E45', '#A50E45', '#A50E45', '#A50E45',
        '#47061D', '#47061D', '#47061D', '#47061D',
        '#E5F6FF', '#E5F6FF', '#E5F6FF', '#E5F6FF',
        '#80D1FF', '#80D1FF', '#80D1FF', '#80D1FF',
        '#1AACFF', '#1AACFF', '#1AACFF', '#1AACFF',
        '#0072B3', '#0072B3', '#0072B3', '#0072B3',
        '#00314D', '#00314D', '#00314D', '#00314D'
    ]

    links = [
        [33, 34], [34, 35], [35, 36], [36, 37],
        [33, 38], [38, 39], [39, 40], [40, 41],
        [33, 42], [42, 43], [43, 44], [44, 45],
        [33, 46], [46, 47], [47, 48], [48, 49],
        [33, 50], [50, 51], [51, 52], [52, 53],
        [54, 55], [55, 56], [56, 57], [57, 58],
        [54, 59], [59, 60], [60, 61], [61, 62],
        [54, 63], [63, 64], [64, 65], [65, 66],
        [54, 67], [67, 68], [68, 69], [69, 70],
        [54, 71], [71, 72], [72, 73], [73, 74]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-400, 400])
    ax.set_ylim3d([-400, 400])
    ax.set_zlim3d([600, 1400])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(-60, -50)

    lines = [ax.plot([], [], [], linewidth=5, color=colours[i], alpha=0.7)[0] for i in range(len(links))]
    scatter = ax.scatter([], [], [], marker='o', s=10, lw=1, c='white', edgecolors='black', alpha=0.7)

    for framenum in range(len(p3ds)):
        for linknum, (link, line) in enumerate(zip(links, lines)):
            line.set_data([p3ds[framenum, link[0], 0], p3ds[framenum, link[1], 0]],
                          [p3ds[framenum, link[0], 1], p3ds[framenum, link[1], 1]])
            line.set_3d_properties([p3ds[framenum, link[0], 2], p3ds[framenum, link[1], 2]])

        scatter._offsets3d = (p3ds[framenum, 33:75, 0],
                              p3ds[framenum, 33:75, 1],
                              p3ds[framenum, 33:75, 2])

        if save_path is not None:
            plt.savefig(save_path.format(framenum), dpi=100)
        else:
            plt.pause(0.01)

    plt.close(fig)


def main(gui_options_json):
    gui_options = json.loads(gui_options_json)

    # Set directories
    idfolders = gui_options['idfolders']
    main_folder = gui_options['main_folder']
    trials = sorted([os.path.join(main_folder, 'landmarks', os.path.basename(f)) for f in idfolders])

    # Camera calibration
    calfiles = sorted(glob.glob(os.path.join(main_folder, 'calibration', '*.yaml')))
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfiles)
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)
    ncams = len(calfiles)

    # Output directories
    outdir_images_refined = os.path.join(main_folder, 'imagesrefined')
    outdir_video = os.path.join(main_folder, 'videos_processed')
    outdir_data2d = os.path.join(main_folder, 'landmarks')
    outdir_data3d = os.path.join(main_folder, 'landmarks')
    os.makedirs(outdir_images_refined, exist_ok=True)
    os.makedirs(outdir_video, exist_ok=True)

    for trial in tqdm(trials):
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

        # Combine data
        data_2d_combined = np.concatenate(
            (data_2d_body[:, :, :, :2], data_2d_right[:, :, :, :2], data_2d_left[:, :, :, :2]), axis=2
        )

        # Video parameters
        nframes = data_2d_combined.shape[1]
        nlandmarks = data_2d_combined.shape[2]

        # Output directories for the trial
        os.makedirs(os.path.join(outdir_images_refined, trialname), exist_ok=True)
        os.makedirs(os.path.join(outdir_video, trialname), exist_ok=True)
        for cam in range(ncams):
            os.makedirs(os.path.join(outdir_images_refined, trialname, f'cam{cam}'), exist_ok=True)

        # Switch hands and smooth
        data_2d = switch_hands(
            data_2d_combined,
            ncams,
            nframes,
            nlandmarks,
            cam_mats_intrinsic,
            cam_mats_extrinsic
        ).reshape((ncams, -1, 2))

        if ncams != data_2d.shape[0]:
            print('Number of cameras in calibration parameters does not match 2D data.')
            quit()

        nancondition = (data_2d[:, :, 0] == -1) & (data_2d[:, :, 1] == -1)
        data_2d[nancondition, :] = np.nan

        # Undistort 2D points
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = undistort_points(
                data_2d[cam].astype(float),
                cam_mats_intrinsic[cam],
                np.array([0, 0, 0, 0, 0])
            ).reshape(len(data_2d[cam]), 2)

        # Triangulate to 3D
        npoints = data_2d_undistort.shape[1]
        data3d = np.empty((npoints, 3))
        data3d[:] = np.nan
        for point in range(npoints):
            subp = data_2d_undistort[:, point, :]
            good = ~np.isnan(subp[:, 0])
            if np.sum(good) >= 2:
                data3d[point] = triangulate_simple(subp[good], cam_mats_extrinsic[good])

        # Reshape to frames x landmarks x 3
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))

        # Get FPS from video
        vidnames = sorted(glob.glob(os.path.join(main_folder, 'videos', trialname, '*.avi')))
        container = av.open(vidnames[0])
        video_stream = container.streams.video[0]
        if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
            fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
        else:
            fps = 30.0
        container.close()

        # Smooth 3D data
        data3d = smooth3d(data3d, fps=fps,
                          centroid_frequency_cutoff=gui_options['hand_centroid_lfc'],
                          point_frequency_cutoff=gui_options['all_landmarks_lfc'])

        # Re-flatten
        data3d = data3d.reshape(-1, 3)

        # Project back to 2D
        data3d_homogeneous = np.hstack([data3d, np.ones((data3d.shape[0], 1))])
        data_2d_new = np.zeros((ncams, data3d.shape[0], 2))
        for cam in range(ncams):
            projected = project_3d_to_2d(
                data3d_homogeneous.transpose(),
                cam_mats_intrinsic[cam],
                cam_mats_extrinsic[cam]
            ).transpose()
            data_2d_new[cam, :, :] = projected
        data_2d_new = data_2d_new.reshape((ncams, int(len(data3d) / nlandmarks), nlandmarks, 2))

        # Save refined 2D landmarks
        np.save(os.path.join(outdir_data2d, trialname, f'{trialname}_2Dlandmarksrefined'), data_2d_new)

        # Reshape 3D data and save
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))
        np.save(os.path.join(outdir_data3d, trialname, f'{trialname}_3Dlandmarks'), data3d)

        # Output directories for visualizations
        outdir_images_trialfolder = os.path.join(outdir_images_refined, trialname, 'data3d')
        outdir_video_trialfolder = os.path.join(outdir_video, trialname)
        os.makedirs(outdir_images_trialfolder, exist_ok=True)
        os.makedirs(outdir_video_trialfolder, exist_ok=True)

        # Visualize 3D data
        if gui_options.get('save_images_triangulation', False):
            print('Saving images.')
            visualize_3d(data3d, save_path=os.path.join(outdir_images_trialfolder, 'frame_{:04d}.png'))

        if gui_options.get('save_video_triangulation', False):
            print('Saving video.')
            createvideo(
                image_folder=outdir_images_trialfolder,
                extension='.png',
                fps=fps,
                output_folder=outdir_video_trialfolder,
                video_name='data3d.mp4'
            )

        # Visualize 2D labels
        if gui_options.get('save_images_triangulation', False):
            print('Saving refined images.')
            visualizelabels(
                vidnames,
                data=data_2d_new,
                outdir_images_refined=outdir_images_refined,
                trialname=trialname
            )

        if gui_options.get('save_video_triangulation', False):
            print('Saving refined videos.')
            for cam in range(ncams):
                imagefolder = os.path.join(outdir_images_refined, trialname, f'cam{cam}')
                createvideo(
                    image_folder=imagefolder,
                    extension='.png',
                    fps=fps,
                    output_folder=os.path.join(outdir_video, trialname),
                    video_name=f'cam{cam}_refined.mp4'
                )