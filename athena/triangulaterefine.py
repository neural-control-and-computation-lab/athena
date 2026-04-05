"""
Triangulation, refinement, smoothing and visualisation of multi-camera landmarks.

Pipeline (executed by ``main``):
    1. Load per-camera 2D landmarks (body, hands, optionally face).
    2. Correct left/right hand swaps across cameras (``_switch_hands``).
    3. Undistort 2D points using camera intrinsics.
    4. Optionally refine camera extrinsics via bundle adjustment
       (``calibration_refine.refine_calibration``).
    5. Triangulate to 3D via DLT with iterative reprojection-error camera
       filtering (``_triangulate_with_filtering``).
    6. Temporal smoothing with a Savitzky-Golay low-pass filter (``_smooth3d``).
    6. Reproject smoothed 3D landmarks back to each camera and save.
    7. Optionally triangulate HaMeR hand-mesh vertices.
    8. Generate visualisation images/videos (2D overlays and 3D skeleton).
"""

import os
import glob
import json
import cv2 as cv
import av
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import savgol_filter
from athena.labels2d import create_video, read_calibration
from athena.visualization import (
    SKELETON_LINKS, BODY_LINKS, SKELETON_COLOURS_HEX, BODY_COLOURS_HEX,
    SKELETON_COLOURS_BGR, MESH_RIGHT_COLOUR_BGR, MESH_LEFT_COLOUR_BGR,
    FACE_COLOUR_BGR, hex_to_bgr, render_mesh_overlay,
)


def _build_view_matrix(elev_deg=-60, azim_deg=-50):
    """Build a 3×3 viewing rotation matrix from elevation and azimuth angles."""
    elev_rad = np.radians(elev_deg)
    azim_rad = np.radians(azim_deg)
    Rz = np.array([[np.cos(azim_rad), -np.sin(azim_rad), 0],
                    [np.sin(azim_rad),  np.cos(azim_rad), 0],
                    [0,                 0,                 1]])
    Rx = np.array([[1, 0,                 0],
                    [0, np.cos(elev_rad), -np.sin(elev_rad)],
                    [0, np.sin(elev_rad),  np.cos(elev_rad)]])
    return Rx @ Rz


def _compute_viz_bounds(p3ds):
    """Compute axis-aligned bounding box for 3D visualization.

    Uses hand landmarks (33–74) when only body+hands are present, or
    hands + face (33+) when face landmarks are available.
    """
    n_lm = p3ds.shape[1]
    # Include hands (33–74) and face (75+) when present
    start = 33
    end = n_lm
    percentile = 25  # mid 50th percentile
    datalow = np.nanmin(np.nanpercentile(
        p3ds[:, start:end, :], percentile, axis=0), axis=0)
    datahigh = np.nanmax(np.nanpercentile(
        p3ds[:, start:end, :], 100 - percentile, axis=0), axis=0)
    dataint = datahigh - datalow
    datamid = (dataint / 2) + datalow
    largestint = np.max(dataint)
    lowerlim = datamid - largestint
    upperlim = datamid + largestint
    return lowerlim, upperlim


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def _undistort_points(points, matrix, dist):
    """Undistort 2D points using a camera intrinsic matrix and distortion coefficients.

    Parameters:
        points (np.ndarray): Shape (n_points, 2) pixel coordinates.
        matrix (np.ndarray): 3x3 camera intrinsic matrix.
        dist (np.ndarray): Distortion coefficients.

    Returns:
        np.ndarray: Undistorted normalised camera coordinates.
    """
    points = points.reshape(-1, 1, 2)
    return cv.undistortPoints(points, matrix, dist)


def _triangulate_batch(points_2d, cam_mats_extrinsic):
    """
    Batch-triangulate 2D points from multiple cameras to 3D via DLT + SVD.

    Points are grouped by their camera-visibility bitmask so that all points
    sharing the same set of visible cameras are triangulated in a single
    vectorised SVD call.  All visible cameras contribute equally.

    Parameters:
        points_2d (np.ndarray): Undistorted normalised 2D points,
            shape (ncams, npoints, 2).  NaN where the landmark is missing.
        cam_mats_extrinsic (np.ndarray): Camera extrinsic matrices,
            shape (ncams, 3, 4) or (ncams, 4, 4).

    Returns:
        np.ndarray: Triangulated 3D points, shape (npoints, 3).
            NaN where fewer than 2 cameras observed the point.
    """
    ncams, npoints, _ = points_2d.shape
    data3d = np.empty((npoints, 3))
    data3d[:] = np.nan

    # Build a visibility mask and group points by their camera-visibility pattern
    good_mask = ~np.isnan(points_2d[:, :, 0])  # (ncams, npoints)

    # Encode each point's visibility as a bitmask for fast grouping
    patterns = np.zeros(npoints, dtype=np.int32)
    for cam in range(ncams):
        patterns += good_mask[cam].astype(np.int32) << cam

    # Process each unique visibility pattern as a batch
    for pat in np.unique(patterns):
        ncams_pat = bin(pat).count('1')
        if ncams_pat < 2:
            continue

        active_cams = [c for c in range(ncams) if (pat >> c) & 1]
        point_indices = np.where(patterns == pat)[0]
        npts_batch = len(point_indices)

        mats = cam_mats_extrinsic[active_cams]  # (ncams_pat, 3or4, 4)
        pts = points_2d[active_cams][:, point_indices, :]  # (ncams_pat, npts_batch, 2)

        # Build the constraint matrix A for all points in this batch
        A = np.zeros((npts_batch, ncams_pat * 2, 4))
        for i in range(ncams_pat):
            x = pts[i, :, 0]  # (npts_batch,)
            y = pts[i, :, 1]
            mat = mats[i]     # (3or4, 4) — use first 3 rows
            A[:, 2 * i, :] = x[:, None] * mat[2][None, :] - mat[0][None, :]
            A[:, 2 * i + 1, :] = y[:, None] * mat[2][None, :] - mat[1][None, :]

        # Batch SVD
        _, _, vh = np.linalg.svd(A, full_matrices=True)
        p3d = vh[:, -1, :]
        p3d = p3d[:, :3] / p3d[:, 3:4]

        data3d[point_indices] = p3d

    return data3d


def _batch_reproject(points_3d, cam_mats_intrinsic, cam_mats_extrinsic):
    """
    Project 3D points back to 2D pixel coordinates for every camera (vectorized).

    Parameters:
        points_3d (np.ndarray): Shape (npoints, 3).
        cam_mats_intrinsic (list or np.ndarray): Per-camera intrinsic matrices, each (3, 3).
        cam_mats_extrinsic (np.ndarray): Shape (ncams, 3or4, 4).

    Returns:
        np.ndarray: Projected 2D pixel coordinates, shape (ncams, npoints, 2).
            NaN where the input 3D point is NaN.
    """
    ncams = len(cam_mats_intrinsic)
    npoints = points_3d.shape[0]
    result = np.full((ncams, npoints, 2), np.nan)

    # Identify valid (non-NaN) 3D points
    valid = ~np.isnan(points_3d[:, 0])
    if not np.any(valid):
        return result

    pts_valid = points_3d[valid]                                     # (N, 3)
    pts_h = np.hstack([pts_valid, np.ones((pts_valid.shape[0], 1))]) # (N, 4)

    for cam in range(ncams):
        E = cam_mats_extrinsic[cam][:3]          # (3, 4)
        K = np.asarray(cam_mats_intrinsic[cam])   # (3, 3)
        proj = K @ E                               # (3, 4)
        img_h = (proj @ pts_h.T)                   # (3, N)
        # Guard against divide-by-zero (points at or behind camera plane)
        z = img_h[2:3]
        z[z == 0] = 1e-12
        uv = img_h[:2] / z                        # (2, N)
        result[cam, valid, :] = uv.T

    return result


def _triangulate_with_filtering(points_2d_undist, points_2d_px,
                                cam_mats_extrinsic, cam_mats_intrinsic,
                                reproj_threshold=15.0, min_cams=2,
                                max_iterations=3):
    """
    DLT triangulation with iterative reprojection-error camera filtering.

    For each point the algorithm:
      1. Triangulates using all visible cameras (equal-weight DLT).
      2. Reprojects the 3D result back to every camera's pixel space.
      3. If the worst camera's reprojection error exceeds *reproj_threshold* and
         more than *min_cams* cameras remain, that camera is masked out and the
         point is re-triangulated.
      4. Steps 2–3 repeat for up to *max_iterations*.

    Parameters:
        points_2d_undist (np.ndarray): Undistorted normalised 2D coords, (ncams, npoints, 2).
        points_2d_px (np.ndarray): Original pixel 2D coords, (ncams, npoints, 2).
            Used for reprojection error computation.  NaN where missing.
        cam_mats_extrinsic (np.ndarray): (ncams, 3or4, 4).
        cam_mats_intrinsic (list): Per-camera (3, 3) intrinsic matrices.
        reproj_threshold (float): Pixel-space error threshold to reject a camera.
        min_cams (int): Never reduce below this many cameras per point.
        max_iterations (int): Maximum filtering passes.

    Returns:
        np.ndarray: Triangulated 3D points, shape (npoints, 3).
    """
    ncams, npoints, _ = points_2d_undist.shape

    # Working copy — we will NaN-out rejected cameras per point
    undist_work = points_2d_undist.copy()

    # Initial triangulation
    data3d = _triangulate_batch(undist_work, cam_mats_extrinsic)

    for _iteration in range(max_iterations):
        # Reproject to pixel space
        reproj = _batch_reproject(data3d, cam_mats_intrinsic, cam_mats_extrinsic)  # (ncams, npoints, 2)

        # Compute per-camera per-point Euclidean error against original pixel coords
        err = np.sqrt(np.nansum((reproj - points_2d_px) ** 2, axis=2))  # (ncams, npoints)

        # Determine which cameras are still active (not NaN) per point
        active = ~np.isnan(undist_work[:, :, 0])  # (ncams, npoints)
        n_active = active.astype(int).sum(axis=0)  # (npoints,)

        # For each point find the worst camera
        err_masked = err.copy()
        err_masked[~active] = -1  # ignore inactive cameras
        worst_cam = np.argmax(err_masked, axis=0)  # (npoints,)
        worst_err = err_masked[worst_cam, np.arange(npoints)]

        # Points to update: worst error exceeds threshold AND we can still drop a camera
        needs_update = (worst_err > reproj_threshold) & (n_active > min_cams) & ~np.isnan(data3d[:, 0])

        if not np.any(needs_update):
            break

        # Mask out the worst camera for each affected point
        update_indices = np.where(needs_update)[0]
        for idx in update_indices:
            cam_to_remove = worst_cam[idx]
            undist_work[cam_to_remove, idx, :] = np.nan

        # Re-triangulate affected points
        data3d = _triangulate_batch(undist_work, cam_mats_extrinsic)

    return data3d


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _project_3d_to_2d(X_world, intrinsic_matrix, extrinsic_matrix):
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


def _restore_long_nan_runs(original_data, filtered_data, min_length=5):
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


def _smooth3d(data3d, fps, frequency_cutoff=20, polyorder=3):
    """
    Apply Savitzky-Golay temporal smoothing to every landmark independently.

    Parameters:
        data3d (np.ndarray): 3D data array of shape [frames, landmarks, coordinates].
        fps (float): Frames per second of the data.
        frequency_cutoff (float): Desired cutoff frequency (Hz) for the low-pass filter.
        polyorder (int): Polynomial order for the Savitzky-Golay filter.

    Returns:
        np.ndarray: Smoothed 3D data.
    """
    n_frames, n_landmarks, _ = data3d.shape

    # Determine window length from cutoff frequency
    window_length = int(fps / frequency_cutoff * 2 + 1)
    window_length = max(3, window_length | 1)          # must be odd and >= 3
    window_length = min(window_length, n_frames | 1)   # can't exceed frame count

    data3d_smoothed = data3d.copy()

    for coord in range(3):
        for landmark in range(n_landmarks):
            data = data3d_smoothed[:, landmark, coord]
            nan_mask = np.isnan(data)
            valid_mask = ~nan_mask
            if np.sum(valid_mask) < window_length:
                continue
            data_valid = data[valid_mask]
            indices = np.arange(len(data))
            data_interp = np.interp(indices, indices[valid_mask], data_valid)
            data_filtered = savgol_filter(data_interp, window_length=window_length,
                                          polyorder=polyorder)
            data_filtered = _restore_long_nan_runs(data, data_filtered, min_length=5)
            data3d_smoothed[:, landmark, coord] = data_filtered

    return data3d_smoothed


def _switch_hands(data2d, ncams, nframes, nlandmarks, cam_mats_intrinsic, cam_mats_extrinsic,
                 verts_2d_left=None, verts_2d_right=None):
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
        verts_2d_left (np.ndarray, optional): Left hand vertex 2D data,
            shape (ncams, nframes, n_verts, 2). Swapped in parallel with keypoints.
        verts_2d_right (np.ndarray, optional): Right hand vertex 2D data,
            shape (ncams, nframes, n_verts, 2). Swapped in parallel with keypoints.

    Returns:
        np.ndarray: Corrected 2D landmark data with hands switched where necessary.
            If verts_2d_left and verts_2d_right are provided, they are modified in-place.
    """
    data_2d_switched = data2d.copy()
    has_verts = verts_2d_left is not None and verts_2d_right is not None

    # Part A: Switch hands based on proximity to wrists (vectorized swap)
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

        # Vectorized swap using boolean mask indexing
        if np.any(combined_condition_a):
            temp = data_2d_switched[cam, combined_condition_a, 33:54, :].copy()
            data_2d_switched[cam, combined_condition_a, 33:54, :] = data_2d_switched[cam, combined_condition_a, 54:75, :]
            data_2d_switched[cam, combined_condition_a, 54:75, :] = temp
            # Swap vertex 2D data in parallel
            if has_verts:
                temp_v = verts_2d_right[cam, combined_condition_a].copy()
                verts_2d_right[cam, combined_condition_a] = verts_2d_left[cam, combined_condition_a]
                verts_2d_left[cam, combined_condition_a] = temp_v

    # Part B: Use estimated 2D projections to further detect hand switching.
    # We triangulate hand roots with reprojection-error filtering so that
    # cameras with incorrectly-assigned hands (large reprojection error)
    # are automatically excluded, producing clean 3D estimates that can
    # reliably detect swaps on those cameras.
    data_2d = data_2d_switched.copy()
    nancondition = (data_2d[:, :, :, 0] == -1) & (data_2d[:, :, :, 1] == -1)
    data_2d[nancondition] = np.nan
    # Keep pixel-space copy for reprojection-error filtering
    data_2d_px = data_2d.copy()
    data_2d = data_2d.reshape((ncams, -1, 2))
    data_2d_undistort = np.empty(data_2d.shape)
    for cam in range(ncams):
        data_2d_undistort[cam] = _undistort_points(
            data_2d[cam].astype(float),
            cam_mats_intrinsic[cam],
            np.array([0, 0, 0, 0, 0])
        ).reshape(len(data_2d[cam]), 2)
    data_2d_undistort = data_2d_undistort.reshape((ncams, nframes, nlandmarks, 2))

    # Triangulate hand root landmarks with reprojection-error filtering
    # (landmark 54=left hand root, 33=right hand root).
    # Using filtered triangulation ensures cameras with wrong hand assignment
    # are excluded as outliers, giving clean 3D positions.
    lhand_pts_undist = data_2d_undistort[:, :, 54, :]  # (ncams, nframes, 2)
    rhand_pts_undist = data_2d_undistort[:, :, 33, :]  # (ncams, nframes, 2)
    lhand_pts_px = data_2d_px[:, :, 54, :]  # (ncams, nframes, 2)
    rhand_pts_px = data_2d_px[:, :, 33, :]  # (ncams, nframes, 2)
    lhand_3d = _triangulate_with_filtering(
        lhand_pts_undist, lhand_pts_px,
        cam_mats_extrinsic, cam_mats_intrinsic,
        reproj_threshold=30.0, min_cams=2)  # (nframes, 3)
    rhand_3d = _triangulate_with_filtering(
        rhand_pts_undist, rhand_pts_px,
        cam_mats_extrinsic, cam_mats_intrinsic,
        reproj_threshold=30.0, min_cams=2)  # (nframes, 3)

    # Vectorized projection back to 2D for all cameras at once
    handestimate = np.empty((ncams, nframes, 2, 2))
    handestimate[:] = np.nan
    cam_mats_intrinsic_arr = np.array(cam_mats_intrinsic)

    for hand_idx, hand_3d in enumerate([rhand_3d, lhand_3d]):
        valid = ~np.isnan(hand_3d[:, 0])
        if not np.any(valid):
            continue
        valid_pts = hand_3d[valid]  # (N, 3)
        pts_h = np.hstack([valid_pts, np.ones((len(valid_pts), 1))])  # (N, 4)
        for cam in range(ncams):
            X_camera = cam_mats_extrinsic[cam] @ pts_h.T  # (3or4, N)
            X_image = cam_mats_intrinsic_arr[cam] @ X_camera[:3]  # (3, N)
            uv = X_image[:2] / X_image[2:3]  # (2, N)
            handestimate[cam, valid, hand_idx, :] = uv.T

    # Part B swap: compare observed hand roots to triangulated estimates.
    # Temporarily replace -1 sentinels with NaN so distance comparisons
    # naturally produce NaN (and NaN < x is False, preventing false swaps).
    sentinel_mask = (data_2d_switched[:, :, :, 0] == -1) & (data_2d_switched[:, :, :, 1] == -1)
    data_2d_switched[sentinel_mask] = np.nan

    for cam in range(ncams):
        rhand = data_2d_switched[cam, :, 33, :]
        lhand = data_2d_switched[cam, :, 54, :]
        rhand_est = handestimate[cam, :, 0, :]
        lhand_est = handestimate[cam, :, 1, :]

        r_valid = ~np.isnan(rhand[:, 0])
        l_valid = ~np.isnan(lhand[:, 0])
        rest_valid = ~np.isnan(rhand_est[:, 0])
        lest_valid = ~np.isnan(lhand_est[:, 0])

        norm_rvsrest = np.linalg.norm(rhand - rhand_est, axis=-1)
        norm_lvsrest = np.linalg.norm(lhand - rhand_est, axis=-1)
        norm_rvslest = np.linalg.norm(rhand - lhand_est, axis=-1)
        norm_lvslest = np.linalg.norm(lhand - lhand_est, axis=-1)

        # Both hands detected: swap if hands are closer to the wrong estimates
        c1 = norm_lvsrest < norm_rvsrest
        c2 = norm_lvsrest < norm_lvslest
        condition1b = c1 & c2

        c3 = norm_rvslest < norm_lvslest
        c4 = norm_rvslest < norm_rvsrest
        condition2b = c3 & c4

        combined_condition_b = condition1b | condition2b

        # Single-hand frames: only one hand detected, check if it belongs
        # to the other side based on triangulated 3D estimates.
        # Only-right detected but closer to left estimate → swap
        only_r = r_valid & ~l_valid & rest_valid & lest_valid
        condition_only_r_swap = only_r & (norm_rvslest < norm_rvsrest)
        # Only-left detected but closer to right estimate → swap
        only_l = ~r_valid & l_valid & rest_valid & lest_valid
        condition_only_l_swap = only_l & (norm_lvsrest < norm_lvslest)

        combined_condition_b = combined_condition_b | condition_only_r_swap | condition_only_l_swap

        # Vectorized swap using boolean mask indexing
        if np.any(combined_condition_b):
            temp = data_2d_switched[cam, combined_condition_b, 33:54, :].copy()
            data_2d_switched[cam, combined_condition_b, 33:54, :] = data_2d_switched[cam, combined_condition_b, 54:75, :]
            data_2d_switched[cam, combined_condition_b, 54:75, :] = temp
            # Swap vertex 2D data in parallel
            if has_verts:
                temp_v = verts_2d_right[cam, combined_condition_b].copy()
                verts_2d_right[cam, combined_condition_b] = verts_2d_left[cam, combined_condition_b]
                verts_2d_left[cam, combined_condition_b] = temp_v

    # Restore -1 sentinels (NaN back to -1 for downstream compatibility)
    nan_mask = np.isnan(data_2d_switched[:, :, :, 0])
    data_2d_switched[nan_mask] = -1

    return data_2d_switched


def _render_mesh_on_camera_frame(img, verts_2d, verts_cam, faces, base_colour_bgr,
                                  alpha=0.7):
    """
    Render a filled, depth-sorted, shaded mesh onto a camera video frame.

    Parameters:
        img (np.ndarray): BGR image to draw on (modified in-place).
        verts_2d (np.ndarray): (778, 2) projected 2D pixel coordinates.
        verts_cam (np.ndarray): (778, 3) vertices in camera coordinate space (for depth/shading).
        faces (np.ndarray): (n_faces, 3) triangle indices.
        base_colour_bgr (tuple): Base BGR colour for the mesh.
        alpha (float): Blending factor for mesh overlay (0 = transparent, 1 = opaque).
    """
    render_mesh_overlay(img, verts_2d, verts_cam, faces, base_colour_bgr, alpha)


def _process_camera(cam, input_stream, data, display_width, display_height, outdir_images_refined, trialname,
                   mesh_verts_2d=None, mesh_verts_cam=None, mesh_faces=None):
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
        mesh_verts_2d (dict, optional): {'left': (nframes, 778, 2), 'right': (nframes, 778, 2)} projected mesh vertices.
        mesh_verts_cam (dict, optional): {'left': (nframes, 778, 3), 'right': (nframes, 778, 3)} camera-space vertices.
        mesh_faces (np.ndarray, optional): (n_faces, 3) triangle indices.
    """
    try:
        container = av.open(input_stream)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        colours_bgr = [hex_to_bgr(c) for c in SKELETON_COLOURS_HEX]
        links = SKELETON_LINKS

        for framenum, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")

                if framenum < data.shape[1] and not np.isnan(data[cam, framenum, :, 0]).all():
                    has_mesh = mesh_verts_2d is not None and mesh_faces is not None

                    # Draw mesh overlays first (behind skeleton)
                    if has_mesh:
                        # Left hand needs reversed face winding after X-flip
                        faces_left = mesh_faces[:, [0, 2, 1]]
                        for side, colour, f in [('right', MESH_RIGHT_COLOUR_BGR, mesh_faces),
                                                ('left', MESH_LEFT_COLOUR_BGR, faces_left)]:
                            v2d = mesh_verts_2d[side][framenum]   # (778, 2)
                            vcam = mesh_verts_cam[side][framenum] # (778, 3)
                            # Skip if no detection (sentinel -1)
                            if v2d[0, 0] != -1 and not np.isnan(v2d[0, 0]):
                                _render_mesh_on_camera_frame(img, v2d, vcam, f, colour)

                    # Draw skeleton links — skip hand links when mesh data is present
                    # Links 0–15 are body, 16–35 are right hand, 36–55 are left hand
                    for number, link in enumerate(links):
                        if has_mesh and number >= 16:
                            continue  # skip hand skeleton when we have meshes
                        start, end = link
                        if not np.isnan(data[cam, framenum, [start, end], 0]).any():
                            posn_start = tuple(data[cam, framenum, start, :2].astype(int))
                            posn_end = tuple(data[cam, framenum, end, :2].astype(int))
                            cv.line(img, posn_start, posn_end, colours_bgr[number], 2)

                    # Draw face landmarks as small dots (indices 75+)
                    n_lm = data.shape[2]
                    if n_lm > 75:
                        for i in range(75, n_lm):
                            if not np.isnan(data[cam, framenum, i, 0]):
                                pt = tuple(data[cam, framenum, i, :2].astype(int))
                                cv.circle(img, pt, 1, FACE_COLOUR_BGR, -1, cv.LINE_AA)

                    resized_frame = cv.resize(img, (display_width, display_height))
                    output_path = os.path.join(outdir_images_refined, trialname, f'cam{cam}', f'frame{framenum:06d}.jpg')
                    cv.imwrite(output_path, resized_frame, [cv.IMWRITE_JPEG_QUALITY, 50])

                elif framenum >= data.shape[1]:
                    break

    except Exception as e:
        print(f"Error processing camera {cam}, frame {framenum}: {e}")
    finally:
        container.close()


def _visualize_labels(input_streams, data, display_width=450, display_height=360, outdir_images_refined='', trialname='',
                    per_cam_mesh_data=None):
    """
    Draws 2D hand landmarks on videos, optionally with HaMeR mesh overlays.

    Parameters:
        input_streams (list): List of video file paths.
        data (np.ndarray): 2D hand landmarks.
        display_width (int, optional): Display width for resizing images. Defaults to 450.
        display_height (int, optional): Display height for resizing images. Defaults to 360.
        outdir_images_refined (str, optional): Output directory for refined images.
        trialname (str, optional): Name of the trial.
        per_cam_mesh_data (list, optional): Per-camera mesh data. Each element is a dict with
            keys 'verts_2d', 'verts_cam', 'faces' or None if no mesh data for that camera.
    """
    max_workers = min(len(input_streams), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cam in range(len(input_streams)):
            cam_mesh = per_cam_mesh_data[cam] if per_cam_mesh_data is not None else None
            mesh_v2d = cam_mesh['verts_2d'] if cam_mesh is not None else None
            mesh_vcam = cam_mesh['verts_cam'] if cam_mesh is not None else None
            mesh_f = cam_mesh['faces'] if cam_mesh is not None else None
            futures.append(executor.submit(
                _process_camera,
                cam, input_streams[cam], data, display_width, display_height,
                outdir_images_refined, trialname,
                mesh_verts_2d=mesh_v2d, mesh_verts_cam=mesh_vcam, mesh_faces=mesh_f
            ))

        for future in futures:
            future.result()


def _project_3d_to_image(points_3d, view_mat, img_width, img_height, lowerlim, upperlim):
    """
    Projects 3D world points to 2D image pixel coordinates using a simple orthographic-like
    projection defined by the view matrix and axis limits.

    Parameters:
        points_3d (np.ndarray): 3D points, shape (N, 3).
        view_mat (np.ndarray): 3x3 rotation matrix for the viewing angle.
        img_width (int): Output image width.
        img_height (int): Output image height.
        lowerlim (np.ndarray): Lower axis limits (3,).
        upperlim (np.ndarray): Upper axis limits (3,).

    Returns:
        np.ndarray: 2D pixel coordinates, shape (N, 2), as integers. NaN rows yield (-1, -1).
    """
    # Normalize to [0, 1] range
    span = upperlim - lowerlim
    span[span == 0] = 1.0
    normalized = (points_3d - lowerlim) / span  # (N, 3) in [0, 1]

    # Apply view rotation (centered)
    centered = normalized - 0.5
    rotated = (view_mat @ centered.T).T + 0.5  # (N, 3)

    # Project: use X for horizontal, Z for vertical (inverted)
    margin = 0.1
    usable_w = img_width * (1 - 2 * margin)
    usable_h = img_height * (1 - 2 * margin)
    # Suppress NaN cast warnings; NaN rows are overwritten below
    with np.errstate(invalid='ignore'):
        px = (rotated[:, 0] * usable_w + img_width * margin).astype(int)
        py = ((1 - rotated[:, 2]) * usable_h + img_height * margin).astype(int)

    coords = np.stack([px, py], axis=-1)
    nan_mask = np.isnan(points_3d).any(axis=-1)
    coords[nan_mask] = -1
    return coords


def _visualize_3d(p3ds, save_path=None):
    """
    Visualize 3D points using OpenCV rendering (much faster than matplotlib).

    Parameters:
        p3ds (np.ndarray): 3D points, shape (n_frames, n_landmarks, 3).
        save_path (str, optional): If provided, saves the images to the specified path format.
    """

    colours_bgr = [hex_to_bgr(c) for c in SKELETON_COLOURS_HEX]
    links = SKELETON_LINKS
    lowerlim, upperlim = _compute_viz_bounds(p3ds)
    view_mat = _build_view_matrix()

    img_width, img_height = 640, 480

    for framenum in tqdm(range(len(p3ds))):
        # Create blank image (dark background like matplotlib default)
        img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

        frame_pts = p3ds[framenum]  # (n_landmarks, 3)
        coords = _project_3d_to_image(frame_pts, view_mat, img_width, img_height, lowerlim, upperlim)

        # Draw links
        for linknum, link in enumerate(links):
            start_idx, end_idx = link
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            if pt1[0] < 0 or pt2[0] < 0:
                continue
            cv.line(img, tuple(pt1), tuple(pt2), colours_bgr[linknum], 3, cv.LINE_AA)

        # Draw scatter points for hand landmarks (33:75)
        for i in range(33, 75):
            pt = coords[i]
            if pt[0] < 0:
                continue
            cv.circle(img, tuple(pt), 4, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(img, tuple(pt), 4, (0, 0, 0), 1, cv.LINE_AA)

        # Draw face landmarks as small dots (indices 75+)
        n_lm = len(frame_pts)
        if n_lm > 75:
            for i in range(75, n_lm):
                pt = coords[i]
                if pt[0] < 0:
                    continue
                cv.circle(img, tuple(pt), 1, FACE_COLOUR_BGR, -1, cv.LINE_AA)

        if save_path is not None:
            cv.imwrite(save_path.format(framenum), img)
        else:
            cv.imshow('3D Visualization', img)
            cv.waitKey(10)


def _render_mesh_on_image(img, aligned_verts, faces, view_mat, img_width, img_height,
                          lowerlim, upperlim, base_colour_bgr):
    """
    Render a filled, depth-sorted, shaded mesh onto an OpenCV image.

    Triangles are painted back-to-front with simple Lambertian shading
    based on face normals (light from camera direction).

    Parameters:
        img (np.ndarray): BGR image to draw on (modified in-place).
        aligned_verts (np.ndarray): (778, 3) mesh vertices in world space.
        faces (np.ndarray): (n_faces, 3) triangle indices.
        view_mat (np.ndarray): 3x3 viewing rotation matrix.
        img_width, img_height (int): Image dimensions.
        lowerlim, upperlim (np.ndarray): Axis bounds for normalisation.
        base_colour_bgr (tuple): Base BGR colour for the mesh.
    """
    # Transform vertices to view space (for depth sorting and shading)
    span = upperlim - lowerlim
    span[span == 0] = 1.0
    normalized = (aligned_verts - lowerlim) / span
    centered = normalized - 0.5
    rotated = (view_mat @ centered.T).T + 0.5   # (778, 3)

    # Project to 2D pixel coordinates
    margin = 0.1
    usable_w = img_width * (1 - 2 * margin)
    usable_h = img_height * (1 - 2 * margin)
    px = (rotated[:, 0] * usable_w + img_width * margin).astype(np.float32)
    py = ((1 - rotated[:, 2]) * usable_h + img_height * margin).astype(np.float32)
    coords_2d = np.stack([px, py], axis=-1)  # (778, 2)

    # Compute per-face depth (average Z in view space) for back-to-front sorting
    tri_verts = rotated[faces]  # (n_faces, 3, 3)
    face_depths = tri_verts[:, :, 1].mean(axis=1)  # Y in view space = depth

    # Compute face normals in view space for shading
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)  # (n_faces, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals = normals / norms

    # Light direction: towards camera (negative Y in view space)
    light_dir = np.array([0.0, -1.0, 0.0])
    diffuse = np.abs(normals @ light_dir)  # abs for double-sided
    # Ambient + diffuse shading
    shade = 0.35 + 0.65 * diffuse  # range [0.35, 1.0]

    # Sort faces back-to-front (painters algorithm)
    sort_order = np.argsort(face_depths)

    # Draw filled triangles
    base_b, base_g, base_r = float(base_colour_bgr[0]), float(base_colour_bgr[1]), float(base_colour_bgr[2])
    tri_2d = coords_2d[faces]  # (n_faces, 3, 2)

    for idx in sort_order:
        s = shade[idx]
        colour = (int(base_b * s), int(base_g * s), int(base_r * s))
        pts = tri_2d[idx].astype(np.int32).reshape((-1, 1, 2))
        cv.fillPoly(img, [pts], colour)

    # Draw thin edges on top for definition
    edge_colour = (
        max(0, int(base_b * 0.5)),
        max(0, int(base_g * 0.5)),
        max(0, int(base_r * 0.5)),
    )
    # Use unique edges to avoid overdraw
    edge_set = set()
    for f_idx in faces:
        for i in range(3):
            e = tuple(sorted((f_idx[i], f_idx[(i + 1) % 3])))
            edge_set.add(e)
    for e0, e1 in edge_set:
        p0 = coords_2d[e0].astype(int)
        p1 = coords_2d[e1].astype(int)
        cv.line(img, tuple(p0), tuple(p1), edge_colour, 1, cv.LINE_AA)


def _visualize_3d_mesh(p3ds, vertices_left, vertices_right, faces,
                      save_path=None):
    """
    Visualize 3D skeleton + shaded hand meshes using OpenCV rendering.

    Renders the body skeleton (landmarks 0–32) as before, plus hand meshes
    (778-vertex MANO triangles) triangulated from multi-camera 2D observations.
    Meshes are rendered with depth-sorted filled triangles and simple
    Lambertian shading.

    Parameters:
        p3ds (np.ndarray): Triangulated 3D keypoints (n_frames, 75, 3).
        vertices_left (np.ndarray): Left hand mesh vertices in world space (n_frames, 778, 3).
            NaN where not triangulated.
        vertices_right (np.ndarray): Right hand mesh vertices in world space (n_frames, 778, 3).
            NaN where not triangulated.
        faces (np.ndarray): Triangle face indices (n_faces, 3), typically (1538, 3).
        save_path (str, optional): Format string for saving frames.
    """

    body_colours_bgr = [hex_to_bgr(c) for c in BODY_COLOURS_HEX]
    lowerlim, upperlim = _compute_viz_bounds(p3ds)
    view_mat = _build_view_matrix()

    img_width, img_height = 640, 480

    for framenum in tqdm(range(len(p3ds))):
        img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

        frame_pts = p3ds[framenum]  # (75, 3)
        coords = _project_3d_to_image(frame_pts, view_mat, img_width, img_height, lowerlim, upperlim)

        # Draw body skeleton links
        for linknum, link in enumerate(BODY_LINKS):
            pt1 = coords[link[0]]
            pt2 = coords[link[1]]
            if pt1[0] < 0 or pt2[0] < 0:
                continue
            cv.line(img, tuple(pt1), tuple(pt2), body_colours_bgr[linknum], 3, cv.LINE_AA)

        # Draw hand meshes (right then left)
        # Vertices are already in world space (triangulated from multi-camera 2D).
        # Reverse face winding for left hand for correct surface normals.
        faces_left = faces[:, [0, 2, 1]]
        for verts_frame, colour, f in [
            (vertices_right[framenum], MESH_RIGHT_COLOUR_BGR, faces),
            (vertices_left[framenum], MESH_LEFT_COLOUR_BGR, faces_left),
        ]:
            # Skip if mesh data is missing (NaN)
            if np.isnan(verts_frame[0, 0]):
                continue

            # Render shaded filled mesh
            _render_mesh_on_image(
                img, verts_frame, f, view_mat,
                img_width, img_height, lowerlim, upperlim, colour
            )

        # Draw face landmarks as small dots (indices 75+)
        n_lm = len(frame_pts)
        if n_lm > 75:
            for i in range(75, n_lm):
                pt = coords[i]
                if pt[0] < 0:
                    continue
                cv.circle(img, tuple(pt), 1, FACE_COLOUR_BGR, -1, cv.LINE_AA)

        if save_path is not None:
            cv.imwrite(save_path.format(framenum), img)
        else:
            cv.imshow('3D Mesh Visualization', img)
            cv.waitKey(10)


def main(gui_options_json):
    """Run the full triangulation and refinement pipeline.

    Loads per-camera 2D landmarks, corrects hand swaps, triangulates to 3D
    with outlier camera filtering, applies temporal smoothing, reprojects
    back to 2D, and optionally generates visualisation images/videos.

    Parameters:
        gui_options_json (str): JSON string of GUI options containing paths,
            trial folders, and processing flags.
    """
    gui_options = json.loads(gui_options_json)

    # Set directories
    idfolders = gui_options['idfolders']
    main_folder = gui_options['main_folder']
    trials = sorted([os.path.join(main_folder, 'landmarks', os.path.basename(f)) for f in idfolders])

    # Camera calibration
    if glob.glob(os.path.join(main_folder, 'calibration', '*.yaml')):
        calfileext = '*.yaml'
    elif glob.glob(os.path.join(main_folder, 'calibration', '*.toml')):
        calfileext = '*.toml'
    calfiles = sorted(glob.glob(os.path.join(main_folder, 'calibration', calfileext)))
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = read_calibration(calfiles, calfileext)
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)
    ncams = cam_mats_extrinsic.shape[0]

    # Output directories
    outdir_images_refined = os.path.join(main_folder, 'imagesrefined')
    outdir_video = os.path.join(main_folder, 'videos_processed')
    outdir_landmarks = os.path.join(main_folder, 'landmarks')
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

        # Check for face landmark data (auto-detect, independent of GUI flag)
        has_face_data = True
        data_2d_face = []
        for cam in range(ncams):
            face_file = os.path.join(landmarkfiles[cam], '2Dlandmarks_face.npy')
            if os.path.exists(face_file):
                data_2d_face.append(np.load(face_file).astype(float))
            else:
                has_face_data = False
                break

        if has_face_data and data_2d_face:
            data_2d_face = np.stack(data_2d_face)
            n_face_valid = np.sum(data_2d_face[:, :, :, 0] != -1)
            n_face_total = data_2d_face[:, :, :, 0].size
            print(f'  Face landmarks loaded: {data_2d_face.shape} '
                  f'({n_face_valid}/{n_face_total} detected)')
        else:
            data_2d_face = None
            print('  No face landmark data found. '
                  'Enable "Include Face Mesh" during 2D detection to add face tracking.')

        # Combine 2D coordinate data (x, y only)
        parts = [data_2d_body[:, :, :, :2], data_2d_right[:, :, :, :2], data_2d_left[:, :, :, :2]]
        if data_2d_face is not None:
            parts.append(data_2d_face[:, :, :, :2])
        data_2d_combined = np.concatenate(parts, axis=2)

        # Video parameters
        nframes = data_2d_combined.shape[1]
        nlandmarks = data_2d_combined.shape[2]

        # Load HaMeR mesh vertex 2D data (if available) before _switch_hands
        # so that vertex data is swapped in parallel with keypoints.
        mesh_v2d_left = None
        mesh_v2d_right = None
        mesh_faces = None
        has_mesh_data = False
        faces_file = os.path.join(trial, 'hamer_faces.npy')
        if os.path.exists(faces_file):
            mesh_faces = np.load(faces_file)
            v2d_l_check = os.path.join(landmarkfiles[0], 'hamer_vertices_2d_left.npy')
            v2d_r_check = os.path.join(landmarkfiles[0], 'hamer_vertices_2d_right.npy')
            if os.path.exists(v2d_l_check) and os.path.exists(v2d_r_check):
                v2d_l_list = []
                v2d_r_list = []
                for cam in range(ncams):
                    v2d_l_list.append(np.load(os.path.join(landmarkfiles[cam], 'hamer_vertices_2d_left.npy')).astype(float))
                    v2d_r_list.append(np.load(os.path.join(landmarkfiles[cam], 'hamer_vertices_2d_right.npy')).astype(float))
                mesh_v2d_left = np.stack(v2d_l_list)   # (ncams, nframes, 778, 2)
                mesh_v2d_right = np.stack(v2d_r_list)  # (ncams, nframes, 778, 2)
                has_mesh_data = True
                print(f'  Loaded vertex 2D data: {mesh_v2d_left.shape}')

        # Optional: refine camera calibration via bundle adjustment
        if gui_options.get('refine_calibration', False):
            from .calibration_refine import refine_calibration as _refine_cal
            # Use combined 2D data (body+hands+face) in pixel space
            data_2d_for_refine = data_2d_combined.copy()
            refined_extrinsics, drift_report = _refine_cal(
                cam_mats_extrinsic, cam_mats_intrinsic,
                data_2d_for_refine, nframes, nlandmarks,
                verbose=True,
            )
            cam_mats_extrinsic = refined_extrinsics

            # Save refined calibration in a subdirectory so the original
            # calibration glob is not polluted on the next run.
            from .calibration_refine import save_refined_calibration
            refined_dir = os.path.join(main_folder, 'calibration', 'refined')
            os.makedirs(refined_dir, exist_ok=True)
            if calfileext == '*.toml':
                save_refined_calibration(
                    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs,
                    os.path.join(refined_dir, 'calibration_refined.toml'), fmt='toml',
                    metadata={'trial': trialname},
                )
            else:
                save_refined_calibration(
                    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs,
                    os.path.join(refined_dir, 'calibration_refined.yaml'), fmt='yaml',
                )

        # Switch hands (also swaps vertex 2D data if provided)
        data_2d = _switch_hands(
            data_2d_combined,
            ncams,
            nframes,
            nlandmarks,
            cam_mats_intrinsic,
            cam_mats_extrinsic,
            verts_2d_left=mesh_v2d_left,
            verts_2d_right=mesh_v2d_right,
        )

        data_2d = data_2d.reshape((ncams, -1, 2))

        if ncams != data_2d.shape[0]:
            raise ValueError('Number of cameras in calibration parameters does not match 2D data.')

        nancondition = (data_2d[:, :, 0] == -1) & (data_2d[:, :, 1] == -1)
        data_2d[nancondition, :] = np.nan

        # Keep pixel-space copy for reprojection-error filtering
        data_2d_px = data_2d.copy()  # (ncams, nframes*nlandmarks, 2) with NaN

        # Undistort 2D points
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = _undistort_points(
                data_2d[cam].astype(float),
                cam_mats_intrinsic[cam],
                np.array([0, 0, 0, 0, 0])
            ).reshape(len(data_2d[cam]), 2)

        # DLT triangulation with iterative reprojection-error camera filtering.
        # Threshold of 30px catches genuine outlier cameras while leaving normal
        # MediaPipe detection noise (up to ~15px) untouched.
        data3d = _triangulate_with_filtering(
            data_2d_undistort, data_2d_px,
            cam_mats_extrinsic, cam_mats_intrinsic,
            reproj_threshold=30.0, min_cams=2
        )

        # Reshape to frames x landmarks x 3
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))

        # Diagnostic: check 3D data range
        valid_3d = ~np.isnan(data3d[:, :, 0])
        print(f'  Triangulated: {np.sum(valid_3d)}/{valid_3d.size} valid 3D points, '
              f'range=[{np.nanmin(data3d):.3f}, {np.nanmax(data3d):.3f}]')
        if nlandmarks > 75:
            valid_face = ~np.isnan(data3d[:, 75:, 0])
            print(f'  Face landmarks: {np.sum(valid_face)}/{valid_face.size} valid 3D points')

        # Get FPS from video
        vidnames = sorted(glob.glob(os.path.join(main_folder, 'videos', trialname, '*.avi')) + glob.glob(os.path.join(main_folder, 'videos', trialname, '*.mp4')))
        container = av.open(vidnames[0])
        video_stream = container.streams.video[0]
        if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
            fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
        else:
            fps = 30.0
        container.close()

        # Savitzky-Golay smoothing on all landmarks (body + hand + face)
        data3d = _smooth3d(data3d, fps=fps,
            frequency_cutoff=gui_options['all_landmarks_lfc'])

        # Re-flatten
        data3d = data3d.reshape(-1, 3)

        # Project back to 2D
        data3d_homogeneous = np.hstack([data3d, np.ones((data3d.shape[0], 1))])
        data_2d_new = np.zeros((ncams, data3d.shape[0], 2))
        for cam in range(ncams):
            projected = _project_3d_to_2d(
                data3d_homogeneous.transpose(),
                cam_mats_intrinsic[cam],
                cam_mats_extrinsic[cam]
            ).transpose()
            data_2d_new[cam, :, :] = projected
        data_2d_new = data_2d_new.reshape((ncams, int(len(data3d) / nlandmarks), nlandmarks, 2))

        # Save refined 2D and 3D landmarks
        np.save(os.path.join(outdir_landmarks, trialname, f'{trialname}_2Dlandmarksrefined'), data_2d_new)
        data3d = data3d.reshape((int(len(data3d) / nlandmarks), nlandmarks, 3))
        np.save(os.path.join(outdir_landmarks, trialname, f'{trialname}_3Dlandmarks'), data3d)

        # Output directories for the trial (visualization)
        outdir_video_trialfolder = os.path.join(outdir_video, trialname)
        outdir_3dimages_trialfolder = os.path.join(outdir_images_refined, trialname, 'data3d')

        # Triangulate mesh vertices (if vertex 2D data was loaded)
        tri_verts_left_3d = None
        tri_verts_right_3d = None
        if has_mesh_data:
            print('  Triangulating mesh vertices across cameras...')
            n_verts = mesh_v2d_left.shape[2]  # 778

            for side, v2d_data in [
                ('left', mesh_v2d_left),
                ('right', mesh_v2d_right),
            ]:
                # Replace -1 with NaN for _triangulate_batch
                v2d = v2d_data.copy()
                nan_mask = (v2d[:, :, :, 0] == -1) & (v2d[:, :, :, 1] == -1)
                v2d[nan_mask] = np.nan

                # Undistort vertex 2D (same as keypoints)
                v2d_flat = v2d.reshape(ncams, -1, 2)
                v2d_undist = np.empty_like(v2d_flat)
                for cam in range(ncams):
                    v2d_undist[cam] = _undistort_points(
                        v2d_flat[cam].astype(float),
                        cam_mats_intrinsic[cam],
                        np.array([0, 0, 0, 0, 0])
                    ).reshape(-1, 2)

                # Triangulate
                verts_3d = _triangulate_batch(v2d_undist, cam_mats_extrinsic)
                verts_3d = verts_3d.reshape(nframes, n_verts, 3)

                if side == 'left':
                    tri_verts_left_3d = verts_3d
                else:
                    tri_verts_right_3d = verts_3d

            left_ok = np.sum(~np.isnan(tri_verts_left_3d[:, 0, 0]))
            right_ok = np.sum(~np.isnan(tri_verts_right_3d[:, 0, 0]))
            print(f'  Mesh triangulated ({mesh_faces.shape[0]} triangles, {n_verts} verts/hand). '
                  f'Coverage: left {left_ok}/{nframes}, right {right_ok}/{nframes}.')

        # Visualize 3D data
        if gui_options.get('save_images_triangulation', False):
            os.makedirs(outdir_3dimages_trialfolder, exist_ok=True)
            if has_mesh_data:
                print('Saving 3D mesh images.')
                _visualize_3d_mesh(
                    data3d, tri_verts_left_3d, tri_verts_right_3d, mesh_faces,
                    save_path=os.path.join(outdir_3dimages_trialfolder, 'frame_{:06d}.jpg')
                )
            else:
                print('Saving 3D images.')
                _visualize_3d(data3d, save_path=os.path.join(outdir_3dimages_trialfolder, 'frame_{:06d}.jpg'))

        if gui_options.get('save_video_triangulation', False):
            os.makedirs(outdir_video_trialfolder, exist_ok=True)
            print('Saving 3D video.')
            create_video(
                image_folder=outdir_3dimages_trialfolder,
                extension='.jpg',
                fps=fps,
                output_folder=outdir_video_trialfolder,
                video_name='data3d.mp4'
            )

        # Pre-compute per-camera 2D mesh projections for 2D overlay
        per_cam_mesh_data = None
        if has_mesh_data:
            print('  Projecting triangulated mesh to cameras...')
            nf = data3d.shape[0]
            n_verts = tri_verts_left_3d.shape[1]  # 778

            per_cam_mesh_data = []
            for cam in range(ncams):
                intrinsic = cam_mats_intrinsic[cam]           # (3, 3)
                extrinsic = cam_mats_extrinsic[cam][:3]       # (3, 4)
                proj_mat = intrinsic @ extrinsic              # (3, 4)

                cam_data = {}
                for side, verts_3d_all in [('right', tri_verts_right_3d), ('left', tri_verts_left_3d)]:
                    # Find valid frames (not NaN)
                    valid = ~np.isnan(verts_3d_all[:, 0, 0])
                    v2d_all = np.full((nf, n_verts, 2), -1, dtype=np.float32)
                    vcam_all = np.full((nf, n_verts, 3), -1, dtype=np.float32)

                    if valid.any():
                        verts_valid = verts_3d_all[valid].astype(np.float32)
                        n_valid_frames = verts_valid.shape[0]
                        ones = np.ones((n_valid_frames, n_verts, 1), dtype=np.float32)
                        verts_h = np.concatenate([verts_valid, ones], axis=2)

                        # Camera space
                        verts_cam_batch = np.einsum('ij,nkj->nki', extrinsic, verts_h)
                        vcam_all[valid] = verts_cam_batch

                        # Image space
                        verts_img_h = np.einsum('ij,nkj->nki', proj_mat, verts_h)
                        verts_img_h[:, :, :2] /= verts_img_h[:, :, 2:3]
                        v2d_all[valid] = verts_img_h[:, :, :2]

                    cam_data[side] = (v2d_all, vcam_all)

                per_cam_mesh_data.append({
                    'verts_2d': {'right': cam_data['right'][0], 'left': cam_data['left'][0]},
                    'verts_cam': {'right': cam_data['right'][1], 'left': cam_data['left'][1]},
                    'faces': mesh_faces,
                })
            print('  Mesh projection done.')

        # Visualize 2D labels
        if gui_options.get('save_images_refine', False):
            os.makedirs(os.path.join(outdir_images_refined, trialname), exist_ok=True)
            for cam in range(ncams):
                os.makedirs(os.path.join(outdir_images_refined, trialname, f'cam{cam}'), exist_ok=True)
            print('Saving refined 2D images.')
            _visualize_labels(
                vidnames,
                data=data_2d_new,
                outdir_images_refined=outdir_images_refined,
                trialname=trialname,
                per_cam_mesh_data=per_cam_mesh_data
            )

        if gui_options.get('save_video_refine', False):
            os.makedirs(outdir_video_trialfolder, exist_ok=True)
            print('Saving refined 2D videos.')
            for cam in range(ncams):
                imagefolder = os.path.join(outdir_images_refined, trialname, f'cam{cam}')
                create_video(
                    image_folder=imagefolder,
                    extension='.jpg',
                    fps=fps,
                    output_folder=outdir_video_trialfolder,
                    video_name=f'cam{cam}_refined.mp4'
                )