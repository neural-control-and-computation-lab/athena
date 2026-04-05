"""Dynamic camera calibration refinement via bundle adjustment.

Optimises camera extrinsic parameters (rotation + translation) using
reprojection error minimisation.  Supports sliding-window processing
for long recordings and drift detection to flag cameras that have
physically moved.

The approach is inspired by SynthMoCap-style multi-view optimisation:
given an initial calibration and confident 2D landmark detections, we
jointly refine camera extrinsics and 3D point positions to minimise
reprojection loss.
"""

import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rodrigues_to_matrix(rvec):
    """Convert a 3-element Rodrigues (axis-angle) vector to a 3x3 rotation matrix."""
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).ravel())
    return R


def matrix_to_rodrigues(R):
    """Convert a 3x3 rotation matrix to a 3-element Rodrigues vector."""
    rvec, _ = cv.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.ravel()


# ---------------------------------------------------------------------------
# Packing / unpacking camera extrinsics
# ---------------------------------------------------------------------------

def _pack_extrinsics(extrinsics, ref_cam=0):
    """Flatten non-reference camera [rvec, tvec] into a 1-D parameter vector.

    Parameters
    ----------
    extrinsics : np.ndarray, shape (ncams, 4, 4)
        Homogeneous extrinsic matrices.
    ref_cam : int
        Index of the fixed reference camera (excluded from optimisation).

    Returns
    -------
    params : np.ndarray, shape (6*(ncams-1),)
    """
    params = []
    for cam in range(len(extrinsics)):
        if cam == ref_cam:
            continue
        R = extrinsics[cam][:3, :3]
        t = extrinsics[cam][:3, 3]
        params.extend(matrix_to_rodrigues(R).tolist())
        params.extend(t.tolist())
    return np.array(params, dtype=np.float64)


def _unpack_extrinsics(cam_params, ref_extrinsic, ncams, ref_cam=0):
    """Reconstruct (ncams, 4, 4) extrinsic matrices from the parameter vector.

    Parameters
    ----------
    cam_params : np.ndarray, shape (6*(ncams-1),)
    ref_extrinsic : np.ndarray, shape (4, 4)
        The fixed reference camera extrinsic.
    ncams : int
    ref_cam : int

    Returns
    -------
    extrinsics : np.ndarray, shape (ncams, 4, 4)
    """
    extrinsics = np.zeros((ncams, 4, 4))
    idx = 0
    for cam in range(ncams):
        if cam == ref_cam:
            extrinsics[cam] = ref_extrinsic
        else:
            rvec = cam_params[idx:idx + 3]
            tvec = cam_params[idx + 3:idx + 6]
            R = rodrigues_to_matrix(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec
            extrinsics[cam] = T
            idx += 6
    return extrinsics


# ---------------------------------------------------------------------------
# Observation selection
# ---------------------------------------------------------------------------

def _select_observations(data_2d, ncams, nframes, nlandmarks,
                         landmark_indices=None, subsample=10,
                         min_cameras=3, frame_start=0, frame_end=None):
    """Select high-confidence 2D observations for optimisation.

    Parameters
    ----------
    data_2d : np.ndarray, shape (ncams, nframes, nlandmarks, 2)
        Pixel-space 2D landmarks.  NaN or -1 where missing.
    ncams, nframes, nlandmarks : int
    landmark_indices : array-like or None
        Which landmark indices to use.  Defaults to hand landmarks 33-74.
    subsample : int
        Use every *subsample*-th frame.
    min_cameras : int
        Require a landmark visible in at least this many cameras.
    frame_start, frame_end : int or None
        Temporal window bounds.

    Returns
    -------
    observations : list of (cam_idx, point_idx, u, v)
        point_idx is a unique identifier for (frame, landmark).
    point_keys : list of (frame, landmark)
        Maps point_idx -> (frame, landmark) tuple.
    """
    if landmark_indices is None:
        landmark_indices = list(range(33, 75))  # hand landmarks
    if frame_end is None:
        frame_end = nframes

    # Build mapping from (frame, landmark) -> point_idx
    point_keys = []
    point_map = {}
    observations = []

    frames = range(frame_start, frame_end, subsample)
    for f in frames:
        for lm in landmark_indices:
            # Count cameras where this landmark is visible
            visible_cams = []
            for cam in range(ncams):
                u, v = data_2d[cam, f, lm]
                if not (np.isnan(u) or np.isnan(v) or u == -1 or v == -1):
                    visible_cams.append((cam, u, v))

            if len(visible_cams) < min_cameras:
                continue

            # Assign a point index
            key = (f, lm)
            if key not in point_map:
                point_map[key] = len(point_keys)
                point_keys.append(key)
            pt_idx = point_map[key]

            for cam, u, v in visible_cams:
                observations.append((cam, pt_idx, u, v))

    return observations, point_keys


# ---------------------------------------------------------------------------
# Sparse Jacobian structure
# ---------------------------------------------------------------------------

def _build_sparsity(n_cam_params, n_points, observations, ncams, ref_cam=0):
    """Build the sparse Jacobian structure for bundle adjustment.

    Each reprojection residual (2 rows) depends on:
    - 6 camera parameters (if not the reference camera)
    - 3 point parameters

    Additional regularisation residuals depend on camera parameters only.

    Parameters
    ----------
    n_cam_params : int
        Total camera parameters (6*(ncams-1)).
    n_points : int
        Number of 3D points being optimised.
    observations : list of (cam_idx, point_idx, u, v)
    ncams : int
    ref_cam : int

    Returns
    -------
    scipy.sparse.lil_matrix
    """
    n_point_params = 3 * n_points
    n_params = n_cam_params + n_point_params
    n_reproj_residuals = 2 * len(observations)
    n_reg_residuals = n_cam_params
    n_residuals = n_reproj_residuals + n_reg_residuals

    A = lil_matrix((n_residuals, n_params), dtype=int)

    # Map camera index -> parameter offset (ref_cam has no params)
    cam_param_offset = {}
    idx = 0
    for cam in range(ncams):
        if cam == ref_cam:
            continue
        cam_param_offset[cam] = idx
        idx += 6

    # Reprojection residuals
    for i, (cam, pt_idx, _, _) in enumerate(observations):
        row = 2 * i
        # Camera parameters
        if cam in cam_param_offset:
            col = cam_param_offset[cam]
            A[row, col:col + 6] = 1
            A[row + 1, col:col + 6] = 1
        # Point parameters
        pt_col = n_cam_params + 3 * pt_idx
        A[row, pt_col:pt_col + 3] = 1
        A[row + 1, pt_col:pt_col + 3] = 1

    # Regularisation residuals (one per camera param)
    for i in range(n_cam_params):
        A[n_reproj_residuals + i, i] = 1

    return A


# ---------------------------------------------------------------------------
# Bundle adjustment residual function
# ---------------------------------------------------------------------------

def _bundle_residuals(params, intrinsics, ref_extrinsic, ncams, n_points,
                      observations, reg_weight, initial_cam_params, ref_cam=0):
    """Compute the residual vector for bundle adjustment.

    Parameters
    ----------
    params : np.ndarray
        Concatenation of [cam_params, point_params].
    intrinsics : list of np.ndarray (3x3)
    ref_extrinsic : np.ndarray (4, 4)
    ncams : int
    n_points : int
    observations : list of (cam_idx, point_idx, u, v)
    reg_weight : float
    initial_cam_params : np.ndarray
    ref_cam : int

    Returns
    -------
    residuals : np.ndarray
    """
    n_cam_params = 6 * (ncams - 1)
    cam_params = params[:n_cam_params]
    point_params = params[n_cam_params:]

    # Reconstruct extrinsics and 3D points
    extrinsics = _unpack_extrinsics(cam_params, ref_extrinsic, ncams, ref_cam)
    points_3d = point_params.reshape(n_points, 3)

    # Precompute projection matrices: P = K @ E[:3]
    proj_mats = np.zeros((ncams, 3, 4))
    for cam in range(ncams):
        K = np.asarray(intrinsics[cam])
        E = extrinsics[cam][:3]  # (3, 4)
        proj_mats[cam] = K @ E

    # Compute reprojection residuals
    residuals = np.empty(2 * len(observations) + n_cam_params)

    for i, (cam, pt_idx, u_obs, v_obs) in enumerate(observations):
        X = np.append(points_3d[pt_idx], 1.0)  # homogeneous
        proj = proj_mats[cam] @ X
        z = proj[2]
        if abs(z) < 1e-12:
            z = 1e-12
        u_proj = proj[0] / z
        v_proj = proj[1] / z
        residuals[2 * i] = u_proj - u_obs
        residuals[2 * i + 1] = v_proj - v_obs

    # Regularisation: penalise drift from initial camera parameters
    residuals[2 * len(observations):] = np.sqrt(reg_weight) * (cam_params - initial_cam_params)

    return residuals


# ---------------------------------------------------------------------------
# Initial 3D point estimates via DLT
# ---------------------------------------------------------------------------

def _initial_triangulation(observations, point_keys, n_points,
                           intrinsics, extrinsics, ncams):
    """Get initial 3D point estimates by DLT for each unique point.

    Groups observations by point_idx, builds a DLT system, and solves
    via SVD for each point.

    Returns
    -------
    points_3d : np.ndarray, shape (n_points, 3)
    """
    # Group observations by point
    obs_by_point = {}
    for cam, pt_idx, u, v in observations:
        obs_by_point.setdefault(pt_idx, []).append((cam, u, v))

    # Precompute projection matrices
    proj_mats = np.zeros((ncams, 3, 4))
    for cam in range(ncams):
        K = np.asarray(intrinsics[cam])
        E = extrinsics[cam][:3]
        proj_mats[cam] = K @ E

    points_3d = np.zeros((n_points, 3))
    for pt_idx in range(n_points):
        obs = obs_by_point.get(pt_idx, [])
        if len(obs) < 2:
            points_3d[pt_idx] = np.nan
            continue

        # Undistort pixel observations to normalised coords for DLT
        # Since images are already undistorted, we just need to go from
        # pixel to normalised coordinates using K^-1
        n_obs = len(obs)
        A = np.zeros((2 * n_obs, 4))
        for j, (cam, u, v) in enumerate(obs):
            # Use projection matrix directly (pixel-space DLT)
            P = proj_mats[cam]
            A[2 * j] = u * P[2] - P[0]
            A[2 * j + 1] = v * P[2] - P[1]

        _, _, vh = np.linalg.svd(A)
        X = vh[-1]
        if abs(X[3]) < 1e-12:
            points_3d[pt_idx] = np.nan
        else:
            points_3d[pt_idx] = X[:3] / X[3]

    return points_3d


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def _compute_drift(extrinsics_prev, extrinsics_curr, ref_cam=0):
    """Compute per-camera rotation and translation drift between two extrinsic sets.

    Parameters
    ----------
    extrinsics_prev, extrinsics_curr : np.ndarray, shape (ncams, 4, 4)
    ref_cam : int

    Returns
    -------
    drift : list of dict
        Per-camera dict with keys 'rotation_deg' and 'translation_mm'.
    """
    ncams = len(extrinsics_prev)
    drift = []
    for cam in range(ncams):
        if cam == ref_cam:
            drift.append({'rotation_deg': 0.0, 'translation_mm': 0.0})
            continue

        R_prev = extrinsics_prev[cam][:3, :3]
        R_curr = extrinsics_curr[cam][:3, :3]
        t_prev = extrinsics_prev[cam][:3, 3]
        t_curr = extrinsics_curr[cam][:3, 3]

        # Rotation angle via relative rotation
        R_rel = R_curr @ R_prev.T
        # Angle from trace: trace(R) = 1 + 2*cos(theta)
        cos_angle = (np.trace(R_rel) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Translation distance
        t_dist = np.linalg.norm(t_curr - t_prev)

        drift.append({'rotation_deg': angle_deg, 'translation_mm': t_dist})

    return drift


# ---------------------------------------------------------------------------
# Single-window bundle adjustment
# ---------------------------------------------------------------------------

def _refine_window(extrinsics, intrinsics, data_2d, ncams, nframes, nlandmarks,
                   landmark_indices=None, subsample=10, min_cameras=3,
                   frame_start=0, frame_end=None, reg_weight=0.01,
                   ref_cam=0, max_iterations=50, verbose=False):
    """Run bundle adjustment on a single temporal window.

    Parameters
    ----------
    extrinsics : np.ndarray, shape (ncams, 4, 4)
    intrinsics : list of np.ndarray
    data_2d : np.ndarray, shape (ncams, nframes, nlandmarks, 2)
    ncams, nframes, nlandmarks : int
    landmark_indices : list or None
    subsample : int
    min_cameras : int
    frame_start, frame_end : int or None
    reg_weight : float
    ref_cam : int
    max_iterations : int
    verbose : bool

    Returns
    -------
    refined_extrinsics : np.ndarray, shape (ncams, 4, 4)
    mean_reproj_error_before : float
    mean_reproj_error_after : float
    n_observations : int
    """
    # Select observations
    observations, point_keys = _select_observations(
        data_2d, ncams, nframes, nlandmarks,
        landmark_indices=landmark_indices,
        subsample=subsample,
        min_cameras=min_cameras,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    if len(observations) < 20:
        if verbose:
            print(f'    Window [{frame_start}:{frame_end}]: too few observations '
                  f'({len(observations)}), skipping refinement.')
        return extrinsics.copy(), np.nan, np.nan, len(observations)

    n_points = len(point_keys)
    n_cam_params = 6 * (ncams - 1)

    # Initial camera parameters
    initial_cam_params = _pack_extrinsics(extrinsics, ref_cam)

    # Initial 3D point estimates via DLT
    points_3d_init = _initial_triangulation(
        observations, point_keys, n_points, intrinsics, extrinsics, ncams
    )

    # Remove points that failed triangulation
    valid_pts = ~np.isnan(points_3d_init[:, 0])
    if np.sum(valid_pts) < 5:
        if verbose:
            print(f'    Window [{frame_start}:{frame_end}]: too few valid 3D points, skipping.')
        return extrinsics.copy(), np.nan, np.nan, len(observations)

    # Filter observations to only valid points
    valid_set = set(np.where(valid_pts)[0])
    observations_filtered = [(c, p, u, v) for c, p, u, v in observations if p in valid_set]

    # Remap point indices to be contiguous
    old_to_new = {}
    new_points = []
    for c, p, u, v in observations_filtered:
        if p not in old_to_new:
            old_to_new[p] = len(new_points)
            new_points.append(points_3d_init[p])
    observations_remapped = [(c, old_to_new[p], u, v) for c, p, u, v in observations_filtered]
    points_3d_init_clean = np.array(new_points)
    n_points_clean = len(new_points)

    if n_points_clean < 5:
        if verbose:
            print(f'    Window [{frame_start}:{frame_end}]: too few valid 3D points after filtering, skipping.')
        return extrinsics.copy(), np.nan, np.nan, len(observations)

    # Build parameter vector: [cam_params, point_params]
    x0 = np.concatenate([initial_cam_params, points_3d_init_clean.ravel()])

    # Build sparse Jacobian structure
    sparsity = _build_sparsity(n_cam_params, n_points_clean, observations_remapped,
                               ncams, ref_cam)

    # Compute initial reprojection error
    r0 = _bundle_residuals(x0, intrinsics, extrinsics[ref_cam], ncams, n_points_clean,
                           observations_remapped, reg_weight, initial_cam_params, ref_cam)
    n_reproj = 2 * len(observations_remapped)
    reproj_residuals_before = r0[:n_reproj]
    mean_err_before = np.sqrt(np.mean(reproj_residuals_before ** 2))

    # Run optimisation with robust (soft L1) loss to downweight outlier
    # detections.  This prevents a few bad 2D landmarks from dragging
    # camera extrinsics away from their true position, while still
    # allowing the optimiser to follow genuine camera movement that
    # produces consistently shifted residuals across many landmarks.
    result = least_squares(
        _bundle_residuals,
        x0,
        jac_sparsity=sparsity,
        method='trf',
        loss='soft_l1',
        f_scale=5.0,
        args=(intrinsics, extrinsics[ref_cam], ncams, n_points_clean,
              observations_remapped, reg_weight, initial_cam_params, ref_cam),
        max_nfev=max_iterations,
        verbose=0,
    )

    # Compute final reprojection error
    reproj_residuals_after = result.fun[:n_reproj]
    mean_err_after = np.sqrt(np.mean(reproj_residuals_after ** 2))

    # Extract refined extrinsics
    refined_cam_params = result.x[:n_cam_params]
    refined_extrinsics = _unpack_extrinsics(
        refined_cam_params, extrinsics[ref_cam], ncams, ref_cam
    )

    # Safety: discard if reprojection error increased
    if mean_err_after > mean_err_before:
        if verbose:
            print(f'    Window [{frame_start}:{frame_end}]: optimisation increased error '
                  f'({mean_err_before:.2f} -> {mean_err_after:.2f} px), discarding.')
        return extrinsics.copy(), mean_err_before, mean_err_after, len(observations_remapped)

    if verbose:
        print(f'    Window [{frame_start}:{frame_end}]: reproj error '
              f'{mean_err_before:.2f} -> {mean_err_after:.2f} px '
              f'({len(observations_remapped)} obs, {n_points_clean} pts)')

    return refined_extrinsics, mean_err_before, mean_err_after, len(observations_remapped)


# ---------------------------------------------------------------------------
# Public API: sliding-window calibration refinement
# ---------------------------------------------------------------------------

def refine_calibration(extrinsics, intrinsics, data_2d, nframes, nlandmarks,
                       window_size=500, window_overlap=100, subsample=10,
                       min_cameras=3, reg_weight=0.01, ref_cam=0,
                       max_iterations=50, drift_rotation_thresh=2.0,
                       drift_translation_thresh=20.0, verbose=True):
    """Refine camera extrinsics using sliding-window bundle adjustment.

    Parameters
    ----------
    extrinsics : np.ndarray, shape (ncams, 4, 4)
        Initial extrinsic matrices.
    intrinsics : list of np.ndarray
        Per-camera 3x3 intrinsic matrices.
    data_2d : np.ndarray, shape (ncams, nframes, nlandmarks, 2)
        Pixel-space 2D landmarks.  -1 or NaN where missing.
    nframes, nlandmarks : int
    window_size : int
        Number of frames per optimisation window.
    window_overlap : int
        Overlap between consecutive windows.
    subsample : int
        Use every *subsample*-th frame within each window.
    min_cameras : int
        Minimum cameras a landmark must be visible in.
    reg_weight : float
        Regularisation weight penalising drift from initial calibration.
    ref_cam : int
        Index of the fixed reference camera.
    max_iterations : int
        Maximum function evaluations per window.
    drift_rotation_thresh : float
        Flag drift if any camera rotates more than this (degrees).
    drift_translation_thresh : float
        Flag drift if any camera translates more than this (mm).
    verbose : bool

    Returns
    -------
    refined_extrinsics : np.ndarray, shape (ncams, 4, 4)
        Best refined extrinsics (from the last successfully optimised window).
    drift_report : list of dict
        Per-window drift information.
    """
    ncams = extrinsics.shape[0]

    # Replace -1 sentinels with NaN for consistent handling
    data_2d = data_2d.copy()
    sentinel = (data_2d[:, :, :, 0] == -1) & (data_2d[:, :, :, 1] == -1)
    data_2d[sentinel] = np.nan

    # Hand landmark indices (right: 33-53, left: 54-74)
    landmark_indices = list(range(33, 75))
    # Only use landmarks that exist in this dataset
    landmark_indices = [l for l in landmark_indices if l < nlandmarks]

    if not landmark_indices:
        if verbose:
            print('  Calibration refinement: no hand landmarks available, skipping.')
        return extrinsics.copy(), []

    # Compute window boundaries
    step = window_size - window_overlap
    if step < 1:
        step = 1
    windows = []
    start = 0
    while start < nframes:
        end = min(start + window_size, nframes)
        windows.append((start, end))
        if end >= nframes:
            break
        start += step

    # If recording is shorter than one window, just use the whole thing
    if not windows:
        windows = [(0, nframes)]

    if verbose:
        print(f'  Calibration refinement: {len(windows)} window(s), '
              f'{ncams} cameras, ref_cam={ref_cam}')

    drift_report = []
    initial_extrinsics = extrinsics.copy()

    # Each window independently refines from the original calibration.
    # This prevents drift accumulation across windows and allows drift
    # detection to identify genuine camera movement.
    window_results = []  # (refined_extrinsics, err_after, n_obs) per window

    for win_idx, (f_start, f_end) in enumerate(windows):
        refined, err_before, err_after, n_obs = _refine_window(
            initial_extrinsics, intrinsics, data_2d, ncams, nframes, nlandmarks,
            landmark_indices=landmark_indices,
            subsample=subsample,
            min_cameras=min_cameras,
            frame_start=f_start,
            frame_end=f_end,
            reg_weight=reg_weight,
            ref_cam=ref_cam,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        # Compute drift from initial calibration
        drift = _compute_drift(initial_extrinsics, refined, ref_cam)

        # Check for large drift (potential camera bump)
        flagged_cameras = []
        for cam, d in enumerate(drift):
            if (d['rotation_deg'] > drift_rotation_thresh or
                    d['translation_mm'] > drift_translation_thresh):
                flagged_cameras.append(cam)

        window_report = {
            'window': (f_start, f_end),
            'reproj_error_before': err_before,
            'reproj_error_after': err_after,
            'n_observations': n_obs,
            'drift': drift,
            'flagged_cameras': flagged_cameras,
        }
        drift_report.append(window_report)

        if flagged_cameras and verbose:
            for cam in flagged_cameras:
                d = drift[cam]
                print(f'    WARNING: Camera {cam} drift detected in window '
                      f'[{f_start}:{f_end}] -- '
                      f'rotation={d["rotation_deg"]:.2f} deg, '
                      f'translation={d["translation_mm"]:.1f} mm')

        # Track successful window results
        if not np.isnan(err_after) and err_after <= err_before:
            window_results.append((refined, err_after, n_obs))

    # Combine results: use the weighted average of all successful windows'
    # camera parameters, weighted by number of observations.  This gives
    # a robust consensus estimate across the full recording.
    if window_results:
        total_weight = sum(n for _, _, n in window_results)
        avg_cam_params = np.zeros(6 * (ncams - 1))
        for refined_ext, _, n_obs in window_results:
            w = n_obs / total_weight
            avg_cam_params += w * _pack_extrinsics(refined_ext, ref_cam)
        best_extrinsics = _unpack_extrinsics(
            avg_cam_params, initial_extrinsics[ref_cam], ncams, ref_cam
        )
    else:
        best_extrinsics = initial_extrinsics.copy()

    if verbose:
        # Summary
        successful = [r for r in drift_report if not np.isnan(r['reproj_error_after'])
                      and r['reproj_error_after'] <= r['reproj_error_before']]
        if successful:
            avg_before = np.mean([r['reproj_error_before'] for r in successful])
            avg_after = np.mean([r['reproj_error_after'] for r in successful])
            print(f'  Calibration refinement complete: '
                  f'avg reproj error {avg_before:.2f} -> {avg_after:.2f} px '
                  f'({len(successful)}/{len(drift_report)} windows improved)')
        else:
            print('  Calibration refinement: no windows improved.')

    return best_extrinsics, drift_report


# ---------------------------------------------------------------------------
# Save refined calibration
# ---------------------------------------------------------------------------

def save_refined_calibration(extrinsics, intrinsics, dist_coeffs, output_path,
                             fmt='toml', metadata=None):
    """Write refined calibration to a new file (never overwrites originals).

    Parameters
    ----------
    extrinsics : np.ndarray, shape (ncams, 4, 4)
    intrinsics : list of np.ndarray
    dist_coeffs : list of np.ndarray
    output_path : str
    fmt : str
        'toml' or 'yaml'.
    metadata : dict or None
        Optional metadata to include (e.g. reprojection error, timestamp).
    """
    ncams = len(intrinsics)

    if fmt == 'toml':
        import toml
        cal = {}
        cal['metadata'] = metadata or {}
        cal['metadata']['refined'] = True

        for cam in range(ncams):
            camname = f'cam_{cam}'
            R = extrinsics[cam][:3, :3]
            t = extrinsics[cam][:3, 3]
            rvec = matrix_to_rodrigues(R)

            cal[camname] = {
                'matrix': np.asarray(intrinsics[cam]).tolist(),
                'distortions': np.asarray(dist_coeffs[cam]).tolist(),
                'rotation': rvec.tolist(),
                'translation': t.tolist(),
            }

        with open(output_path, 'w') as f:
            toml.dump(cal, f)

    elif fmt == 'yaml':
        for cam in range(ncams):
            cam_path = output_path.replace('.yaml', f'_cam{cam}.yaml')
            fs = cv.FileStorage(cam_path, cv.FILE_STORAGE_WRITE)
            fs.write('intrinsicMatrix', np.asarray(intrinsics[cam]).T)
            fs.write('distortionCoefficients', np.asarray(dist_coeffs[cam]).reshape(-1, 1))
            fs.write('R', extrinsics[cam][:3, :3].T)
            fs.write('T', extrinsics[cam][:3, 3].reshape(3, 1))
            fs.release()
