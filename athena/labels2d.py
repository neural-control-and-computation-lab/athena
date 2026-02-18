import os
import glob
import json
import time
import threading
import tkinter as tk
import tkinter.ttk as ttk  # For the progress bar
import toml
import av
import cv2 as cv
import numpy as np
import mediapipe as mp
import concurrent.futures
from athena.visualization import (draw_landmarks_unified, render_mesh_overlay,
                                  SKELETON_LINKS, SKELETON_COLOURS_BGR,
                                  SKELETON_COLOURS_HEX, FACE_COLOUR_BGR,
                                  MESH_RIGHT_COLOUR_BGR, MESH_LEFT_COLOUR_BGR,
                                  hex_to_bgr)
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    PoseLandmarker,
    HandLandmarkerOptions,
    PoseLandmarkerOptions,
    RunningMode
)
from multiprocessing import Manager, set_start_method

models_dir = os.path.join(os.path.dirname(__file__), "models")
hand_model_path = os.path.join(models_dir, "hand_landmarker.task")
pose_model_path = os.path.join(models_dir, "pose_landmarker_full.task")
face_model_path = os.path.join(models_dir, "face_landmarker.task")
face_detector_model_path = os.path.join(models_dir, "blaze_face_short_range.tflite")

# Body-pose head landmark indices (MediaPipe Pose: 0=nose, 1-6=eyes, 7-8=ears, 9-10=mouth)
_HEAD_INDICES = list(range(11))


# ---------------------------------------------------------------------------
# Triangulation-guided hand reassignment (second pass)
# ---------------------------------------------------------------------------

def _reassign_hands_from_3d_wrists(cam_mats_intrinsic, cam_mats_extrinsic,
                                    outdir_data2d_trial, reject_threshold=60.0):
    """Reassign and validate hand labels using triangulated 3D body wrists.

    After all cameras have been processed independently, this function:
      1. Loads body landmarks from every camera.
      2. Triangulates body wrists (landmarks 15=left, 16=right) to 3D.
      3. Reprojects the 3D wrists into every camera's pixel space.
      4. Reassigns each camera's hand landmarks to the nearest reprojected
         wrist, fixing misassignments from cameras that lacked body detections.
      5. Rejects (sets to sentinel -1) hand detections whose wrist root is
         further than ``reject_threshold`` pixels from the corresponding
         reprojected 3D wrist — these are spatially inconsistent and would
         corrupt the downstream triangulation.
      6. Re-saves corrected hand (and vertex) .npy files.

    Parameters
    ----------
    cam_mats_intrinsic : list of np.ndarray
        ``(3, 3)`` intrinsic matrix per camera.
    cam_mats_extrinsic : list or np.ndarray
        ``(3or4, 4)`` extrinsic matrix per camera.
    outdir_data2d_trial : str
        Directory containing per-camera landmark subdirectories (cam0/, cam1/, ...).
    reject_threshold : float
        Maximum pixel distance from hand root to reprojected 3D wrist.
        Hands exceeding this are rejected (set to sentinel).
    """
    cam_mats_extrinsic = np.asarray(cam_mats_extrinsic)
    cam_dirs = sorted(glob.glob(os.path.join(outdir_data2d_trial, 'cam*')))
    ncams = len(cam_dirs)
    if ncams < 2:
        return

    # ── 1. Load body landmarks from all cameras ──────────────────────
    body_all = []
    for cam_dir in cam_dirs:
        body_file = os.path.join(cam_dir, '2Dlandmarks_body.npy')
        body_all.append(np.load(body_file).astype(float))
    nframes = body_all[0].shape[0]

    # ── 2. Triangulate wrists to 3D ─────────────────────────────────
    #    landmark 15 = left wrist,  landmark 16 = right wrist
    wrist_2d = {}  # {side: (ncams, nframes, 2)}
    for side, lm_idx in [('left', 15), ('right', 16)]:
        pts = np.stack([b[:, lm_idx, :2] for b in body_all])  # (ncams, nframes, 2)
        sentinel = (pts[:, :, 0] == -1) & (pts[:, :, 1] == -1)
        pts[sentinel] = np.nan
        wrist_2d[side] = pts

    # Undistort
    wrist_undist = {}
    for side in ('left', 'right'):
        pts = wrist_2d[side].copy()
        for cam in range(ncams):
            nan_mask = np.isnan(pts[cam, :, 0])
            if (~nan_mask).any():
                valid = pts[cam, ~nan_mask].reshape(-1, 1, 2)
                undist = cv.undistortPoints(valid, cam_mats_intrinsic[cam],
                                            np.zeros(5), P=cam_mats_intrinsic[cam])
                pts[cam, ~nan_mask] = undist.reshape(-1, 2)
        wrist_undist[side] = pts

    # Per-frame DLT triangulation for each wrist
    wrist_3d = {}
    for side in ('left', 'right'):
        pts_u = wrist_undist[side]  # (ncams, nframes, 2)
        w3d = np.full((nframes, 3), np.nan)
        for f in range(nframes):
            visible = ~np.isnan(pts_u[:, f, 0])
            if visible.sum() < 2:
                continue
            cams_vis = np.where(visible)[0]
            A = np.zeros((2 * len(cams_vis), 4))
            for i, c in enumerate(cams_vis):
                P = cam_mats_intrinsic[c] @ cam_mats_extrinsic[c][:3]
                x, y = pts_u[c, f]
                A[2 * i] = x * P[2] - P[0]
                A[2 * i + 1] = y * P[2] - P[1]
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            w3d[f] = X[:3] / X[3]
        wrist_3d[side] = w3d

    valid_l = ~np.isnan(wrist_3d['left'][:, 0])
    valid_r = ~np.isnan(wrist_3d['right'][:, 0])
    valid_both = valid_l & valid_r
    print(f'  Wrist triangulation: left={valid_l.sum()}, right={valid_r.sum()}, '
          f'both={valid_both.sum()}/{nframes} frames')

    if valid_both.sum() == 0:
        print('  Skipping hand reassignment (no frames with both wrists triangulated).')
        return

    # ── 3. Reproject 3D wrists to every camera ──────────────────────
    wrist_reproj = {}  # {side: (ncams, nframes, 2)}
    for side in ('left', 'right'):
        w3d = wrist_3d[side]
        reproj = np.full((ncams, nframes, 2), np.nan)
        valid = ~np.isnan(w3d[:, 0])
        if valid.any():
            pts_h = np.hstack([w3d[valid], np.ones((valid.sum(), 1))])
            for cam in range(ncams):
                ext = cam_mats_extrinsic[cam][:3]
                intr = cam_mats_intrinsic[cam]
                img_h = intr @ (ext @ pts_h.T)
                uv = (img_h[:2] / img_h[2:3]).T
                reproj[cam, valid] = uv
        wrist_reproj[side] = reproj

    # ── 4. Load hand landmarks, reassign, and reject outliers ────────
    sentinel_val = np.float32(-1)
    total_swaps = 0
    total_rejects = 0
    for cam_idx, cam_dir in enumerate(cam_dirs):
        right_file = os.path.join(cam_dir, '2Dlandmarks_right.npy')
        left_file = os.path.join(cam_dir, '2Dlandmarks_left.npy')
        right_data = np.load(right_file).astype(float)
        left_data = np.load(left_file).astype(float)

        # Optional HaMeR vertex data
        v2d_r_file = os.path.join(cam_dir, 'hamer_vertices_2d_right.npy')
        v2d_l_file = os.path.join(cam_dir, 'hamer_vertices_2d_left.npy')
        v3d_r_file = os.path.join(cam_dir, 'hamer_vertices_right.npy')
        v3d_l_file = os.path.join(cam_dir, 'hamer_vertices_left.npy')
        has_verts = os.path.exists(v2d_r_file) and os.path.exists(v2d_l_file)
        if has_verts:
            v2d_r = np.load(v2d_r_file).astype(float)
            v2d_l = np.load(v2d_l_file).astype(float)
            v3d_r = np.load(v3d_r_file).astype(float)
            v3d_l = np.load(v3d_l_file).astype(float)

        cam_swaps = 0
        cam_rejects = 0
        cam_modified = False
        for f in range(nframes):
            lw = wrist_reproj['left'][cam_idx, f]   # reprojected left wrist
            rw = wrist_reproj['right'][cam_idx, f]   # reprojected right wrist
            if np.isnan(lw[0]) or np.isnan(rw[0]):
                continue  # need both wrists to decide

            # Hand root positions (landmark 0 = wrist of each hand)
            r_root = right_data[f, 0, :2]
            l_root = left_data[f, 0, :2]
            r_valid = r_root[0] != -1 or r_root[1] != -1
            l_valid = l_root[0] != -1 or l_root[1] != -1

            if not r_valid and not l_valid:
                continue

            # ── Step A: Reassign (swap) if hands are closer to wrong wrists ──
            need_swap = False
            if r_valid and l_valid:
                cost_correct = (np.linalg.norm(r_root - rw) +
                                np.linalg.norm(l_root - lw))
                cost_swapped = (np.linalg.norm(r_root - lw) +
                                np.linalg.norm(l_root - rw))
                need_swap = cost_swapped < cost_correct
            elif r_valid:
                if np.linalg.norm(r_root - lw) < np.linalg.norm(r_root - rw):
                    need_swap = True
            elif l_valid:
                if np.linalg.norm(l_root - rw) < np.linalg.norm(l_root - lw):
                    need_swap = True

            if need_swap:
                right_data[f], left_data[f] = left_data[f].copy(), right_data[f].copy()
                if has_verts:
                    v2d_r[f], v2d_l[f] = v2d_l[f].copy(), v2d_r[f].copy()
                    v3d_r[f], v3d_l[f] = v3d_l[f].copy(), v3d_r[f].copy()
                cam_swaps += 1
                # Re-read roots after swap
                r_root = right_data[f, 0, :2]
                l_root = left_data[f, 0, :2]
                r_valid = r_root[0] != -1 or r_root[1] != -1
                l_valid = l_root[0] != -1 or l_root[1] != -1

            # ── Step B: Reject hands too far from reprojected wrist ──
            if r_valid and np.linalg.norm(r_root - rw) > reject_threshold:
                right_data[f] = sentinel_val
                if has_verts:
                    v2d_r[f] = sentinel_val
                    v3d_r[f] = sentinel_val
                cam_rejects += 1

            if l_valid and np.linalg.norm(l_root - lw) > reject_threshold:
                left_data[f] = sentinel_val
                if has_verts:
                    v2d_l[f] = sentinel_val
                    v3d_l[f] = sentinel_val
                cam_rejects += 1

        cam_modified = cam_swaps > 0 or cam_rejects > 0
        if cam_modified:
            np.save(right_file, right_data.astype(np.float32))
            np.save(left_file, left_data.astype(np.float32))
            if has_verts:
                np.save(v2d_r_file, v2d_r.astype(np.float32))
                np.save(v2d_l_file, v2d_l.astype(np.float32))
                np.save(v3d_r_file, v3d_r.astype(np.float32))
                np.save(v3d_l_file, v3d_l.astype(np.float32))

        total_swaps += cam_swaps
        total_rejects += cam_rejects
        if cam_swaps > 0 or cam_rejects > 0:
            print(f'  cam{cam_idx}: {cam_swaps} swaps, {cam_rejects} rejected '
                  f'(>{reject_threshold:.0f}px from 3D wrist)')

    print(f'  Hand reassignment complete: {total_swaps} swaps, '
          f'{total_rejects} rejections across {ncams} cameras.')


def _redraw_saved_images(input_streams, outdir_images_trial, outdir_data2d_trial,
                         cam_mats_intrinsic, cam_dist_coeffs,
                         display_width=450, display_height=360,
                         hand_backend='mediapipe', use_gpu=False):
    """Re-render saved PNG images using corrected (post-reassignment) landmark data.

    During per-camera processing, PNG images are drawn with the *original*
    hand labels.  After ``_reassign_hands_from_3d_wrists`` modifies the
    ``.npy`` files on disk, the PNG images are stale.  This function re-reads
    each video frame, undistorts it, overlays the *corrected* landmarks using
    ``draw_landmarks_unified``, and overwrites the PNGs so that videos built
    from them accurately reflect the reassignment.

    Parameters
    ----------
    input_streams : list of str
        Video file paths, one per camera.
    outdir_images_trial : str
        Root image output directory (contains cam0/, cam1/, … sub-dirs).
    outdir_data2d_trial : str
        Root landmark data directory (contains cam0/, cam1/, … sub-dirs).
    cam_mats_intrinsic : list of np.ndarray
        ``(3, 3)`` intrinsic matrix per camera.
    cam_dist_coeffs : list of np.ndarray
        Distortion coefficient array per camera.
    display_width, display_height : int
        Size to which saved frames are resized.
    hand_backend : str
        ``'hamer'`` or ``'mediapipe'``.  When ``'hamer'`` and vertex data
        exists, mesh overlays are rendered instead of skeleton hand lines.
    use_gpu : bool
        Whether to decode RGBA frames (GPU delegate).
    """
    ncams = len(input_streams)

    # Precompute undistortion maps
    undistort_maps = []
    for cam in range(ncams):
        cont = av.open(input_streams[cam])
        for frame in cont.decode(video=0):
            arr = frame.to_ndarray(format='rgb24')
            fh, fw = arr.shape[:2]
            m1, m2 = cv.initUndistortRectifyMap(
                cam_mats_intrinsic[cam], cam_dist_coeffs[cam],
                None, cam_mats_intrinsic[cam], (fw, fh), cv.CV_16SC2)
            undistort_maps.append((m1, m2))
            break
        cont.close()

    # Load MANO faces once if HaMeR vertex data is available
    mano_faces = None
    if hand_backend == 'hamer':
        faces_file = os.path.join(outdir_data2d_trial, 'hamer_faces.npy')
        if os.path.exists(faces_file):
            mano_faces = np.load(faces_file)

    for cam in range(ncams):
        img_dir = os.path.join(outdir_images_trial, f'cam{cam}')
        if not os.path.isdir(img_dir):
            continue
        # Check that PNGs exist for this camera
        existing_pngs = sorted(glob.glob(os.path.join(img_dir, 'frame*.png')))
        if not existing_pngs:
            continue

        cam_data_dir = os.path.join(outdir_data2d_trial, f'cam{cam}')
        body = np.load(os.path.join(cam_data_dir, '2Dlandmarks_body.npy'))
        right = np.load(os.path.join(cam_data_dir, '2Dlandmarks_right.npy'))
        left = np.load(os.path.join(cam_data_dir, '2Dlandmarks_left.npy'))

        face_file = os.path.join(cam_data_dir, '2Dlandmarks_face.npy')
        face = np.load(face_file) if os.path.exists(face_file) else None

        # HaMeR vertex data (optional)
        v2d_r = v2d_l = v3d_r = v3d_l = None
        if mano_faces is not None:
            f_v2d_r = os.path.join(cam_data_dir, 'hamer_vertices_2d_right.npy')
            f_v2d_l = os.path.join(cam_data_dir, 'hamer_vertices_2d_left.npy')
            f_v3d_r = os.path.join(cam_data_dir, 'hamer_vertices_right.npy')
            f_v3d_l = os.path.join(cam_data_dir, 'hamer_vertices_left.npy')
            if os.path.exists(f_v2d_r):
                v2d_r = np.load(f_v2d_r)
                v2d_l = np.load(f_v2d_l)
                v3d_r = np.load(f_v3d_r)
                v3d_l = np.load(f_v3d_l)

        nframes = body.shape[0]
        map1, map2 = undistort_maps[cam]

        container = av.open(input_streams[cam])
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        framenum = 0
        redrawn = 0
        for frame in container.decode(video=0):
            if framenum >= nframes:
                break

            save_path = os.path.join(img_dir, f'frame{framenum:06d}.png')
            if not os.path.exists(save_path):
                framenum += 1
                continue

            # Decode and undistort (same as during processing)
            if use_gpu:
                arr = frame.to_ndarray(format='rgba')
            else:
                arr = frame.to_ndarray(format='rgb24')
            arr = cv.remap(arr, map1, map2, cv.INTER_LINEAR)

            # Convert to BGR for drawing
            if use_gpu:
                bgr = cv.cvtColor(arr, cv.COLOR_RGBA2BGR)
            else:
                bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)

            # Build keypoint lists from corrected .npy data
            body_kpts = body[framenum].tolist()
            right_kpts = right[framenum].tolist()
            left_kpts = left[framenum].tolist()
            face_kpts = face[framenum].tolist() if face is not None else None

            # Build mesh dicts
            mesh_r = None
            mesh_l = None
            if mano_faces is not None and v2d_r is not None:
                if v2d_r[framenum, 0, 0] != -1:
                    mesh_r = {'verts_2d': v2d_r[framenum], 'verts_3d': v3d_r[framenum]}
                if v2d_l[framenum, 0, 0] != -1:
                    mesh_l = {'verts_2d': v2d_l[framenum], 'verts_3d': v3d_l[framenum]}

            draw_landmarks_unified(
                bgr, body_kpts, right_kpts, left_kpts,
                face_kpts=face_kpts,
                mesh_right=mesh_r, mesh_left=mesh_l, mesh_faces=mano_faces,
            )

            # Resize and save (same as during processing)
            resized = cv.resize(bgr, (display_width, display_height))
            cv.imwrite(save_path, resized)
            redrawn += 1
            framenum += 1

        container.close()
        if redrawn > 0:
            print(f'  cam{cam}: re-drew {redrawn} images with corrected hand labels')


# ---------------------------------------------------------------------------
# Triangulation-guided face detection (second pass)
# ---------------------------------------------------------------------------

def _triangulate_head(data_2d_body, cam_mats_intrinsic, cam_mats_extrinsic):
    """Triangulate head landmarks to 3D and reproject to each camera.

    Parameters
    ----------
    data_2d_body : list of np.ndarray
        Per-camera body landmarks, each ``(nframes, 33, 5)``.
    cam_mats_intrinsic : list of np.ndarray
        ``(3, 3)`` intrinsic per camera.
    cam_mats_extrinsic : list of np.ndarray
        ``(3or4, 4)`` extrinsic per camera.

    Returns
    -------
    head_bboxes : np.ndarray
        ``(ncams, nframes, 4)`` with ``[x1, y1, x2, y2]`` per frame,
        NaN where head is not available.
    """
    ncams = len(data_2d_body)
    nframes = data_2d_body[0].shape[0]
    cam_mats_extrinsic = np.array(cam_mats_extrinsic)

    # Gather head 2D positions from all cameras: (ncams, nframes, 11, 2)
    head_2d = np.stack([b[:, _HEAD_INDICES, :2] for b in data_2d_body])
    sentinel = (head_2d[:, :, :, 0] == -1) & (head_2d[:, :, :, 1] == -1)
    head_2d[sentinel] = np.nan

    # Undistort
    head_flat = head_2d.reshape(ncams, -1, 2)
    head_undist = np.empty_like(head_flat, dtype=np.float64)
    for cam in range(ncams):
        pts = head_flat[cam].astype(np.float64)
        nan_mask = np.isnan(pts[:, 0])
        out = np.full_like(pts, np.nan)
        if (~nan_mask).any():
            valid = pts[~nan_mask].reshape(-1, 1, 2)
            undist = cv.undistortPoints(valid, cam_mats_intrinsic[cam],
                                        np.zeros(5), P=cam_mats_intrinsic[cam])
            out[~nan_mask] = undist.reshape(-1, 2)
        head_undist[cam] = out

    # Simple DLT triangulation per head landmark per frame.
    # For the head centre we average the triangulated head landmarks.
    n_head = len(_HEAD_INDICES)
    head_3d = np.full((nframes, n_head, 3), np.nan, dtype=np.float64)

    for lm in range(n_head):
        pts = head_undist[:, lm::n_head, :]  # (ncams, nframes, 2)
        # Per-frame DLT
        for f in range(nframes):
            visible = ~np.isnan(pts[:, f, 0])
            if visible.sum() < 2:
                continue
            cams_vis = np.where(visible)[0]
            A = np.zeros((2 * len(cams_vis), 4))
            for i, c in enumerate(cams_vis):
                P = cam_mats_intrinsic[c] @ cam_mats_extrinsic[c][:3]
                x, y = pts[c, f]
                A[2 * i] = x * P[2] - P[0]
                A[2 * i + 1] = y * P[2] - P[1]
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            head_3d[f, lm] = X[:3] / X[3]

    # Head centre and radius
    head_centre = np.nanmean(head_3d, axis=1)  # (nframes, 3)
    head_span = np.nanmax(head_3d, axis=1) - np.nanmin(head_3d, axis=1)
    head_radius = np.nanmax(head_span, axis=1) / 2.0  # (nframes,)

    valid = ~np.isnan(head_centre[:, 0])
    print(f'  Head triangulated for face bboxes: {valid.sum()}/{nframes} frames')

    # Reproject to each camera → bboxes
    margin_factor = 2.0
    min_crop_px = 100
    head_bboxes = np.full((ncams, nframes, 4), np.nan, dtype=np.float64)

    pts_h = np.hstack([head_centre[valid], np.ones((valid.sum(), 1))])
    pts_off = head_centre[valid].copy()
    pts_off[:, 0] += head_radius[valid]
    pts_off_h = np.hstack([pts_off, np.ones((valid.sum(), 1))])

    for cam in range(ncams):
        ext = cam_mats_extrinsic[cam]
        intr = cam_mats_intrinsic[cam]

        X_img = intr @ (ext @ pts_h.T)[:3]
        uv = (X_img[:2] / X_img[2:3]).T

        X_img_off = intr @ (ext @ pts_off_h.T)[:3]
        uv_off = (X_img_off[:2] / X_img_off[2:3]).T
        px_radius = np.linalg.norm(uv_off - uv, axis=1)

        half = np.maximum(px_radius * margin_factor, min_crop_px / 2.0)
        head_bboxes[cam, valid] = np.column_stack([
            uv[:, 0] - half, uv[:, 1] - half,
            uv[:, 0] + half, uv[:, 1] + half,
        ])

    return head_bboxes


def _detect_face_with_bboxes(cam, input_stream, head_bboxes_cam, nframes,
                              undistort_map, data_save_path, use_gpu=False):
    """Two-stage face detection on a single camera.

    Stage 1: ``FaceDetector`` (BlazeFace) on the triangulated head-bbox
    crop → robust face bounding-box even under difficult lighting /
    oblique angles.

    Stage 2: ``FaceLandmarker`` on the tight face crop → 478 landmarks.

    Parameters
    ----------
    cam : int
        Camera index.
    input_stream : str
        Video file path.
    head_bboxes_cam : np.ndarray
        ``(nframes, 4)`` head bboxes ``[x1, y1, x2, y2]`` for this camera.
    nframes : int
        Number of frames to process.
    undistort_map : tuple
        ``(map1, map2)`` for cv.remap undistortion.
    data_save_path : str
        Directory to save ``2Dlandmarks_face.npy``.
    use_gpu : bool
        Whether to use GPU delegate.
    """
    delegate = mp.tasks.BaseOptions.Delegate.GPU if use_gpu else mp.tasks.BaseOptions.Delegate.CPU

    # Stage 1: FaceDetector — robust face localization
    detector_options = FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=face_detector_model_path, delegate=delegate),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.25,
    )
    face_detector = FaceDetector.create_from_options(detector_options)

    # Stage 2: FaceLandmarker — 478-landmark extraction.
    # Low thresholds are safe here because Stage 1 already confirmed a
    # face is present; we just need the landmark model to accept the crop.
    landmarker_options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=face_model_path, delegate=delegate),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = FaceLandmarker.create_from_options(landmarker_options)

    container = av.open(input_stream)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    map1, map2 = undistort_map

    kpts_face = []
    n_detected = 0
    num_face_keypoints = 478
    face_bbox_pad = 0.30  # 30 % padding around the detected face bbox

    framenum = 0
    for frame in container.decode(video=0):
        if framenum >= nframes:
            break

        frame_keypoints_face = []

        bbox = head_bboxes_cam[framenum]
        if not np.isnan(bbox[0]):
            img = frame.to_ndarray(format='rgb24')
            img = cv.remap(img, map1, map2, cv.INTER_LINEAR)
            img_h, img_w = img.shape[:2]

            # Head-bbox crop (coarse, from triangulated body)
            hx1 = max(0, int(bbox[0]))
            hy1 = max(0, int(bbox[1]))
            hx2 = min(img_w, int(bbox[2]))
            hy2 = min(img_h, int(bbox[3]))

            if (hx2 - hx1) >= 50 and (hy2 - hy1) >= 50:
                head_crop = np.ascontiguousarray(img[hy1:hy2, hx1:hx2])
                mp_head = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=head_crop)

                # --- Stage 1: FaceDetector on head crop ---
                try:
                    det_result = face_detector.detect(mp_head)
                except Exception:
                    det_result = None

                if (det_result is not None and det_result.detections):
                    det = det_result.detections[0]
                    db = det.bounding_box  # pixel coords in head_crop

                    # Tight face bbox in full-frame coords (with padding)
                    fw = int(db.width * (1 + face_bbox_pad))
                    fh = int(db.height * (1 + face_bbox_pad))
                    fcx = hx1 + db.origin_x + db.width // 2
                    fcy = hy1 + db.origin_y + db.height // 2
                    fx1 = max(0, fcx - fw // 2)
                    fy1 = max(0, fcy - fh // 2)
                    fx2 = min(img_w, fcx + fw // 2)
                    fy2 = min(img_h, fcy + fh // 2)

                    if (fx2 - fx1) >= 30 and (fy2 - fy1) >= 30:
                        face_crop = np.ascontiguousarray(
                            img[fy1:fy2, fx1:fx2])
                        mp_face = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=face_crop)

                        # --- Stage 2: FaceLandmarker on face crop ---
                        try:
                            lm_result = face_landmarker.detect(mp_face)
                        except Exception:
                            lm_result = None

                        if (lm_result is not None
                                and lm_result.face_landmarks):
                            face_lm = lm_result.face_landmarks[0]
                            fc_W = fx2 - fx1
                            fc_H = fy2 - fy1
                            frame_keypoints_face = [
                                [round(lm.x * fc_W) + fx1,
                                 round(lm.y * fc_H) + fy1,
                                 lm.z, lm.visibility, lm.presence]
                                for lm in face_lm
                            ]
                            n_detected += 1

        if len(frame_keypoints_face) < num_face_keypoints:
            frame_keypoints_face += [[-1, -1, -1, -1, -1]] * (
                num_face_keypoints - len(frame_keypoints_face))
        kpts_face.append(frame_keypoints_face)
        framenum += 1

    face_detector.close()
    face_landmarker.close()
    container.close()

    kpts_face_arr = np.array(kpts_face)
    np.save(os.path.join(data_save_path, '2Dlandmarks_face.npy'), kpts_face_arr)
    print(f'    cam{cam}: face detected {n_detected}/{nframes} frames (3D-guided)')
    return cam


def _run_face_detection_pass(input_streams, gui_options, cam_mats_intrinsic,
                              cam_mats_extrinsic, cam_dist_coeffs,
                              outdir_data2d_trial, undistort_maps):
    """Second-pass face detection using triangulated head bboxes.

    Called after the main body+hand detection pass completes.  Loads the
    body landmark data from disk, triangulates head landmarks to 3D,
    reprojects to each camera for face bboxes, then runs face detection
    per camera with cropped frames.
    """
    ncams = len(input_streams)
    use_gpu = gui_options.get('use_gpu', False)

    # Load body data from all cameras (saved by the first pass)
    body_data = []
    for cam in range(ncams):
        path = os.path.join(outdir_data2d_trial, f'cam{cam}', '2Dlandmarks_body.npy')
        body_data.append(np.load(path).astype(float))
    nframes = body_data[0].shape[0]

    # Triangulate head → reproject → bboxes
    head_bboxes = _triangulate_head(body_data, cam_mats_intrinsic, cam_mats_extrinsic)

    # Run face detection per camera (parallel)
    num_processes = gui_options.get('num_processes', os.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for cam in range(ncams):
            data_save_path = os.path.join(outdir_data2d_trial, f'cam{cam}')
            futures.append(executor.submit(
                _detect_face_with_bboxes,
                cam, input_streams[cam], head_bboxes[cam], nframes,
                undistort_maps[cam], data_save_path, use_gpu,
            ))
        for future in concurrent.futures.as_completed(futures):
            future.result()


def create_video(image_folder, extension, fps, output_folder, video_name):
    """
    Compiles a set of images into a video in sequential order.

    Parameters:
        image_folder (str): The directory containing the images.
        extension (str): The file extension of the images (e.g., '.png', '.jpg').
        fps (float): Frames per second for the output video.
        output_folder (str): The directory where the output video will be saved.
        video_name (str): The filename of the output video.

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get the list of images and sort them by the frame number
    images = [img for img in os.listdir(image_folder) if img.endswith(extension)]
    if not images:
        print(f"No images found in {image_folder}.")
        return

    images.sort()
    # Read the first image to get the frame dimensions
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv.imread(first_frame_path)
    if frame is None:
        print(f"Failed to read the first image at {first_frame_path}.")
        return
    height, width, layers = frame.shape

    # Set the codec and create the video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_folder, video_name)
    video = cv.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each image to the video file
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        video.write(frame)

    # Release the video writer
    video.release()
    cv.destroyAllWindows()


def read_calibration(calibration_files, extension):
    """
    Reads camera calibration parameters from YAML files.

    Parameters:
        calibration_files (list of str): List of file paths to the camera calibration YAML files.

    Returns:
        tuple: A tuple containing:
            - extrinsics (list of np.ndarray): List of 4x4 extrinsic matrices for each camera.
            - intrinsics (list of np.ndarray): List of 3x3 intrinsic matrices for each camera.
            - dist_coeffs (list of np.ndarray): List of distortion coefficients for each camera.
    """
    extrinsics = []
    intrinsics = []
    dist_coeffs = []

    if extension == '*.yaml':
        for cam_file in calibration_files:
            # Grab camera calibration parameters
            cam_yaml = cv.FileStorage(cam_file, cv.FILE_STORAGE_READ)
            cam_int = cam_yaml.getNode("intrinsicMatrix").mat()
            cam_dist = cam_yaml.getNode("distortionCoefficients").mat()
            cam_rotn = cam_yaml.getNode("R").mat().transpose()
            cam_transln = cam_yaml.getNode("T").mat()
            cam_transform = transformation_matrix(cam_rotn, cam_transln)

            # Store calibration parameters
            extrinsics.append(cam_transform)
            intrinsics.append(cam_int.transpose())
            dist_coeffs.append(cam_dist.reshape(-1))

    elif extension == '*.toml':
        cal = toml.load(calibration_files)
        ncams = len(cal) - 1
        
        for cam in range(ncams):
            camname = 'cam_' + str(cam)

            # Camera extrinsic parameters
            cam_rotn = np.array(cal[camname]['rotation'])
            cam_transln = np.array(cal[camname]['translation'])
            cam_transform = transformation_matrix(rotation_matrix(cam_rotn), cam_transln)
            extrinsics.append(cam_transform)
            
            # Camera intrinsic parameters
            cam_int = np.array(cal[camname]['matrix'])
            intrinsics.append(cam_int)

            # Camera distortion coefficients
            cam_dist = np.array(cal[camname]['distortions'])
            dist_coeffs.append(cam_dist)

    return extrinsics, intrinsics, dist_coeffs


def transformation_matrix(R, t):
    """
    Creates a 4x4 homogeneous transformation matrix from rotation and translation.

    Parameters:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-element translation vector.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    """
    T = np.concatenate((R, t.reshape(3, 1)), axis=1)
    T = np.vstack((T, [0, 0, 0, 1]))
    return T


def rotation_matrix(r):
    """
    Create rotation matrix from a rotation vector.

    :param r: Axis of rotation.
    :return: 3x3 rotation matrix.
    """

    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        axis = r / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        return R


def process_camera(cam, input_stream, gui_options, cam_mats_intrinsic, cam_dist_coeffs, undistort_map, display_width,
                   display_height, progress_queue):
    """
    Processes a single camera stream and saves keypoints directly to disk.

    Parameters:
        cam (int): Camera index.
        input_stream (str): Path to the input video file.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        undistort_map (tuple): Precomputed undistortion maps for the camera (map1, map2).
        display_width (int): Width for displaying frames.
        display_height (int): Height for displaying frames.
        progress_queue (multiprocessing.Queue): Queue for communicating progress.

    Returns:
        int: The camera index.
    """
    print(f"Starting processing for camera {cam}")

    # Extract options from the gui_options dictionary
    save_images = gui_options['save_images_mp']
    use_gpu = gui_options['use_gpu']
    process_to_frame = gui_options['fraction_frames']
    outdir_images_trial = gui_options['outdir_images_trial']
    outdir_data2d_trial = gui_options['outdir_data2d_trial']
    hand_confidence = gui_options['hand_confidence']
    pose_confidence = gui_options['pose_confidence']
    use_face_mesh = gui_options.get('use_face_mesh', False)

    # Paths for saving data
    data_save_path = os.path.join(outdir_data2d_trial, f'cam{cam}')
    os.makedirs(data_save_path, exist_ok=True)

    # Initialize file paths for saving keypoints
    kpts_cam_l_file = os.path.join(data_save_path, '2Dlandmarks_left.npy')
    kpts_cam_r_file = os.path.join(data_save_path, '2Dlandmarks_right.npy')
    kpts_body_file = os.path.join(data_save_path, '2Dlandmarks_body.npy')
    kpts_cam_l_world_file = os.path.join(data_save_path, '2Dworldlandmarks_left.npy')
    kpts_cam_r_world_file = os.path.join(data_save_path, '2Dworldlandmarks_right.npy')
    kpts_body_world_file = os.path.join(data_save_path, '2Dworldlandmarks_body.npy')
    confidence_hand_file = os.path.join(data_save_path, 'handedness_score.npy')

    # Prepare lists to store keypoints
    kpts_cam_l = []
    kpts_cam_r = []
    kpts_body = []
    kpts_cam_l_world = []
    kpts_cam_r_world = []
    kpts_body_world = []
    handscore = []
    kpts_face = []

    # Set GPU delegate based on user selection
    delegate = mp.tasks.BaseOptions.Delegate.GPU if use_gpu else mp.tasks.BaseOptions.Delegate.CPU

    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=hand_confidence,
        min_hand_presence_confidence=hand_confidence,
        min_tracking_confidence=hand_confidence
    )
    pose_options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=pose_confidence,
        min_pose_presence_confidence=pose_confidence,
        min_tracking_confidence=pose_confidence
    )

    # Face mesh options (optional).
    # Lower confidence thresholds than default (0.5) to improve detection in
    # multi-camera setups where many cameras see the face at oblique angles.
    face_landmarker = None
    if use_face_mesh:
        face_options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path, delegate=delegate),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)

    # Create PyAV container and video stream
    container = av.open(input_stream)
    video_stream = container.streams.video[0]
    video_stream.thread_type = 'AUTO'  # Enable multi-threaded decoding

    # Get video FPS and total frames
    if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
        fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
    else:
        fps = 30.0  # Default FPS if not available

    total_frames = video_stream.frames
    if total_frames == 0:
        # Estimate total frames if not available
        duration_s = container.duration / av.time_base
        total_frames = int(duration_s * fps)

    # Initialize HandLandmarker and PoseLandmarker for this camera
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Define expected lengths
    num_hand_keypoints = 21
    num_body_keypoints = 33

    # Start time for processing FPS calculation
    start_time = time.time()
    last_fps_time = start_time
    frames_since_last_fps = 0

    # Initialize frame number
    framenum = 0
    max_frames = int(process_to_frame * total_frames)

    prev_timestamp_ms = -1  # Initialize previous timestamp

    # Use frame iterator directly
    frame_iter = container.decode(video=0)
    for frame in frame_iter:
        if framenum >= max_frames:
            break

        # Convert PyAV frame to NumPy array
        # Use RGBA format directly when GPU is enabled to avoid a separate conversion
        if use_gpu:
            frame_array = frame.to_ndarray(format='rgba')
        else:
            frame_array = frame.to_ndarray(format='rgb24')

        # Undistort image using precomputed maps
        map1, map2 = undistort_map
        frame_array = cv.remap(frame_array, map1, map2, interpolation=cv.INTER_LINEAR)

        # Create MediaPipe image
        if use_gpu:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_array)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)

        # Calculate timestamp for MediaPipe in milliseconds
        timestamp_ms = int(framenum * 1000 / fps)

        # Ensure timestamps are strictly increasing
        if timestamp_ms <= prev_timestamp_ms:
            timestamp_ms = prev_timestamp_ms + 1  # Increment by 1 ms
        prev_timestamp_ms = timestamp_ms

        # Hand Landmarks detection
        try:
            hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in hand_landmarker.detect_for_video: {e}")
            hand_results = mp.tasks.vision.HandLandmarkerResult([], [], [])

        # Pose Landmarks detection
        try:
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in pose_landmarker.detect_for_video: {e}")
            pose_results = mp.tasks.vision.PoseLandmarkerResult([], [])

        # Hand Landmarks processing
        frame_keypoints_l = []
        frame_keypoints_r = []
        frame_keypoints_l_world = []
        frame_keypoints_r_world = []
        frame_handscore = [-1, -1]  # Default -1 (not detected)

        if hand_results.hand_landmarks:
            for hand_landmarks, hand_world_landmarks, handedness_list in zip(
                hand_results.hand_landmarks,
                hand_results.hand_world_landmarks,
                hand_results.handedness
            ):
                handedness = handedness_list[0]

                # Process Left and Right hands separately
                if handedness.category_name == 'Left':
                    frame_keypoints_l = [[
                        int(frame_array.shape[1] * hand_landmark.x),
                        int(frame_array.shape[0] * hand_landmark.y),
                        hand_landmark.z,
                        hand_landmark.visibility,
                        hand_landmark.presence
                    ] for hand_landmark in hand_landmarks]
                    frame_keypoints_l_world = [[
                        hand_world_landmark.x,
                        hand_world_landmark.y,
                        hand_world_landmark.z,
                        hand_world_landmark.visibility,
                        hand_world_landmark.presence
                    ] for hand_world_landmark in hand_world_landmarks]
                    frame_handscore[0] = handedness.score
                else:
                    frame_keypoints_r = [[
                        int(frame_array.shape[1] * hand_landmark.x),
                        int(frame_array.shape[0] * hand_landmark.y),
                        hand_landmark.z,
                        hand_landmark.visibility,
                        hand_landmark.presence
                    ] for hand_landmark in hand_landmarks]
                    frame_keypoints_r_world = [[
                        hand_world_landmark.x,
                        hand_world_landmark.y,
                        hand_world_landmark.z,
                        hand_world_landmark.visibility,
                        hand_world_landmark.presence
                    ] for hand_world_landmark in hand_world_landmarks]
                    frame_handscore[1] = handedness.score

        # Pose Landmarks processing
        frame_keypoints_body = []
        frame_keypoints_body_world = []
        if pose_results.pose_landmarks:
            for pose_landmarks, pose_world_landmarks in zip(
                pose_results.pose_landmarks,
                pose_results.pose_world_landmarks
            ):
                frame_keypoints_body = [[
                    int(body_landmark.x * frame_array.shape[1]),
                    int(body_landmark.y * frame_array.shape[0]),
                    body_landmark.z,
                    body_landmark.visibility,
                    body_landmark.presence
                ] for body_landmark in pose_landmarks]
                frame_keypoints_body_world = [[
                    body_world_landmark.x,
                    body_world_landmark.y,
                    body_world_landmark.z,
                    body_world_landmark.visibility,
                    body_world_landmark.presence
                ] for body_world_landmark in pose_world_landmarks]

        # Face Landmarks detection (optional)
        num_face_keypoints = 478
        frame_keypoints_face = []
        if face_landmarker is not None:
            try:
                face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"Error in face_landmarker.detect_for_video: {e}")
                face_results = None

            if face_results is not None and face_results.face_landmarks:
                face_lm = face_results.face_landmarks[0]  # first (most confident) face
                W = frame_array.shape[1]
                H = frame_array.shape[0]
                frame_keypoints_face = [
                    [int(lm.x * W), int(lm.y * H), lm.z, lm.visibility, lm.presence]
                    for lm in face_lm
                ]

            # Pad missing face detections
            if len(frame_keypoints_face) < num_face_keypoints:
                frame_keypoints_face += [[-1, -1, -1, -1, -1]] * (num_face_keypoints - len(frame_keypoints_face))

            kpts_face.append(frame_keypoints_face)

        # Ensure correct number of keypoints by padding
        if len(frame_keypoints_l) < num_hand_keypoints:
            frame_keypoints_l += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l))
            frame_keypoints_l_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l_world))
        if len(frame_keypoints_r) < num_hand_keypoints:
            frame_keypoints_r += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r))
            frame_keypoints_r_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r_world))
        if len(frame_keypoints_body) < num_body_keypoints:
            frame_keypoints_body += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body))
            frame_keypoints_body_world += [[-1, -1, -1, -1, -1]] * (
                num_body_keypoints - len(frame_keypoints_body_world)
            )

        # Append keypoints
        kpts_cam_l.append(frame_keypoints_l)
        kpts_cam_r.append(frame_keypoints_r)
        kpts_body.append(frame_keypoints_body)
        kpts_cam_l_world.append(frame_keypoints_l_world)
        kpts_cam_r_world.append(frame_keypoints_r_world)
        kpts_body_world.append(frame_keypoints_body_world)

        # Handedness confidence
        handscore.append(frame_handscore)

        # Draw skeleton using the same format as the triangulation phase
        if save_images:
            frame_bgr_draw = cv.cvtColor(frame_array, cv.COLOR_RGB2BGR)
            draw_landmarks_unified(
                frame_bgr_draw, frame_keypoints_body,
                frame_keypoints_r, frame_keypoints_l,
                face_kpts=frame_keypoints_face if frame_keypoints_face else None,
            )
            frame_array = cv.cvtColor(frame_bgr_draw, cv.COLOR_BGR2RGB)

        # Resize and save images if needed
        if save_images:
            resized_frame = cv.resize(frame_array, (display_width, display_height))
            save_path = os.path.join(outdir_images_trial, f'cam{cam}', f'frame{framenum:06d}.png')
            result = cv.imwrite(save_path, cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR))
            if not result:
                print(f"Failed to save frame {framenum:06d} for cam {cam} at {save_path}")

        # Increment frame number
        framenum += 1
        frames_since_last_fps += 1

        # Send progress update every N frames
        if framenum % 10 == 0 or framenum == max_frames:
            progress = (framenum / max_frames) * 100
            progress_queue.put({'cam': cam, 'progress': progress})

        # Calculate processing FPS every N frames
        if frames_since_last_fps >= 10 or framenum == max_frames:
            current_time = time.time()
            elapsed_time = current_time - last_fps_time
            if elapsed_time > 0:
                processing_fps = frames_since_last_fps / elapsed_time
                progress_queue.put({'cam': cam, 'fps': processing_fps})
            last_fps_time = current_time
            frames_since_last_fps = 0

    # After processing all frames, convert lists to NumPy arrays and save to disk
    kpts_cam_l = np.array(kpts_cam_l)
    kpts_cam_r = np.array(kpts_cam_r)
    kpts_body = np.array(kpts_body)
    kpts_cam_l_world = np.array(kpts_cam_l_world)
    kpts_cam_r_world = np.array(kpts_cam_r_world)
    kpts_body_world = np.array(kpts_body_world)
    confidence_hand = np.array(handscore)

    # Save the results to disk
    np.save(kpts_cam_l_file, kpts_cam_l)
    np.save(kpts_cam_r_file, kpts_cam_r)
    np.save(kpts_body_file, kpts_body)
    np.save(kpts_cam_l_world_file, kpts_cam_l_world)
    np.save(kpts_cam_r_world_file, kpts_cam_r_world)
    np.save(kpts_body_world_file, kpts_body_world)
    np.save(confidence_hand_file, confidence_hand)

    # Save face landmarks (optional)
    if use_face_mesh and kpts_face:
        kpts_face_arr = np.array(kpts_face)  # (nframes, 478, 5)
        n_detected = np.sum(kpts_face_arr[:, 0, 0] != -1)
        np.save(os.path.join(data_save_path, '2Dlandmarks_face.npy'), kpts_face_arr)
        print(f'  Cam {cam}: Face landmarks saved ({n_detected}/{len(kpts_face_arr)} frames detected)')

    # Release resources
    hand_landmarker.close()
    pose_landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()
    container.close()

    # Send a completion message for this camera
    progress_queue.put({'cam': cam, 'done': True})

    return cam


def run_mediapipe(input_streams, gui_options, cam_mats_intrinsic, cam_dist_coeffs, outdir_images_trial,
                  outdir_data2d_trial, trialname, cam_mats_extrinsic=None,
                  display_width=450, display_height=360, progress_queue=None):
    """
    Processes multiple camera streams in parallel using multiprocessing.

    Parameters:
        input_streams (list): List of input video file paths.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        outdir_images_trial (str): Output directory for images.
        outdir_data2d_trial (str): Output directory for data.
        trialname (str): Name of the trial.
        display_width (int, optional): Width for displaying frames. Defaults to 450.
        display_height (int, optional): Height for displaying frames. Defaults to 360.
        progress_queue (multiprocessing.Queue, optional): Queue for communicating progress.

    Returns:
        None
    """
    num_processes = gui_options.get('num_processes', os.cpu_count())

    # Precompute undistortion maps
    undistort_maps = []
    frame_width, frame_height = None, None

    # Get frame dimensions and undistort maps from the first frame of each camera
    for cam, input_stream in enumerate(input_streams):
        frame_count = 0
        container = av.open(input_stream)
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frame_array = frame.to_ndarray(format='rgb24')
                frame_height, frame_width = frame_array.shape[:2]
                # Set up undistort maps
                map1, map2 = cv.initUndistortRectifyMap(
                    cam_mats_intrinsic[cam],
                    cam_dist_coeffs[cam],
                    None,
                    cam_mats_intrinsic[cam],
                    (frame_width, frame_height),
                    cv.CV_16SC2
                )
                undistort_maps.append((map1, map2))

                # Need to force this to occur once a frame is read, otherwise it breaks too early for some vids
                frame_count += 1 
                if frame_count == 1:  # Only need one frame
                    break
            if frame_count == 1:  # Only need one packet
                break

        container.close()

    # Update gui_options with additional information
    gui_options['outdir_images_trial'] = outdir_images_trial
    gui_options['outdir_data2d_trial'] = outdir_data2d_trial
    gui_options['trialname'] = trialname

    # Create a copy of gui_options without GUI elements
    gui_options_no_gui = gui_options.copy()
    # Remove any GUI elements from gui_options if they are present
    gui_elements = ['fps_label', 'progress_bar', 'root', 'fps_value', 'progress_var']
    for key in gui_elements:
        gui_options_no_gui.pop(key, None)  # Safely remove if present

    # Use ProcessPoolExecutor to process cameras in parallel
    total_cameras = len(input_streams)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for cam, input_stream in enumerate(input_streams):
            futures.append(executor.submit(
                process_camera,
                cam,
                input_stream,
                gui_options_no_gui,
                cam_mats_intrinsic,
                cam_dist_coeffs,
                undistort_maps[cam],
                display_width,
                display_height,
                progress_queue  # Pass the progress_queue to child processes
            ))

        # Wait for all processes to complete
        for future in concurrent.futures.as_completed(futures):
            cam = future.result()
            print(f"Camera {cam} processing complete.")

    # Hand reassignment pass: triangulate body wrists to 3D, reproject to
    # every camera, and reassign left/right hands based on proximity.
    if cam_mats_extrinsic is not None:
        print('Reassigning hand labels from triangulated 3D wrists...')
        _reassign_hands_from_3d_wrists(
            cam_mats_intrinsic, cam_mats_extrinsic, outdir_data2d_trial)

        # Redraw saved PNGs so that videos reflect post-reassignment labels
        if gui_options.get('save_images_mp', False):
            print('Redrawing saved images with corrected hand labels...')
            _redraw_saved_images(
                input_streams, outdir_images_trial, outdir_data2d_trial,
                cam_mats_intrinsic, cam_dist_coeffs,
                display_width=display_width, display_height=display_height,
                hand_backend='mediapipe',
                use_gpu=gui_options.get('use_gpu', False),
            )

    # Second pass: 3D-guided face detection.
    # After all cameras have body landmarks, triangulate the head to 3D
    # and reproject into every camera for robust face bboxes.
    if gui_options.get('use_face_mesh', False):
        print('Running 3D-guided face detection (second pass)...')
        _run_face_detection_pass(
            input_streams, gui_options_no_gui, cam_mats_intrinsic,
            cam_mats_extrinsic, cam_dist_coeffs,
            outdir_data2d_trial, undistort_maps,
        )


def process_camera_hybrid(cam, input_stream, gui_options, cam_mats_intrinsic, cam_dist_coeffs, undistort_map,
                           display_width, display_height, progress_queue):
    """
    Processes a single camera stream using MediaPipe for body pose and HaMeR for hands.

    Produces identical .npy output format to process_camera(), so downstream
    triangulaterefine.py works unchanged.

    Parameters:
        cam (int): Camera index.
        input_stream (str): Path to the input video file.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        undistort_map (tuple): Precomputed undistortion maps for the camera (map1, map2).
        display_width (int): Width for displaying frames.
        display_height (int): Height for displaying frames.
        progress_queue (multiprocessing.Queue): Queue for communicating progress.

    Returns:
        int: The camera index.
    """
    print(f"[HaMeR+MediaPipe] Starting processing for camera {cam}")

    # Lazy import of HaMeR module
    from athena.hamer_hands import (
        load_models,
        detect_hands_mp_landmarks,
        get_mano_faces,
        _get_device,
    )

    # Extract options
    save_images = gui_options['save_images_mp']
    use_gpu = gui_options['use_gpu']
    process_to_frame = gui_options['fraction_frames']
    outdir_images_trial = gui_options['outdir_images_trial']
    outdir_data2d_trial = gui_options['outdir_data2d_trial']
    pose_confidence = gui_options['pose_confidence']
    use_face_mesh = gui_options.get('use_face_mesh', False)

    # Device selection (CUDA > MPS > CPU)
    import torch
    device = _get_device(use_gpu)

    # Load HaMeR hand model
    hamer_model, hamer_cfg, device = load_models(device=device, use_gpu=use_gpu)

    # MediaPipe PoseLandmarker for body keypoints and hand bounding boxes
    delegate = mp.tasks.BaseOptions.Delegate.GPU if use_gpu else mp.tasks.BaseOptions.Delegate.CPU
    pose_options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=pose_confidence,
        min_pose_presence_confidence=pose_confidence,
        min_tracking_confidence=pose_confidence
    )

    # MediaPipe HandLandmarker provides adaptive bounding boxes for HaMeR
    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    # Face mesh (optional).
    # Lower confidence thresholds than default (0.5) to improve detection in
    # multi-camera setups where many cameras see the face at oblique angles.
    face_landmarker = None
    if use_face_mesh:
        face_options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path, delegate=delegate),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)

    # Paths for saving data
    data_save_path = os.path.join(outdir_data2d_trial, f'cam{cam}')
    os.makedirs(data_save_path, exist_ok=True)

    kpts_cam_l_file = os.path.join(data_save_path, '2Dlandmarks_left.npy')
    kpts_cam_r_file = os.path.join(data_save_path, '2Dlandmarks_right.npy')
    kpts_body_file = os.path.join(data_save_path, '2Dlandmarks_body.npy')
    kpts_cam_l_world_file = os.path.join(data_save_path, '2Dworldlandmarks_left.npy')
    kpts_cam_r_world_file = os.path.join(data_save_path, '2Dworldlandmarks_right.npy')
    kpts_body_world_file = os.path.join(data_save_path, '2Dworldlandmarks_body.npy')
    confidence_hand_file = os.path.join(data_save_path, 'handedness_score.npy')

    # Prepare lists to store keypoints
    kpts_cam_l = []
    kpts_cam_r = []
    kpts_body = []
    kpts_cam_l_world = []
    kpts_cam_r_world = []
    kpts_body_world = []
    handscore = []
    kpts_face = []
    verts_l_all = []   # HaMeR mesh vertices per frame (778, 3)
    verts_r_all = []
    verts_2d_l_all = []  # HaMeR mesh vertex 2D pixel coords per frame (778, 2)
    verts_2d_r_all = []

    # Expected keypoint counts
    num_hand_keypoints = 21
    num_body_keypoints = 33

    # Open video
    container = av.open(input_stream)
    video_stream = container.streams.video[0]
    video_stream.thread_type = 'AUTO'

    if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
        fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
    else:
        fps = 30.0

    total_frames = video_stream.frames
    if total_frames == 0:
        duration_s = container.duration / av.time_base
        total_frames = int(duration_s * fps)

    # Initialize MediaPipe PoseLandmarker
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Start time for FPS calculation
    start_time = time.time()
    last_fps_time = start_time
    frames_since_last_fps = 0

    framenum = 0
    max_frames = int(process_to_frame * total_frames)
    prev_timestamp_ms = -1

    frame_iter = container.decode(video=0)
    for frame in frame_iter:
        if framenum >= max_frames:
            break

        # Decode frame
        if use_gpu:
            frame_array = frame.to_ndarray(format='rgba')
        else:
            frame_array = frame.to_ndarray(format='rgb24')

        # Undistort
        map1, map2 = undistort_map
        frame_array = cv.remap(frame_array, map1, map2, interpolation=cv.INTER_LINEAR)

        frame_height, frame_width = frame_array.shape[:2]

        # Create MediaPipe image for pose detection
        if use_gpu:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_array)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)

        # Timestamp
        timestamp_ms = int(framenum * 1000 / fps)
        if timestamp_ms <= prev_timestamp_ms:
            timestamp_ms = prev_timestamp_ms + 1
        prev_timestamp_ms = timestamp_ms

        # --- MediaPipe Pose Detection (always) ---
        try:
            pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in pose_landmarker.detect_for_video: {e}")
            pose_results = mp.tasks.vision.PoseLandmarkerResult([], [])

        # --- Body Landmarks Processing (from MediaPipe) ---
        frame_keypoints_body = []
        frame_keypoints_body_world = []
        left_wrist_px = None
        right_wrist_px = None

        if pose_results.pose_landmarks:
            for pose_landmarks, pose_world_landmarks in zip(
                pose_results.pose_landmarks,
                pose_results.pose_world_landmarks
            ):
                frame_keypoints_body = [[
                    int(body_landmark.x * frame_width),
                    int(body_landmark.y * frame_height),
                    body_landmark.z,
                    body_landmark.visibility,
                    body_landmark.presence
                ] for body_landmark in pose_landmarks]
                frame_keypoints_body_world = [[
                    body_world_landmark.x,
                    body_world_landmark.y,
                    body_world_landmark.z,
                    body_world_landmark.visibility,
                    body_world_landmark.presence
                ] for body_world_landmark in pose_world_landmarks]

                # Extract wrist locations for wrist-prior mode
                # MediaPipe: index 15 = left wrist, 16 = right wrist
                lw = pose_landmarks[15]
                rw = pose_landmarks[16]
                if lw.visibility > 0.3 and lw.presence > 0.3:
                    left_wrist_px = (int(lw.x * frame_width), int(lw.y * frame_height))
                if rw.visibility > 0.3 and rw.presence > 0.3:
                    right_wrist_px = (int(rw.x * frame_width), int(rw.y * frame_height))

        # --- HaMeR Hand Detection ---
        # Convert frame to BGR for HaMeR
        if use_gpu:
            frame_bgr = cv.cvtColor(frame_array, cv.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv.cvtColor(frame_array, cv.COLOR_RGB2BGR)

        # Use MediaPipe HandLandmarker for adaptive bounding boxes.
        # MediaPipe's handedness labels are unreliable from single
        # camera views, so we re-assign left/right using the body
        # PoseLandmarker wrists (indices 15=left, 16=right) which
        # are far more consistent across cameras.
        try:
            hl_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in hand_landmarker.detect_for_video: {e}")
            hl_results = mp.tasks.vision.HandLandmarkerResult([], [], [])

        left_lm_arr = None
        right_lm_arr = None

        # Collect ALL detected hands (ignoring MediaPipe's label initially)
        detected_hands = []
        if hl_results.hand_landmarks:
            for hl_lm, hl_handedness in zip(hl_results.hand_landmarks, hl_results.handedness):
                lm_arr = np.array([
                    [int(lm.x * frame_width), int(lm.y * frame_height), lm.z, lm.visibility, lm.presence]
                    for lm in hl_lm
                ], dtype=np.float32)
                mp_label = hl_handedness[0].category_name  # original label (fallback only)
                detected_hands.append((lm_arr, mp_label))

        have_both_wrists = left_wrist_px is not None and right_wrist_px is not None

        if have_both_wrists and len(detected_hands) == 2:
            # Two hands detected — assign each to the closest body wrist
            lw = np.array(left_wrist_px, dtype=np.float32)
            rw = np.array(right_wrist_px, dtype=np.float32)
            c0 = detected_hands[0][0][:, :2].mean(axis=0)
            c1 = detected_hands[1][0][:, :2].mean(axis=0)
            cost_a = np.linalg.norm(c0 - lw) + np.linalg.norm(c1 - rw)
            cost_b = np.linalg.norm(c0 - rw) + np.linalg.norm(c1 - lw)
            if cost_a <= cost_b:
                left_lm_arr = detected_hands[0][0]
                right_lm_arr = detected_hands[1][0]
            else:
                left_lm_arr = detected_hands[1][0]
                right_lm_arr = detected_hands[0][0]

        elif have_both_wrists and len(detected_hands) == 1:
            # One hand detected — assign to the closer body wrist
            lm_arr = detected_hands[0][0]
            centroid = lm_arr[:, :2].mean(axis=0)
            lw = np.array(left_wrist_px, dtype=np.float32)
            rw = np.array(right_wrist_px, dtype=np.float32)
            if np.linalg.norm(centroid - lw) < np.linalg.norm(centroid - rw):
                left_lm_arr = lm_arr
            else:
                right_lm_arr = lm_arr

        else:
            # No body wrists available — fall back to MediaPipe label
            for lm_arr, mp_label in detected_hands:
                if mp_label == 'Left':
                    left_lm_arr = lm_arr
                else:
                    right_lm_arr = lm_arr

        hand_result = detect_hands_mp_landmarks(
            frame_bgr, left_lm_arr, right_lm_arr,
            hamer_model, hamer_cfg, device
        )

        # --- Convert HaMeR results to the standard 5-column format ---
        frame_keypoints_l = []
        frame_keypoints_r = []
        frame_keypoints_l_world = []
        frame_keypoints_r_world = []
        frame_handscore = [-1, -1]
        frame_verts_l = np.full((778, 3), -1, dtype=np.float32)
        frame_verts_r = np.full((778, 3), -1, dtype=np.float32)
        frame_verts_2d_l = np.full((778, 2), -1, dtype=np.float32)
        frame_verts_2d_r = np.full((778, 2), -1, dtype=np.float32)

        if hand_result['left_keypoints_2d'] is not None:
            kpts_2d = hand_result['left_keypoints_2d']   # (21, 2)
            kpts_3d = hand_result['left_keypoints_3d']   # (21, 3)
            frame_keypoints_l = [
                [int(kpts_2d[j, 0]), int(kpts_2d[j, 1]), float(kpts_3d[j, 2]), 1.0, 1.0]
                for j in range(21)
            ]
            frame_keypoints_l_world = [
                [float(kpts_3d[j, 0]), float(kpts_3d[j, 1]), float(kpts_3d[j, 2]), 1.0, 1.0]
                for j in range(21)
            ]
            frame_handscore[0] = 1.0
            if hand_result['left_vertices_3d'] is not None:
                frame_verts_l = hand_result['left_vertices_3d'].astype(np.float32)
            if hand_result.get('left_vertices_2d') is not None:
                frame_verts_2d_l = hand_result['left_vertices_2d'].astype(np.float32)

        if hand_result['right_keypoints_2d'] is not None:
            kpts_2d = hand_result['right_keypoints_2d']
            kpts_3d = hand_result['right_keypoints_3d']
            frame_keypoints_r = [
                [int(kpts_2d[j, 0]), int(kpts_2d[j, 1]), float(kpts_3d[j, 2]), 1.0, 1.0]
                for j in range(21)
            ]
            frame_keypoints_r_world = [
                [float(kpts_3d[j, 0]), float(kpts_3d[j, 1]), float(kpts_3d[j, 2]), 1.0, 1.0]
                for j in range(21)
            ]
            frame_handscore[1] = 1.0
            if hand_result['right_vertices_3d'] is not None:
                frame_verts_r = hand_result['right_vertices_3d'].astype(np.float32)
            if hand_result.get('right_vertices_2d') is not None:
                frame_verts_2d_r = hand_result['right_vertices_2d'].astype(np.float32)

        # --- Face Landmarks detection (optional) ---
        num_face_keypoints = 478
        frame_keypoints_face = []
        if face_landmarker is not None:
            try:
                face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"Error in face_landmarker.detect_for_video: {e}")
                face_results = None

            if face_results is not None and face_results.face_landmarks:
                face_lm = face_results.face_landmarks[0]  # first (most confident) face
                frame_keypoints_face = [
                    [int(lm.x * frame_width), int(lm.y * frame_height), lm.z, lm.visibility, lm.presence]
                    for lm in face_lm
                ]

            # Pad missing face detections
            if len(frame_keypoints_face) < num_face_keypoints:
                frame_keypoints_face += [[-1, -1, -1, -1, -1]] * (num_face_keypoints - len(frame_keypoints_face))

            kpts_face.append(frame_keypoints_face)

        # --- Padding for missing detections ---
        if len(frame_keypoints_l) < num_hand_keypoints:
            frame_keypoints_l += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l))
            frame_keypoints_l_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_l_world))
        if len(frame_keypoints_r) < num_hand_keypoints:
            frame_keypoints_r += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r))
            frame_keypoints_r_world += [[-1, -1, -1, -1, -1]] * (num_hand_keypoints - len(frame_keypoints_r_world))
        if len(frame_keypoints_body) < num_body_keypoints:
            frame_keypoints_body += [[-1, -1, -1, -1, -1]] * (num_body_keypoints - len(frame_keypoints_body))
            frame_keypoints_body_world += [[-1, -1, -1, -1, -1]] * (
                num_body_keypoints - len(frame_keypoints_body_world)
            )

        # --- Drawing (only when saving images) ---
        # Uses the unified skeleton drawing that matches the triangulation phase,
        # showing body + mesh hands + face in the same format.
        if save_images:
            # Build mesh dicts for this frame (None if no detection)
            mesh_r = None
            mesh_l = None
            mano_faces = get_mano_faces()
            if frame_verts_2d_r[0, 0] != -1:
                mesh_r = {'verts_2d': frame_verts_2d_r, 'verts_3d': frame_verts_r}
            if frame_verts_2d_l[0, 0] != -1:
                mesh_l = {'verts_2d': frame_verts_2d_l, 'verts_3d': frame_verts_l}

            frame_bgr_draw = cv.cvtColor(frame_array, cv.COLOR_RGB2BGR)
            draw_landmarks_unified(
                frame_bgr_draw, frame_keypoints_body,
                frame_keypoints_r, frame_keypoints_l,
                face_kpts=frame_keypoints_face if frame_keypoints_face else None,
                mesh_right=mesh_r, mesh_left=mesh_l, mesh_faces=mano_faces,
            )
            frame_array = cv.cvtColor(frame_bgr_draw, cv.COLOR_BGR2RGB)

        # Append keypoints
        kpts_cam_l.append(frame_keypoints_l)
        kpts_cam_r.append(frame_keypoints_r)
        kpts_body.append(frame_keypoints_body)
        kpts_cam_l_world.append(frame_keypoints_l_world)
        kpts_cam_r_world.append(frame_keypoints_r_world)
        kpts_body_world.append(frame_keypoints_body_world)
        handscore.append(frame_handscore)
        verts_l_all.append(frame_verts_l)
        verts_r_all.append(frame_verts_r)
        verts_2d_l_all.append(frame_verts_2d_l)
        verts_2d_r_all.append(frame_verts_2d_r)

        # Resize and save images if needed
        if save_images:
            resized_frame = cv.resize(frame_array, (display_width, display_height))
            save_path = os.path.join(outdir_images_trial, f'cam{cam}', f'frame{framenum:06d}.png')
            result_save = cv.imwrite(save_path, cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR))
            if not result_save:
                print(f"Failed to save frame {framenum:06d} for cam {cam} at {save_path}")

        framenum += 1
        frames_since_last_fps += 1

        if framenum % 10 == 0 or framenum == max_frames:
            progress = (framenum / max_frames) * 100
            progress_queue.put({'cam': cam, 'progress': progress})

        if frames_since_last_fps >= 10 or framenum == max_frames:
            current_time = time.time()
            elapsed_time = current_time - last_fps_time
            if elapsed_time > 0:
                processing_fps = frames_since_last_fps / elapsed_time
                progress_queue.put({'cam': cam, 'fps': processing_fps})
            last_fps_time = current_time
            frames_since_last_fps = 0

    # Save results
    kpts_cam_l = np.array(kpts_cam_l)
    kpts_cam_r = np.array(kpts_cam_r)
    kpts_body = np.array(kpts_body)
    kpts_cam_l_world = np.array(kpts_cam_l_world)
    kpts_cam_r_world = np.array(kpts_cam_r_world)
    kpts_body_world = np.array(kpts_body_world)
    confidence_hand = np.array(handscore)

    np.save(kpts_cam_l_file, kpts_cam_l)
    np.save(kpts_cam_r_file, kpts_cam_r)
    np.save(kpts_body_file, kpts_body)
    np.save(kpts_cam_l_world_file, kpts_cam_l_world)
    np.save(kpts_cam_r_world_file, kpts_cam_r_world)
    np.save(kpts_body_world_file, kpts_body_world)
    np.save(confidence_hand_file, confidence_hand)

    # Save HaMeR mesh vertices (778 vertices × 3 coords per hand per frame)
    verts_l_arr = np.array(verts_l_all, dtype=np.float32)  # (nframes, 778, 3)
    verts_r_arr = np.array(verts_r_all, dtype=np.float32)
    np.save(os.path.join(data_save_path, 'hamer_vertices_left.npy'), verts_l_arr)
    np.save(os.path.join(data_save_path, 'hamer_vertices_right.npy'), verts_r_arr)

    # Save HaMeR mesh vertex 2D pixel coordinates (778 vertices × 2 per hand per frame)
    verts_2d_l_arr = np.array(verts_2d_l_all, dtype=np.float32)  # (nframes, 778, 2)
    verts_2d_r_arr = np.array(verts_2d_r_all, dtype=np.float32)
    np.save(os.path.join(data_save_path, 'hamer_vertices_2d_left.npy'), verts_2d_l_arr)
    np.save(os.path.join(data_save_path, 'hamer_vertices_2d_right.npy'), verts_2d_r_arr)

    # Save face landmarks (optional)
    if use_face_mesh and kpts_face:
        kpts_face_arr = np.array(kpts_face)  # (nframes, 478, 5)
        n_detected = np.sum(kpts_face_arr[:, 0, 0] != -1)
        np.save(os.path.join(data_save_path, '2Dlandmarks_face.npy'), kpts_face_arr)
        print(f'  Cam {cam}: Face landmarks saved ({n_detected}/{len(kpts_face_arr)} frames detected)')

    # Save MANO face topology once at the trial level (shared across all cameras)
    from athena.hamer_hands import get_mano_faces
    faces_file = os.path.join(os.path.dirname(data_save_path), 'hamer_faces.npy')
    if not os.path.exists(faces_file):
        np.save(faces_file, get_mano_faces())

    # Release resources
    if hand_landmarker is not None:
        hand_landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()
    pose_landmarker.close()
    container.close()

    progress_queue.put({'cam': cam, 'done': True})
    return cam


def run_hybrid(input_streams, gui_options, cam_mats_intrinsic, cam_dist_coeffs, outdir_images_trial,
               outdir_data2d_trial, trialname, cam_mats_extrinsic=None,
               display_width=450, display_height=360, progress_queue=None):
    """
    Processes multiple camera streams using HaMeR for hands + MediaPipe for body.

    When using GPU-accelerated HaMeR, cameras are processed sequentially (1 process)
    to avoid GPU memory issues. On CPU, parallel processing is allowed.

    Parameters:
        input_streams (list): List of input video file paths.
        gui_options (dict): Dictionary containing GUI options and settings.
        cam_mats_intrinsic (list of np.ndarray): List of intrinsic camera matrices.
        cam_dist_coeffs (list of np.ndarray): List of camera distortion coefficients.
        outdir_images_trial (str): Output directory for images.
        outdir_data2d_trial (str): Output directory for data.
        trialname (str): Name of the trial.
        display_width (int, optional): Width for displaying frames.
        display_height (int, optional): Height for displaying frames.
        progress_queue (multiprocessing.Queue, optional): Queue for communicating progress.

    Returns:
        None
    """
    # HaMeR on GPU uses significant memory; limit to 1 process to avoid OOM
    if gui_options.get('use_gpu', False):
        num_processes = 1
    else:
        num_processes = gui_options.get('num_processes', os.cpu_count())

    # Precompute undistortion maps (same as run_mediapipe)
    undistort_maps = []
    for cam, input_stream in enumerate(input_streams):
        frame_count = 0
        cont = av.open(input_stream)
        for packet in cont.demux(video=0):
            for frame in packet.decode():
                frame_array = frame.to_ndarray(format='rgb24')
                frame_height, frame_width = frame_array.shape[:2]
                map1, map2 = cv.initUndistortRectifyMap(
                    cam_mats_intrinsic[cam],
                    cam_dist_coeffs[cam],
                    None,
                    cam_mats_intrinsic[cam],
                    (frame_width, frame_height),
                    cv.CV_16SC2
                )
                undistort_maps.append((map1, map2))
                frame_count += 1
                if frame_count == 1:
                    break
            if frame_count == 1:
                break
        cont.close()

    # Update gui_options
    gui_options['outdir_images_trial'] = outdir_images_trial
    gui_options['outdir_data2d_trial'] = outdir_data2d_trial
    gui_options['trialname'] = trialname

    # Clean gui_options for pickling
    gui_options_no_gui = gui_options.copy()
    gui_elements = ['fps_label', 'progress_bar', 'root', 'fps_value', 'progress_var']
    for key in gui_elements:
        gui_options_no_gui.pop(key, None)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for cam, input_stream in enumerate(input_streams):
            futures.append(executor.submit(
                process_camera_hybrid,
                cam,
                input_stream,
                gui_options_no_gui,
                cam_mats_intrinsic,
                cam_dist_coeffs,
                undistort_maps[cam],
                display_width,
                display_height,
                progress_queue
            ))

        for future in concurrent.futures.as_completed(futures):
            cam = future.result()
            print(f"Camera {cam} processing complete.")

    # Hand reassignment pass: triangulate body wrists to 3D, reproject to
    # every camera, and reassign left/right hands based on proximity.
    # This fixes cameras that lacked body detections and had wrong labels.
    if cam_mats_extrinsic is not None:
        print('Reassigning hand labels from triangulated 3D wrists...')
        _reassign_hands_from_3d_wrists(
            cam_mats_intrinsic, cam_mats_extrinsic, outdir_data2d_trial)

        # Redraw saved PNGs so that videos reflect post-reassignment labels
        if gui_options.get('save_images_mp', False):
            print('Redrawing saved images with corrected hand labels...')
            _redraw_saved_images(
                input_streams, outdir_images_trial, outdir_data2d_trial,
                cam_mats_intrinsic, cam_dist_coeffs,
                display_width=display_width, display_height=display_height,
                hand_backend='hamer',
                use_gpu=gui_options.get('use_gpu', False),
            )

    # Second pass: 3D-guided face detection.
    # After all cameras have body landmarks, triangulate the head to 3D
    # and reproject into every camera for robust face bboxes.
    if gui_options.get('use_face_mesh', False):
        print('Running 3D-guided face detection (second pass)...')
        _run_face_detection_pass(
            input_streams, gui_options_no_gui, cam_mats_intrinsic,
            cam_mats_extrinsic, cam_dist_coeffs,
            outdir_data2d_trial, undistort_maps,
        )


def main(gui_options_json):
    # Set the multiprocessing start method to 'spawn'
    set_start_method('spawn')

    gui_options = json.loads(gui_options_json)

    # Get the GUI options
    idfolders = gui_options['idfolders']
    main_folder = gui_options['main_folder']

    # Create the main root window for progress
    progress_root = tk.Tk()
    progress_root.title("Processing Progress")
    progress_root.attributes("-topmost", True)

    window_width = 500  # Width of the window
    window_height = 100  # Height of the window
    screen_width = progress_root.winfo_screenwidth()
    screen_height = progress_root.winfo_screenheight()
    position_x = (screen_width - window_width) // 2
    position_y = (screen_height - window_height) // 2
    progress_root.geometry(f'{window_width}x{window_height}+{position_x}+{position_y}')

    # Add progress bar and FPS label to progress_root
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        progress_root,
        orient="horizontal",
        mode="determinate",
        variable=progress_var,
        maximum=100
    )
    progress_bar.pack(pady=10, padx=10, fill=tk.X)

    fps_value = tk.DoubleVar()
    fps_label = tk.Label(progress_root, text="Avg FPS: 0")
    fps_label.pack(pady=5)

    # Create a Manager for shared objects
    manager = Manager()
    progress_queue = manager.Queue()
    cam_progress = manager.dict()  # For tracking progress per camera
    cam_fps = manager.dict()       # For tracking FPS per camera

    def update_progress():
        """
        Updates the progress bar and FPS label based on messages from the progress queue.
        """
        try:
            # Try to get messages from the queue without blocking
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                if 'progress' in progress and 'cam' in progress:
                    cam = progress['cam']
                    cam_progress[cam] = progress['progress']
                    # Calculate total progress
                    total_progress = sum(cam_progress.values()) / (len(cam_progress) * 100) * 100
                    progress_var.set(total_progress)
                    progress_bar["value"] = progress_var.get()
                if 'fps' in progress and 'cam' in progress:
                    cam = progress['cam']
                    fps = progress['fps']
                    cam_fps[cam] = fps
                    # Calculate average FPS
                    avg_fps = sum(cam_fps.values()) / len(cam_fps)
                    fps_value.set(avg_fps)
                    fps_label.config(text=f"Avg FPS: {fps_value.get():.2f}")
                if 'done' in progress:
                    if progress.get('cam') is not None:
                        cam = progress['cam']
                        cam_progress[cam] = 100
                    else:
                        fps_label.config(text="Processing Complete")
                        # Instead of quitting immediately, set a flag
                        update_progress.processing_done = True
        except Exception as e:
            print(f"Error in update_progress: {e}")
        if not update_progress.processing_done or not progress_queue.empty():
            # Schedule the function to run again after 100 milliseconds
            progress_root.after(100, update_progress)
        else:
            # All processing is done, and the queue is empty; now we can quit
            progress_root.quit()

    # Initialize the processing_done flag
    update_progress.processing_done = False

    def process_videos():
        """
        Processes the videos for each trial and updates the progress queue.
        """
        if idfolders:
            trialfolders = sorted(idfolders)
            outdir_images = os.path.join(main_folder, 'images/')
            outdir_video = os.path.join(main_folder, 'videos_processed/')
            outdir_data2d = os.path.join(main_folder, 'landmarks/')

            print(f"Selected Folder: {main_folder}")
            print(f"Save Images: {gui_options['save_images_mp']}")
            print(f"Save Video: {gui_options['save_video_mp']}")
            print(f"Use GPU: {gui_options['use_gpu']}")

            if gui_options['save_video_mp'] and not gui_options['save_images_mp']:
                print("Cannot save video without saving images. Adjusting settings.")
                gui_options['save_video_mp'] = False  # Adjust the setting in gui_options

            # Gather camera calibration parameters
            if glob.glob(os.path.join(main_folder, 'calibration', '*.yaml')):
                calfileext = '*.yaml'
            elif glob.glob(os.path.join(main_folder, 'calibration', '*.toml')):
                calfileext = '*.toml'
            calfiles = sorted(glob.glob(os.path.join(main_folder, 'calibration', calfileext)))
            cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = read_calibration(calfiles, calfileext)

            total_trials = len(trialfolders)
            processed_trials = 0

            for trial in trialfolders:
                trialname = os.path.basename(trial)
                print(f"Processing trial: {trialname}")

                vidnames = sorted(glob.glob(os.path.join(trial, '*.avi')) + glob.glob(os.path.join(trial, '*.mp4')))
                ncams = len(vidnames)

                container = av.open(vidnames[0])
                video_stream = container.streams.video[0]
                # Get video FPS and total frames
                if video_stream.average_rate is not None and video_stream.average_rate.denominator != 0:
                    fps = video_stream.average_rate.numerator / video_stream.average_rate.denominator
                else:
                    fps = 30.0  # Default FPS if not available
                container.close()

                outdir_images_trial = os.path.join(outdir_images, trialname)
                outdir_video_trial = os.path.join(outdir_video, trialname)
                outdir_data2d_trial = os.path.join(outdir_data2d, trialname)

                os.makedirs(outdir_data2d_trial, exist_ok=True)
                if gui_options['save_images_mp']:
                    os.makedirs(outdir_images_trial, exist_ok=True)
                    for cam in range(ncams):
                        os.makedirs(os.path.join(outdir_images_trial, f'cam{cam}'), exist_ok=True)

                # Initialize cam_progress and cam_fps for each camera
                for cam in range(ncams):
                    cam_progress[cam] = 0.0
                    cam_fps[cam] = 0.0

                # Dispatch to appropriate backend
                hand_backend = gui_options.get('hand_backend', 'mediapipe')
                if hand_backend == 'hamer':
                    print('Using HaMeR for hand detection.')
                    run_hybrid(
                        vidnames,
                        gui_options,
                        cam_mats_intrinsic,
                        cam_dist_coeffs,
                        outdir_images_trial,
                        outdir_data2d_trial,
                        trialname,
                        cam_mats_extrinsic=cam_mats_extrinsic,
                        progress_queue=progress_queue
                    )
                else:
                    run_mediapipe(
                        vidnames,
                        gui_options,
                        cam_mats_intrinsic,
                        cam_dist_coeffs,
                        outdir_images_trial,
                        outdir_data2d_trial,
                        trialname,
                        cam_mats_extrinsic=cam_mats_extrinsic,
                        progress_queue=progress_queue
                    )

                # Save detection method metadata for downstream code
                meta = {'hand_backend': hand_backend}
                meta_path = os.path.join(outdir_data2d_trial, 'detection_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)

                if gui_options['save_images_mp'] and gui_options['save_video_mp']:
                    os.makedirs(outdir_video_trial, exist_ok=True)
                    video_suffix = 'hamer' if hand_backend == 'hamer' else 'mediapipe'
                    for cam in range(ncams):
                        imagefolder = os.path.join(outdir_images_trial, f'cam{cam}')
                        create_video(
                            image_folder=imagefolder,
                            extension='.png',
                            fps=fps,
                            output_folder=outdir_video_trial,
                            video_name=f'cam{cam}_{video_suffix}.mp4'
                        )

                # Update progress per trial
                processed_trials += 1
                total_progress = (processed_trials / total_trials) * 100
                progress_queue.put({'progress': total_progress})

            # When done, put 'done' in the queue
            print("Processing complete, putting 'done' into queue")
            progress_queue.put({'done': True})

    # Start the processing in a separate thread
    threading.Thread(target=process_videos).start()

    # Start updating the progress window
    update_progress()

    # Start the Tkinter main loop for the progress window
    progress_root.mainloop()