#!/usr/bin/env python3
"""Full reprojection test for face mesh landmarks.

Loads calibration + 2D face data from the test dataset, triangulates face
landmarks, reprojects back to every camera, and reports per-camera
reprojection errors.  Also checks alignment of body-pose head landmarks
vs face mesh landmarks in both 2D and 3D.
"""

import os, sys, glob
import numpy as np
import cv2 as cv

# ── Paths ───────────────────────────────────────────────────────────────
DATA_ROOT = os.path.expanduser('~/Desktop/ATHENA-tests/Markerless')
TRIAL = 'Recording_2024-11-08T111849'
CAL_DIR = os.path.join(DATA_ROOT, 'calibration')
LANDMARK_DIR = os.path.join(DATA_ROOT, 'landmarks', TRIAL)

# ── Helpers from the pipeline ───────────────────────────────────────────
def transformationmatrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def readcalibration_yaml(cal_dir):
    files = sorted(glob.glob(os.path.join(cal_dir, '*.yaml')))
    extrinsics, intrinsics, dist_coeffs = [], [], []
    for f in files:
        fs = cv.FileStorage(f, cv.FILE_STORAGE_READ)
        K = fs.getNode("intrinsicMatrix").mat().T
        D = fs.getNode("distortionCoefficients").mat().reshape(-1)
        R = fs.getNode("R").mat().T
        t = fs.getNode("T").mat()
        extrinsics.append(transformationmatrix(R, t))
        intrinsics.append(K)
        dist_coeffs.append(D)
    return np.array(extrinsics), intrinsics, dist_coeffs


def undistort_points(pts, K, D):
    """Undistort and normalize. pts: (N,2) → (N,2) normalized camera coords."""
    return cv.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64),
                              K, D).reshape(-1, 2)


def triangulate_batch(points_2d, cam_mats_extrinsic):
    """DLT triangulation.  points_2d: (ncams, npoints, 2) normalized coords."""
    ncams, npoints, _ = points_2d.shape
    data3d = np.full((npoints, 3), np.nan)
    good = ~np.isnan(points_2d[:, :, 0])
    patterns = np.zeros(npoints, dtype=np.int32)
    for c in range(ncams):
        patterns += good[c].astype(np.int32) << c
    for pat in np.unique(patterns):
        if bin(pat).count('1') < 2:
            continue
        active = [c for c in range(ncams) if (pat >> c) & 1]
        idx = np.where(patterns == pat)[0]
        mats = cam_mats_extrinsic[active]
        pts = points_2d[active][:, idx, :]
        A = np.zeros((len(idx), len(active) * 2, 4))
        for i in range(len(active)):
            x, y = pts[i, :, 0], pts[i, :, 1]
            m = mats[i]
            A[:, 2*i]   = x[:, None] * m[2][None, :] - m[0][None, :]
            A[:, 2*i+1] = y[:, None] * m[2][None, :] - m[1][None, :]
        _, _, vh = np.linalg.svd(A, full_matrices=True)
        p = vh[:, -1, :]
        data3d[idx] = p[:, :3] / p[:, 3:4]
    return data3d


def reproject(pts_3d, K, ext):
    """Reproject 3D points to 2D pixel coords. pts_3d: (N,3), returns (N,2)."""
    h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    cam = ext @ h.T  # (4,N) or (3,N)
    img = K @ cam[:3]
    uv = img[:2] / img[2:3]
    return uv.T


# ── Load data ───────────────────────────────────────────────────────────
print('Loading calibration...')
cam_mats_ext, cam_mats_int, cam_dist = readcalibration_yaml(CAL_DIR)
ncams = len(cam_mats_int)
print(f'  {ncams} cameras')

print('Loading 2D face landmarks...')
face_2d = []
for cam in range(ncams):
    f = os.path.join(LANDMARK_DIR, f'cam{cam}', '2Dlandmarks_face.npy')
    face_2d.append(np.load(f).astype(float))
face_2d = np.stack(face_2d)  # (ncams, nframes, 478, 5)
nframes = face_2d.shape[1]
nlandmarks_face = face_2d.shape[2]
print(f'  Shape: {face_2d.shape}')

# Detection stats
for cam in range(ncams):
    detected = np.sum(face_2d[cam, :, 0, 0] != -1)
    print(f'  cam{cam}: {detected}/{nframes} frames with face ({100*detected/nframes:.1f}%)')

# Count multi-camera overlap
overlap = np.sum(face_2d[:, :, 0, 0] != -1, axis=0)
for n in range(ncams+1):
    cnt = np.sum(overlap == n)
    if cnt > 0:
        print(f'  {n} cameras: {cnt} frames ({100*cnt/nframes:.1f}%)')

print('\nLoading body landmarks...')
body_2d = []
for cam in range(ncams):
    f = os.path.join(LANDMARK_DIR, f'cam{cam}', '2Dlandmarks_body.npy')
    body_2d.append(np.load(f).astype(float))
body_2d = np.stack(body_2d)  # (ncams, nframes, 33, 5)

# ── Sentinel → NaN ──────────────────────────────────────────────────────
face_xy = face_2d[:, :, :, :2].copy()  # (ncams, nframes, 478, 2)
sentinel = (face_xy[:, :, :, 0] == -1) & (face_xy[:, :, :, 1] == -1)
face_xy[sentinel] = np.nan

body_xy = body_2d[:, :, :, :2].copy()
sentinel_b = (body_xy[:, :, :, 0] == -1) & (body_xy[:, :, :, 1] == -1)
body_xy[sentinel_b] = np.nan

# ── Undistort + triangulate face (subset of landmarks for speed) ────────
# Use a representative subset of face landmarks for the reprojection test.
# Landmarks: 1 (nose tip), 33 (left eyebrow inner), 263 (right eyebrow inner),
# 61 (left mouth), 291 (right mouth), 10 (forehead top), 152 (chin), 234 (left contour), 454 (right contour)
TEST_LMS = [1, 10, 33, 61, 152, 234, 263, 291, 454]
print(f'\n=== FACE LANDMARK REPROJECTION TEST ({len(TEST_LMS)} landmarks) ===')

# Undistort and normalize - points are already in undistorted pixel space
# (frames were undistorted via remap before detection), so use dist=0
face_flat = face_xy[:, :, TEST_LMS, :].reshape(ncams, -1, 2)  # (ncams, nframes*n_test, 2)
face_norm = np.empty_like(face_flat)
for cam in range(ncams):
    pts = face_flat[cam]
    nan_mask = np.isnan(pts[:, 0])
    out = np.full_like(pts, np.nan)
    if (~nan_mask).any():
        out[~nan_mask] = undistort_points(pts[~nan_mask], cam_mats_int[cam], np.zeros(5))
    face_norm[cam] = out

# Triangulate
pts_3d = triangulate_batch(face_norm, cam_mats_ext)  # (nframes*n_test, 3)
pts_3d_reshaped = pts_3d.reshape(nframes, len(TEST_LMS), 3)

valid_3d = ~np.isnan(pts_3d_reshaped[:, :, 0])
print(f'  Valid 3D points: {valid_3d.sum()} / {valid_3d.size}')
print(f'  Frames with any valid face 3D: {np.any(valid_3d, axis=1).sum()}/{nframes}')
print(f'  Frames with ALL {len(TEST_LMS)} landmarks valid: {np.all(valid_3d, axis=1).sum()}/{nframes}')

# ── Reprojection error ──────────────────────────────────────────────────
print('\n--- Per-camera reprojection error (pixels) ---')
all_errors = []
for cam in range(ncams):
    K = cam_mats_int[cam]
    ext = cam_mats_ext[cam]

    # Get original 2D observations for this camera (undistorted pixel space)
    obs_2d = face_xy[cam, :, TEST_LMS, :].reshape(-1, 2)  # (nframes*n_test, 2)

    # Only evaluate where we have both observation and 3D
    valid = ~np.isnan(obs_2d[:, 0]) & ~np.isnan(pts_3d[:, 0])

    if valid.sum() == 0:
        print(f'  cam{cam}: no valid points to evaluate')
        continue

    # Reproject
    reproj = reproject(pts_3d[valid], K, ext)

    # Error
    err = np.linalg.norm(reproj - obs_2d[valid], axis=1)
    all_errors.extend(err.tolist())

    print(f'  cam{cam}: n={valid.sum():5d}  '
          f'mean={err.mean():7.2f}  median={np.median(err):7.2f}  '
          f'p95={np.percentile(err,95):7.2f}  max={err.max():7.2f}')

if all_errors:
    all_errors = np.array(all_errors)
    print(f'\n  OVERALL: n={len(all_errors)}  '
          f'mean={all_errors.mean():.2f}  median={np.median(all_errors):.2f}  '
          f'p95={np.percentile(all_errors,95):.2f}  max={all_errors.max():.2f}')

# ── Also test reprojection with filtering (drop high-error points) ──────
print('\n--- Reprojection with outlier filtering ---')
THRESHOLD = 30  # pixels
for cam in range(ncams):
    K = cam_mats_int[cam]
    ext = cam_mats_ext[cam]
    obs_2d = face_xy[cam, :, TEST_LMS, :].reshape(-1, 2)
    valid = ~np.isnan(obs_2d[:, 0]) & ~np.isnan(pts_3d[:, 0])
    if valid.sum() == 0:
        continue
    reproj = reproject(pts_3d[valid], K, ext)
    err = np.linalg.norm(reproj - obs_2d[valid], axis=1)
    inlier = err < THRESHOLD
    if inlier.sum() > 0:
        print(f'  cam{cam}: inliers={inlier.sum()}/{valid.sum()} ({100*inlier.mean():.0f}%)  '
              f'mean={err[inlier].mean():.2f}  median={np.median(err[inlier]):.2f}')
    else:
        print(f'  cam{cam}: 0 inliers below {THRESHOLD}px')

# ── Body-head vs face-mesh alignment test ───────────────────────────────
print('\n=== BODY-HEAD vs FACE-MESH ALIGNMENT ===')
# Body nose = landmark 0, Face nose tip = landmark 1
body_nose_2d = body_xy[:, :, 0, :]  # (ncams, nframes, 2)
face_nose_2d = face_xy[:, :, 1, :]  # (ncams, nframes, 2)

for cam in range(ncams):
    both_valid = ~np.isnan(body_nose_2d[cam, :, 0]) & ~np.isnan(face_nose_2d[cam, :, 0])
    if both_valid.sum() == 0:
        print(f'  cam{cam}: no frames with both body nose and face nose')
        continue
    dist = np.linalg.norm(body_nose_2d[cam, both_valid] - face_nose_2d[cam, both_valid], axis=1)
    print(f'  cam{cam}: n={both_valid.sum():4d}  '
          f'mean_dist={dist.mean():.1f}px  median={np.median(dist):.1f}px  max={dist.max():.1f}px')

# ── 3D alignment: body nose vs face nose ────────────────────────────────
print('\n--- 3D body nose vs face nose ---')
# Triangulate body nose (landmark 0) and face nose (landmark 1)
body_nose_flat = body_xy[:, :, 0, :].reshape(ncams, nframes, 2)
body_norm = np.empty((ncams, nframes, 2))
for cam in range(ncams):
    pts = body_nose_flat[cam]
    nan_mask = np.isnan(pts[:, 0])
    out = np.full_like(pts, np.nan)
    if (~nan_mask).any():
        out[~nan_mask] = undistort_points(pts[~nan_mask], cam_mats_int[cam], np.zeros(5))
    body_norm[cam] = out

body_nose_3d = triangulate_batch(body_norm, cam_mats_ext)  # (nframes, 3)
face_nose_3d = pts_3d_reshaped[:, 0, :]  # landmark 1 is TEST_LMS[0]=1

both_valid = ~np.isnan(body_nose_3d[:, 0]) & ~np.isnan(face_nose_3d[:, 0])
if both_valid.sum() > 0:
    dist_3d = np.linalg.norm(body_nose_3d[both_valid] - face_nose_3d[both_valid], axis=1)
    offset = np.nanmean(body_nose_3d[both_valid] - face_nose_3d[both_valid], axis=0)
    print(f'  n_frames={both_valid.sum()}')
    print(f'  mean 3D distance: {dist_3d.mean():.1f} mm')
    print(f'  median 3D distance: {np.median(dist_3d):.1f} mm')
    print(f'  systematic offset (body-face): X={offset[0]:.1f} Y={offset[1]:.1f} Z={offset[2]:.1f} mm')
else:
    print('  No frames with both body and face nose in 3D')

# ── Head bbox quality check ─────────────────────────────────────────────
print('\n=== HEAD BBOX QUALITY CHECK ===')
# Triangulate all 11 head landmarks to 3D, reproject, check face nose is inside bbox
HEAD_LMS = list(range(11))
head_2d_all = body_xy[:, :, HEAD_LMS, :]  # (ncams, nframes, 11, 2)
head_flat = head_2d_all.reshape(ncams, -1, 2)
head_norm = np.empty_like(head_flat)
for cam in range(ncams):
    pts = head_flat[cam]
    nan_mask = np.isnan(pts[:, 0])
    out = np.full_like(pts, np.nan)
    if (~nan_mask).any():
        out[~nan_mask] = undistort_points(pts[~nan_mask], cam_mats_int[cam], np.zeros(5))
    head_norm[cam] = out

head_3d = triangulate_batch(head_norm, cam_mats_ext)
head_3d = head_3d.reshape(nframes, 11, 3)
head_centre = np.nanmean(head_3d, axis=1)  # (nframes, 3)
head_span = np.nanmax(head_3d, axis=1) - np.nanmin(head_3d, axis=1)
head_radius = np.nanmax(head_span, axis=1) / 2.0

valid_head = ~np.isnan(head_centre[:, 0])
print(f'  Head triangulated: {valid_head.sum()}/{nframes} frames')

# Compute bboxes and check face nose is inside
margin_factor = 2.0
min_crop_px = 100
for cam in range(ncams):
    K = cam_mats_int[cam]
    ext = cam_mats_ext[cam]

    pts_h = np.hstack([head_centre[valid_head], np.ones((valid_head.sum(), 1))])
    pts_off = head_centre[valid_head].copy()
    pts_off[:, 0] += head_radius[valid_head]
    pts_off_h = np.hstack([pts_off, np.ones((valid_head.sum(), 1))])

    X_img = K @ (ext @ pts_h.T)[:3]
    uv = (X_img[:2] / X_img[2:3]).T
    X_img_off = K @ (ext @ pts_off_h.T)[:3]
    uv_off = (X_img_off[:2] / X_img_off[2:3]).T
    px_rad = np.linalg.norm(uv_off - uv, axis=1)
    half = np.maximum(px_rad * margin_factor, min_crop_px / 2.0)

    bboxes = np.column_stack([uv[:, 0] - half, uv[:, 1] - half,
                               uv[:, 0] + half, uv[:, 1] + half])

    # Check face nose (from face_xy) falls inside bbox
    face_nose = face_xy[cam, valid_head, 1, :]  # (n_valid, 2)
    has_face = ~np.isnan(face_nose[:, 0])

    if has_face.sum() > 0:
        fn = face_nose[has_face]
        bb = bboxes[has_face]
        inside = ((fn[:, 0] >= bb[:, 0]) & (fn[:, 0] <= bb[:, 2]) &
                  (fn[:, 1] >= bb[:, 1]) & (fn[:, 1] <= bb[:, 3]))

        # Distance from bbox centre
        cx = (bb[:, 0] + bb[:, 2]) / 2
        cy = (bb[:, 1] + bb[:, 3]) / 2
        dist_from_centre = np.sqrt((fn[:, 0] - cx)**2 + (fn[:, 1] - cy)**2)
        bbox_half_size = (bb[:, 2] - bb[:, 0]) / 2

        print(f'  cam{cam}: face in bbox {inside.sum()}/{has_face.sum()} ({100*inside.mean():.0f}%)  '
              f'dist_from_centre: mean={dist_from_centre.mean():.0f}px  '
              f'bbox_half={bbox_half_size.mean():.0f}px')
    else:
        print(f'  cam{cam}: no face detections where head is triangulated')

print('\n=== BODY LANDMARK REPROJECTION TEST (sanity check) ===')
# Quick sanity: triangulate + reproject body nose to check calibration is good
body_nose_norm = np.empty((ncams, nframes, 2))
for cam in range(ncams):
    pts = body_xy[cam, :, 0, :]
    nan_mask = np.isnan(pts[:, 0])
    out = np.full_like(pts, np.nan)
    if (~nan_mask).any():
        out[~nan_mask] = undistort_points(pts[~nan_mask], cam_mats_int[cam], np.zeros(5))
    body_nose_norm[cam] = out

body_nose_3d_check = triangulate_batch(body_nose_norm, cam_mats_ext)

for cam in range(ncams):
    K = cam_mats_int[cam]
    ext = cam_mats_ext[cam]
    obs = body_xy[cam, :, 0, :]
    valid = ~np.isnan(obs[:, 0]) & ~np.isnan(body_nose_3d_check[:, 0])
    if valid.sum() == 0:
        print(f'  cam{cam}: no valid body nose')
        continue
    reproj = reproject(body_nose_3d_check[valid], K, ext)
    err = np.linalg.norm(reproj - obs[valid], axis=1)
    print(f'  cam{cam}: n={valid.sum():5d}  mean={err.mean():.2f}px  median={np.median(err):.2f}px  p95={np.percentile(err,95):.2f}px')

print('\nDone.')
