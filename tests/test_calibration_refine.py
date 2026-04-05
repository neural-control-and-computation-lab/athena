"""Tests for athena.calibration_refine — dynamic camera calibration optimisation.

All tests use synthetic cameras and 2D observations; no real data needed.
"""

import numpy as np
import cv2 as cv
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intrinsic(fx=500, fy=500, cx=320, cy=240):
    """Create a simple pinhole intrinsic matrix."""
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def _make_extrinsic(rvec, tvec):
    """Create a 4x4 extrinsic matrix from axis-angle rotation and translation."""
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).ravel())
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def _make_camera_rig(ncams=6, radius=500):
    """Create a synthetic multi-camera rig arranged in a circle looking inward.

    Returns (extrinsics, intrinsics) — list of 4x4 and list of 3x3.
    """
    intrinsics = [_make_intrinsic() for _ in range(ncams)]
    extrinsics = []
    for i in range(ncams):
        angle = 2 * np.pi * i / ncams
        # Camera position on a circle
        tx = radius * np.cos(angle)
        ty = radius * np.sin(angle)
        tz = 0.0
        # Look toward origin: camera z-axis points inward
        # Simple approach: rotation about y then adjust
        cam_pos = np.array([tx, ty, tz])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, -up, forward], axis=0)  # camera axes
        t = -R @ cam_pos
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        extrinsics.append(T)
    return np.array(extrinsics), intrinsics


def _generate_synthetic_data(extrinsics, intrinsics, n_points=42,
                             nframes=100, noise_px=2.0, seed=42):
    """Generate synthetic 2D observations from known 3D points.

    Returns (data_2d, points_3d_true).
    data_2d shape: (ncams, nframes, nlandmarks, 2) in pixel space.
    We place hand-landmark-like data at indices 33-74 (42 landmarks).
    """
    rng = np.random.default_rng(seed)
    ncams = len(intrinsics)
    nlandmarks = 75  # body(33) + right hand(21) + left hand(21)

    # Generate 3D points in a cluster near origin (hand-like)
    # Different points per frame to simulate movement
    points_3d = np.zeros((nframes, n_points, 3))
    for f in range(nframes):
        centre = np.array([0, 0, 0]) + rng.normal(0, 20, 3) * (f / nframes)
        points_3d[f] = centre + rng.normal(0, 30, (n_points, 3))

    # Project to 2D for each camera
    data_2d = np.full((ncams, nframes, nlandmarks, 2), -1.0)

    for cam in range(ncams):
        K = np.asarray(intrinsics[cam])
        E = extrinsics[cam][:3]
        P = K @ E

        for f in range(nframes):
            for lm_local in range(n_points):
                lm_global = 33 + lm_local  # hand landmarks start at 33
                pt = np.append(points_3d[f, lm_local], 1.0)
                proj = P @ pt
                if proj[2] <= 0:
                    continue
                u = proj[0] / proj[2] + rng.normal(0, noise_px)
                v = proj[1] / proj[2] + rng.normal(0, noise_px)
                # Check if within image bounds
                if 0 <= u <= 640 and 0 <= v <= 480:
                    data_2d[cam, f, lm_global, 0] = u
                    data_2d[cam, f, lm_global, 1] = v

    return data_2d, points_3d


def _perturb_extrinsics(extrinsics, rotation_deg=2.0, translation_mm=15.0,
                        ref_cam=0, seed=123):
    """Add noise to extrinsic parameters (except reference camera)."""
    rng = np.random.default_rng(seed)
    perturbed = extrinsics.copy()
    for cam in range(len(extrinsics)):
        if cam == ref_cam:
            continue
        # Perturb rotation
        R = extrinsics[cam][:3, :3]
        rvec = cv.Rodrigues(R)[0].ravel()
        rvec += rng.normal(0, np.radians(rotation_deg), 3)
        R_new = cv.Rodrigues(rvec)[0]
        perturbed[cam][:3, :3] = R_new
        # Perturb translation
        perturbed[cam][:3, 3] += rng.normal(0, translation_mm, 3)
    return perturbed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRotationHelpers:
    def test_roundtrip(self):
        from athena.calibration_refine import rodrigues_to_matrix, matrix_to_rodrigues
        rvec_in = np.array([0.1, -0.2, 0.3])
        R = rodrigues_to_matrix(rvec_in)
        rvec_out = matrix_to_rodrigues(R)
        # Rodrigues vectors can differ by sign of full rotation, but
        # the resulting matrix should be the same
        R2 = rodrigues_to_matrix(rvec_out)
        assert_allclose(R, R2, atol=1e-10)

    def test_identity(self):
        from athena.calibration_refine import rodrigues_to_matrix
        R = rodrigues_to_matrix([0, 0, 0])
        assert_allclose(R, np.eye(3), atol=1e-10)


class TestPackUnpack:
    def test_roundtrip(self):
        from athena.calibration_refine import _pack_extrinsics, _unpack_extrinsics
        extrinsics, _ = _make_camera_rig(4)
        ref_cam = 0
        params = _pack_extrinsics(extrinsics, ref_cam)
        assert params.shape == (6 * 3,)  # 4 cams - 1 ref = 3
        recovered = _unpack_extrinsics(params, extrinsics[ref_cam], 4, ref_cam)
        assert_allclose(recovered, extrinsics, atol=1e-8)


class TestSelectObservations:
    def test_basic_selection(self):
        from athena.calibration_refine import _select_observations
        ncams, nframes, nlandmarks = 4, 50, 75
        # Create data where hand landmarks are visible in all cameras
        data_2d = np.full((ncams, nframes, nlandmarks, 2), -1.0)
        rng = np.random.default_rng(0)
        for cam in range(ncams):
            for f in range(nframes):
                for lm in range(33, 75):
                    data_2d[cam, f, lm] = rng.uniform(10, 630, 2)

        obs, keys = _select_observations(data_2d, ncams, nframes, nlandmarks,
                                         subsample=10, min_cameras=3)
        assert len(obs) > 0
        assert len(keys) > 0
        # Each observation should reference a valid point
        for cam, pt_idx, u, v in obs:
            assert 0 <= cam < ncams
            assert 0 <= pt_idx < len(keys)
            assert u > 0 and v > 0


class TestDriftDetection:
    def test_no_drift_for_identical(self):
        from athena.calibration_refine import _compute_drift
        extrinsics, _ = _make_camera_rig(4)
        drift = _compute_drift(extrinsics, extrinsics, ref_cam=0)
        for d in drift:
            assert d['rotation_deg'] < 0.01
            assert d['translation_mm'] < 0.01

    def test_detects_rotation(self):
        from athena.calibration_refine import _compute_drift
        extrinsics, _ = _make_camera_rig(4)
        perturbed = _perturb_extrinsics(extrinsics, rotation_deg=5.0,
                                         translation_mm=0.0)
        drift = _compute_drift(extrinsics, perturbed, ref_cam=0)
        # Non-reference cameras should show rotation drift
        for cam in range(1, 4):
            assert drift[cam]['rotation_deg'] > 1.0

    def test_detects_translation(self):
        from athena.calibration_refine import _compute_drift
        extrinsics, _ = _make_camera_rig(4)
        perturbed = _perturb_extrinsics(extrinsics, rotation_deg=0.0,
                                         translation_mm=30.0)
        drift = _compute_drift(extrinsics, perturbed, ref_cam=0)
        for cam in range(1, 4):
            assert drift[cam]['translation_mm'] > 5.0


class TestBundleAdjustment:
    def test_refine_window_reduces_error(self):
        """Perturbed extrinsics should be improved by bundle adjustment."""
        from athena.calibration_refine import _refine_window

        extrinsics_true, intrinsics = _make_camera_rig(6)
        data_2d, _ = _generate_synthetic_data(extrinsics_true, intrinsics,
                                               nframes=200, noise_px=1.5)
        extrinsics_perturbed = _perturb_extrinsics(extrinsics_true,
                                                     rotation_deg=1.5,
                                                     translation_mm=10.0)
        ncams = 6
        nframes = 200
        nlandmarks = 75

        refined, err_before, err_after, n_obs = _refine_window(
            extrinsics_perturbed, intrinsics, data_2d,
            ncams, nframes, nlandmarks,
            subsample=5, min_cameras=3,
            reg_weight=0.001, verbose=False,
        )

        assert n_obs > 50, f"Too few observations: {n_obs}"
        assert not np.isnan(err_before)
        assert not np.isnan(err_after)
        assert err_after < err_before, (
            f"Expected improvement: {err_before:.2f} -> {err_after:.2f}")

    def test_perfect_calibration_unchanged(self):
        """With correct extrinsics, refinement should not degrade quality."""
        from athena.calibration_refine import _refine_window

        extrinsics_true, intrinsics = _make_camera_rig(6)
        data_2d, _ = _generate_synthetic_data(extrinsics_true, intrinsics,
                                               nframes=200, noise_px=1.5)
        ncams = 6
        nframes = 200
        nlandmarks = 75

        refined, err_before, err_after, n_obs = _refine_window(
            extrinsics_true, intrinsics, data_2d,
            ncams, nframes, nlandmarks,
            subsample=5, min_cameras=3,
            reg_weight=0.01, verbose=False,
        )

        # Error should stay similar (not get worse)
        if not np.isnan(err_after):
            assert err_after <= err_before + 0.5, (
                f"Degraded: {err_before:.2f} -> {err_after:.2f}")


class TestRefineCalibration:
    def test_sliding_window_integration(self):
        """Full sliding-window refinement should reduce reprojection error."""
        from athena.calibration_refine import refine_calibration

        extrinsics_true, intrinsics = _make_camera_rig(6)
        data_2d, _ = _generate_synthetic_data(extrinsics_true, intrinsics,
                                               nframes=300, noise_px=1.5)
        extrinsics_perturbed = _perturb_extrinsics(extrinsics_true,
                                                     rotation_deg=1.5,
                                                     translation_mm=10.0)
        nframes = 300
        nlandmarks = 75

        refined, drift_report = refine_calibration(
            extrinsics_perturbed, intrinsics, data_2d,
            nframes, nlandmarks,
            window_size=200, window_overlap=50,
            subsample=5, reg_weight=0.001,
            verbose=False,
        )

        assert len(drift_report) > 0
        # At least one window should show improvement
        improved = [r for r in drift_report
                    if not np.isnan(r['reproj_error_after'])
                    and r['reproj_error_after'] < r['reproj_error_before']]
        assert len(improved) > 0

    def test_drift_detection_flags_bump(self):
        """A large perturbation in one window should be flagged."""
        from athena.calibration_refine import refine_calibration

        extrinsics_true, intrinsics = _make_camera_rig(4)
        data_2d, _ = _generate_synthetic_data(extrinsics_true, intrinsics,
                                               n_points=42, nframes=200,
                                               noise_px=1.5)
        # Large perturbation to simulate a camera bump
        extrinsics_bumped = _perturb_extrinsics(extrinsics_true,
                                                  rotation_deg=5.0,
                                                  translation_mm=50.0)
        nframes = 200
        nlandmarks = 75

        _, drift_report = refine_calibration(
            extrinsics_bumped, intrinsics, data_2d,
            nframes, nlandmarks,
            window_size=200, window_overlap=50,
            subsample=5, reg_weight=0.001,
            drift_rotation_thresh=2.0,
            drift_translation_thresh=20.0,
            verbose=False,
        )

        # Should flag at least some cameras
        all_flagged = set()
        for r in drift_report:
            all_flagged.update(r['flagged_cameras'])
        assert len(all_flagged) > 0, "Expected drift detection to flag cameras"
