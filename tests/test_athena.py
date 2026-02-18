"""Comprehensive test suite for the ATHENA package.

Tests cover:
  - athena.visualization: hex_to_bgr, skeleton constants, draw_landmarks_unified,
    render_mesh_overlay
  - athena.labels2d: transformation_matrix, rotation_matrix, importability of
    read_calibration and create_video
  - athena.triangulaterefine: _undistort_points, _triangulate_batch,
    _batch_reproject, _triangulate_with_filtering, _smooth3d,
    _restore_long_nan_runs
  - Package structure: __version__, module imports

All tests use synthetic data and do not require real video files, calibration
files, or GPU hardware.

Dependencies:
  - athena.visualization requires only numpy and opencv-python (always available).
  - athena.labels2d additionally requires av, mediapipe, toml, and tkinter.
  - athena.triangulaterefine additionally requires av, tqdm, and scipy.
  Tests for modules with heavy dependencies are skipped when those packages are
  not installed.
"""

import importlib
import numpy as np
import cv2 as cv
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# Helpers: conditional module imports
# ---------------------------------------------------------------------------

def _can_import(module_name):
    """Return True if *module_name* can be imported without error."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


# Build skip markers for modules with heavy external dependencies.
_has_labels2d = _can_import("athena.labels2d")
_has_triangulaterefine = _can_import("athena.triangulaterefine")

skip_labels2d = pytest.mark.skipif(
    not _has_labels2d,
    reason="athena.labels2d requires av, mediapipe, toml (not installed)",
)
skip_triangulaterefine = pytest.mark.skipif(
    not _has_triangulaterefine,
    reason="athena.triangulaterefine requires av, tqdm, scipy (not installed)",
)


# ============================================================================
# Package structure
# ============================================================================


class TestPackageStructure:
    """Verify package-level attributes and module importability."""

    def test_version_exists_and_is_string(self):
        """athena.__version__ must exist and be a non-empty string."""
        import athena
        assert hasattr(athena, "__version__")
        assert isinstance(athena.__version__, str)
        assert len(athena.__version__) > 0

    def test_import_visualization(self):
        """athena.visualization should be importable."""
        import athena.visualization  # noqa: F401

    @skip_labels2d
    def test_import_labels2d(self):
        """athena.labels2d should be importable (requires av, mediapipe)."""
        import athena.labels2d  # noqa: F401

    @skip_triangulaterefine
    def test_import_triangulaterefine(self):
        """athena.triangulaterefine should be importable (requires av, scipy)."""
        import athena.triangulaterefine  # noqa: F401


# ============================================================================
# athena.visualization
# ============================================================================


class TestHexToBgr:
    """Tests for the hex_to_bgr colour conversion function."""

    def test_red(self):
        """#FF0000 (pure red in RGB) should map to (0, 0, 255) in BGR."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#FF0000") == (0, 0, 255)

    def test_green(self):
        """#00FF00 (pure green in RGB) should map to (0, 255, 0) in BGR."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#00FF00") == (0, 255, 0)

    def test_blue(self):
        """#0000FF (pure blue in RGB) should map to (255, 0, 0) in BGR."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#0000FF") == (255, 0, 0)

    def test_white(self):
        """#FFFFFF should map to (255, 255, 255)."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#FFFFFF") == (255, 255, 255)

    def test_black(self):
        """#000000 should map to (0, 0, 0)."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#000000") == (0, 0, 0)

    def test_arbitrary_colour(self):
        """#EEDE33 -> R=238, G=222, B=51 -> BGR=(51, 222, 238)."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#EEDE33") == (51, 222, 238)

    def test_lowercase_hex(self):
        """Lowercase hex digits should work identically to uppercase."""
        from athena.visualization import hex_to_bgr
        assert hex_to_bgr("#ff0000") == (0, 0, 255)

    def test_without_hash(self):
        """hex_to_bgr should handle input with or without leading '#'."""
        from athena.visualization import hex_to_bgr
        # The lstrip('#') in the implementation handles both forms
        assert hex_to_bgr("FF0000") == (0, 0, 255)


class TestSkeletonConstants:
    """Tests for skeleton link and colour constants."""

    def test_skeleton_links_length(self):
        """SKELETON_LINKS should contain exactly 56 links."""
        from athena.visualization import SKELETON_LINKS
        assert len(SKELETON_LINKS) == 56

    def test_first_16_are_body_links(self):
        """The first 16 entries in SKELETON_LINKS should match BODY_LINKS."""
        from athena.visualization import SKELETON_LINKS, BODY_LINKS
        assert SKELETON_LINKS[:16] == BODY_LINKS

    def test_body_links_length(self):
        """BODY_LINKS should have exactly 16 elements."""
        from athena.visualization import BODY_LINKS
        assert len(BODY_LINKS) == 16

    def test_skeleton_colours_hex_length(self):
        """SKELETON_COLOURS_HEX must have the same length as SKELETON_LINKS."""
        from athena.visualization import SKELETON_LINKS, SKELETON_COLOURS_HEX
        assert len(SKELETON_COLOURS_HEX) == len(SKELETON_LINKS)

    def test_skeleton_colours_bgr_length(self):
        """SKELETON_COLOURS_BGR must have the same length as SKELETON_LINKS."""
        from athena.visualization import SKELETON_LINKS, SKELETON_COLOURS_BGR
        assert len(SKELETON_COLOURS_BGR) == len(SKELETON_LINKS)

    def test_skeleton_colours_hex_bgr_same_length(self):
        """HEX and BGR colour lists should have the same length."""
        from athena.visualization import SKELETON_COLOURS_HEX, SKELETON_COLOURS_BGR
        assert len(SKELETON_COLOURS_HEX) == len(SKELETON_COLOURS_BGR)

    def test_skeleton_links_are_pairs(self):
        """Every entry in SKELETON_LINKS should be a length-2 list of ints."""
        from athena.visualization import SKELETON_LINKS
        for link in SKELETON_LINKS:
            assert len(link) == 2
            assert isinstance(link[0], int)
            assert isinstance(link[1], int)

    def test_skeleton_colours_bgr_are_tuples(self):
        """Every entry in SKELETON_COLOURS_BGR should be a 3-tuple of ints."""
        from athena.visualization import SKELETON_COLOURS_BGR
        for colour in SKELETON_COLOURS_BGR:
            assert len(colour) == 3
            for channel in colour:
                assert isinstance(channel, int)
                assert 0 <= channel <= 255

    def test_right_hand_links(self):
        """Links 16-35 should connect right-hand landmarks (root at 33)."""
        from athena.visualization import SKELETON_LINKS
        right_hand_links = SKELETON_LINKS[16:36]
        assert len(right_hand_links) == 20
        # First link of the right hand should start at 33 (right hand root)
        assert right_hand_links[0][0] == 33

    def test_left_hand_links(self):
        """Links 36-55 should connect left-hand landmarks (root at 54)."""
        from athena.visualization import SKELETON_LINKS
        left_hand_links = SKELETON_LINKS[36:56]
        assert len(left_hand_links) == 20
        # First link of the left hand should start at 54 (left hand root)
        assert left_hand_links[0][0] == 54


class TestDrawLandmarksUnified:
    """Tests for the draw_landmarks_unified skeleton drawing function."""

    @staticmethod
    def _make_dummy_keypoints(n, x_base=100.0, y_base=100.0):
        """Create n dummy keypoints at (x_base + i, y_base + i)."""
        return [[x_base + i, y_base + i] for i in range(n)]

    def test_no_crash_with_valid_data(self):
        """Calling draw_landmarks_unified with valid dummy data should not raise."""
        from athena.visualization import draw_landmarks_unified

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        body = self._make_dummy_keypoints(33, 200, 200)
        right = self._make_dummy_keypoints(21, 300, 300)
        left = self._make_dummy_keypoints(21, 100, 100)
        result = draw_landmarks_unified(img, body, right, left)
        assert result is not None

    def test_image_is_modified(self):
        """After drawing, the image should differ from the original blank image."""
        from athena.visualization import draw_landmarks_unified

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = img.sum()
        body = self._make_dummy_keypoints(33, 200, 200)
        right = self._make_dummy_keypoints(21, 300, 300)
        left = self._make_dummy_keypoints(21, 100, 100)
        draw_landmarks_unified(img, body, right, left)
        assert img.sum() > original_sum, "Image should have been modified in-place"

    def test_returns_same_image(self):
        """draw_landmarks_unified should return the same image array it was given."""
        from athena.visualization import draw_landmarks_unified

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        body = self._make_dummy_keypoints(33, 200, 200)
        right = self._make_dummy_keypoints(21, 300, 300)
        left = self._make_dummy_keypoints(21, 100, 100)
        result = draw_landmarks_unified(img, body, right, left)
        assert result is img

    def test_sentinel_keypoints_do_not_crash(self):
        """Keypoints at sentinel value (-1, -1) should be skipped gracefully."""
        from athena.visualization import draw_landmarks_unified

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        body = [[-1, -1]] * 33
        right = [[-1, -1]] * 21
        left = [[-1, -1]] * 21
        result = draw_landmarks_unified(img, body, right, left)
        # Image should remain blank since all keypoints are sentinel
        assert img.sum() == 0
        assert result is img

    def test_face_keypoints_are_drawn(self):
        """When face_kpts are provided, they should add pixels to the image."""
        from athena.visualization import draw_landmarks_unified

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        body = [[-1, -1]] * 33
        right = [[-1, -1]] * 21
        left = [[-1, -1]] * 21
        face = self._make_dummy_keypoints(478, 200, 200)
        draw_landmarks_unified(img, body, right, left, face_kpts=face)
        assert img.sum() > 0, "Face keypoints should have drawn on the image"


class TestRenderMeshOverlay:
    """Tests for the render_mesh_overlay mesh rendering function."""

    def test_no_crash_with_simple_triangle(self):
        """Rendering a single triangle should not raise any exceptions."""
        from athena.visualization import render_mesh_overlay

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        verts_2d = np.array([[50, 50], [150, 50], [100, 150]], dtype=np.float64)
        verts_3d = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        base_colour = (100, 100, 230)
        render_mesh_overlay(img, verts_2d, verts_3d, faces, base_colour)
        # Should not raise; image should be modified
        assert img.sum() > 0

    def test_mesh_overlay_modifies_image(self):
        """The rendered mesh should change pixel values in the covered region."""
        from athena.visualization import render_mesh_overlay

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        verts_2d = np.array([[10, 10], [190, 10], [100, 190]], dtype=np.float64)
        verts_3d = np.array([[0, 0, 5], [1, 0, 5], [0.5, 1, 5]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        before = img.copy()
        render_mesh_overlay(img, verts_2d, verts_3d, faces, (200, 150, 100))
        assert not np.array_equal(img, before)

    def test_alpha_zero_leaves_image_unchanged(self):
        """With alpha=0, the overlay should have no visible effect."""
        from athena.visualization import render_mesh_overlay

        img = np.zeros((200, 200, 3), dtype=np.uint8) + 128
        original = img.copy()
        verts_2d = np.array([[10, 10], [190, 10], [100, 190]], dtype=np.float64)
        verts_3d = np.array([[0, 0, 5], [1, 0, 5], [0.5, 1, 5]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        render_mesh_overlay(img, verts_2d, verts_3d, faces, (200, 150, 100),
                            alpha=0.0)
        assert_allclose(img, original, atol=1)

    def test_multiple_faces(self):
        """Rendering a mesh with multiple faces should succeed."""
        from athena.visualization import render_mesh_overlay

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        verts_2d = np.array([
            [50, 50], [250, 50], [250, 250], [50, 250]
        ], dtype=np.float64)
        verts_3d = np.array([
            [0, 0, 2], [1, 0, 2], [1, 1, 3], [0, 1, 3]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        render_mesh_overlay(img, verts_2d, verts_3d, faces, (100, 200, 100))
        assert img.sum() > 0


# ============================================================================
# athena.labels2d
# ============================================================================


@skip_labels2d
class TestTransformationMatrix:
    """Tests for labels2d.transformation_matrix."""

    def test_identity_rotation_zero_translation(self):
        """Identity R and zero t should give a 4x4 identity matrix."""
        from athena.labels2d import transformation_matrix

        R = np.eye(3)
        t = np.zeros(3)
        T = transformation_matrix(R, t)
        assert T.shape == (4, 4)
        assert_allclose(T, np.eye(4))

    def test_shape(self):
        """Output must always be (4, 4)."""
        from athena.labels2d import transformation_matrix

        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = transformation_matrix(R, t)
        assert T.shape == (4, 4)

    def test_translation_in_last_column(self):
        """The translation vector should appear in the first three rows of the last column."""
        from athena.labels2d import transformation_matrix

        R = np.eye(3)
        t = np.array([10.0, 20.0, 30.0])
        T = transformation_matrix(R, t)
        assert_allclose(T[:3, 3], t)

    def test_bottom_row(self):
        """The bottom row of the 4x4 matrix should be [0, 0, 0, 1]."""
        from athena.labels2d import transformation_matrix

        R = np.random.randn(3, 3)
        t = np.random.randn(3)
        T = transformation_matrix(R, t)
        assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_rotation_block(self):
        """The upper-left 3x3 block should equal the input rotation matrix."""
        from athena.labels2d import transformation_matrix

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t = np.array([5, 6, 7], dtype=float)
        T = transformation_matrix(R, t)
        assert_allclose(T[:3, :3], R)


@skip_labels2d
class TestRotationMatrix:
    """Tests for labels2d.rotation_matrix (Rodrigues formula)."""

    def test_zero_rotation(self):
        """A zero rotation vector should produce the 3x3 identity matrix."""
        from athena.labels2d import rotation_matrix

        R = rotation_matrix(np.zeros(3))
        assert R.shape == (3, 3)
        assert_allclose(R, np.eye(3))

    def test_rotation_about_z_90_degrees(self):
        """A 90-degree rotation about the z-axis should map x -> y, y -> -x."""
        from athena.labels2d import rotation_matrix

        angle = np.pi / 2
        r = np.array([0, 0, angle])
        R = rotation_matrix(r)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        assert_allclose(R, expected, atol=1e-12)

    def test_rotation_about_x_180_degrees(self):
        """A 180-degree rotation about the x-axis: y -> -y, z -> -z."""
        from athena.labels2d import rotation_matrix

        r = np.array([np.pi, 0, 0])
        R = rotation_matrix(r)
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        assert_allclose(R, expected, atol=1e-12)

    def test_rotation_is_orthogonal(self):
        """The rotation matrix should be orthogonal: R^T R = I."""
        from athena.labels2d import rotation_matrix

        r = np.array([0.3, 0.5, -0.7])
        R = rotation_matrix(r)
        assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_rotation_determinant_is_one(self):
        """A proper rotation matrix has determinant +1."""
        from athena.labels2d import rotation_matrix

        r = np.array([1.2, -0.4, 0.8])
        R = rotation_matrix(r)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    def test_shape(self):
        """Output must be (3, 3)."""
        from athena.labels2d import rotation_matrix

        R = rotation_matrix(np.array([0.1, 0.2, 0.3]))
        assert R.shape == (3, 3)


@skip_labels2d
class TestLabels2dImports:
    """Verify that key labels2d functions are importable."""

    def test_read_calibration_importable(self):
        """read_calibration should be importable from athena.labels2d."""
        from athena.labels2d import read_calibration
        assert callable(read_calibration)

    def test_create_video_importable(self):
        """create_video should be importable from athena.labels2d."""
        from athena.labels2d import create_video
        assert callable(create_video)


# ============================================================================
# athena.triangulaterefine
# ============================================================================


def _make_synthetic_cameras(n_cameras=4, focal_length=500.0, image_size=640):
    """Create synthetic cameras arranged in a ring looking at the origin.

    Returns
    -------
    cam_mats_intrinsic : list of np.ndarray
        (3, 3) intrinsic matrices.
    cam_mats_extrinsic : np.ndarray
        (n_cameras, 4, 4) extrinsic matrices (world-to-camera).
    """
    cx = cy = image_size / 2.0
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    intrinsics = [K.copy() for _ in range(n_cameras)]

    extrinsics = np.zeros((n_cameras, 4, 4), dtype=np.float64)
    radius = 5.0
    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        cam_pos = np.array([radius * np.cos(angle),
                            radius * np.sin(angle),
                            0.0])
        # Camera looks towards origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.stack([right, up, -forward], axis=0)  # (3, 3) camera rotation
        t = -R @ cam_pos  # translation

        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = t
        extrinsics[i, 3, 3] = 1.0

    return intrinsics, extrinsics


def _project_point(point_3d, K, E):
    """Project a single 3D point to 2D using intrinsic K and extrinsic E.

    Returns pixel coordinates (u, v).
    """
    pt_h = np.append(point_3d, 1.0)
    cam_pt = E[:3] @ pt_h
    px_h = K @ cam_pt
    return px_h[:2] / px_h[2]


@skip_triangulaterefine
class TestUndistortPoints:
    """Tests for triangulaterefine._undistort_points."""

    def test_identity_intrinsic_zero_distortion(self):
        """With identity K and zero distortion, normalised coords should equal
        the output of cv.undistortPoints (which strips K and applies no distortion).
        """
        from athena.triangulaterefine import _undistort_points

        K = np.eye(3, dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        pts = np.array([[0.5, 0.3], [-0.2, 0.8]], dtype=np.float64)
        result = _undistort_points(pts, K, dist)
        # With identity K and zero dist, cv.undistortPoints returns normalised coords
        # which for identity K are just the input points themselves.
        assert_allclose(result.reshape(-1, 2), pts, atol=1e-10)

    def test_output_shape(self):
        """Output shape should be (n, 1, 2)."""
        from athena.triangulaterefine import _undistort_points

        K = np.eye(3, dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        pts = np.array([[100, 200], [300, 400]], dtype=np.float64)
        result = _undistort_points(pts, K, dist)
        assert result.shape == (2, 1, 2)

    def test_with_focal_length(self):
        """With a known focal length and zero distortion, undistorted points
        should equal (px - cx) / f, (py - cy) / f."""
        from athena.triangulaterefine import _undistort_points

        f = 500.0
        cx, cy = 320.0, 240.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        pts = np.array([[cx, cy]], dtype=np.float64)  # principal point
        result = _undistort_points(pts, K, dist)
        # Principal point should map to (0, 0) in normalised coordinates
        assert_allclose(result.reshape(-1, 2), [[0, 0]], atol=1e-10)


@skip_triangulaterefine
class TestTriangulateBatch:
    """Tests for triangulaterefine._triangulate_batch."""

    def test_known_3d_point(self):
        """Triangulate a known 3D point from synthetic cameras; verify recovery."""
        from athena.triangulaterefine import _triangulate_batch

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=4)
        true_3d = np.array([0.5, -0.3, 0.2])

        ncams = len(intrinsics)
        # Project to normalised camera coordinates (what _undistort_points returns)
        pts_2d = np.zeros((ncams, 1, 2), dtype=np.float64)
        for c in range(ncams):
            pt_h = np.append(true_3d, 1.0)
            cam_pt = extrinsics[c, :3] @ pt_h  # (3,)
            # Normalised: x/z, y/z
            pts_2d[c, 0, 0] = cam_pt[0] / cam_pt[2]
            pts_2d[c, 0, 1] = cam_pt[1] / cam_pt[2]

        result = _triangulate_batch(pts_2d, extrinsics)
        assert result.shape == (1, 3)
        assert_allclose(result[0], true_3d, atol=1e-6)

    def test_fewer_than_two_cameras_gives_nan(self):
        """A point visible in only 1 camera should produce NaN."""
        from athena.triangulaterefine import _triangulate_batch

        _, extrinsics = _make_synthetic_cameras(n_cameras=3)
        pts_2d = np.full((3, 1, 2), np.nan)
        pts_2d[0, 0] = [0.1, 0.2]  # only one camera sees it
        result = _triangulate_batch(pts_2d, extrinsics)
        assert np.all(np.isnan(result[0]))

    def test_multiple_points(self):
        """Triangulate several 3D points simultaneously."""
        from athena.triangulaterefine import _triangulate_batch

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=4)
        true_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        npts = true_points.shape[0]
        ncams = len(intrinsics)

        pts_2d = np.zeros((ncams, npts, 2), dtype=np.float64)
        for c in range(ncams):
            for p in range(npts):
                pt_h = np.append(true_points[p], 1.0)
                cam_pt = extrinsics[c, :3] @ pt_h
                pts_2d[c, p, 0] = cam_pt[0] / cam_pt[2]
                pts_2d[c, p, 1] = cam_pt[1] / cam_pt[2]

        result = _triangulate_batch(pts_2d, extrinsics)
        assert result.shape == (npts, 3)
        assert_allclose(result, true_points, atol=1e-6)


@skip_triangulaterefine
class TestBatchReproject:
    """Tests for triangulaterefine._batch_reproject."""

    def test_reproject_matches_input(self):
        """Reprojecting a triangulated 3D point should recover the original 2D pixels."""
        from athena.triangulaterefine import _triangulate_batch, _batch_reproject

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=4)
        true_3d = np.array([0.3, -0.5, 0.1])
        ncams = len(intrinsics)

        # Generate pixel-space observations and normalised coordinates
        pts_2d_px = np.zeros((ncams, 1, 2), dtype=np.float64)
        pts_2d_norm = np.zeros((ncams, 1, 2), dtype=np.float64)
        for c in range(ncams):
            pt_h = np.append(true_3d, 1.0)
            cam_pt = extrinsics[c, :3] @ pt_h
            pts_2d_norm[c, 0] = [cam_pt[0] / cam_pt[2], cam_pt[1] / cam_pt[2]]
            px = intrinsics[c] @ cam_pt
            pts_2d_px[c, 0] = [px[0] / px[2], px[1] / px[2]]

        # Triangulate then reproject
        p3d = _triangulate_batch(pts_2d_norm, extrinsics)
        reproj = _batch_reproject(p3d, intrinsics, extrinsics)

        assert reproj.shape == (ncams, 1, 2)
        for c in range(ncams):
            assert_allclose(reproj[c, 0], pts_2d_px[c, 0], atol=1e-4)

    def test_nan_3d_gives_nan_2d(self):
        """NaN 3D input should produce NaN 2D output for all cameras."""
        from athena.triangulaterefine import _batch_reproject

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=3)
        p3d = np.array([[np.nan, np.nan, np.nan]])
        reproj = _batch_reproject(p3d, intrinsics, extrinsics)
        assert np.all(np.isnan(reproj))

    def test_output_shape(self):
        """Output should be (ncams, npoints, 2)."""
        from athena.triangulaterefine import _batch_reproject

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=3)
        p3d = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        reproj = _batch_reproject(p3d, intrinsics, extrinsics)
        assert reproj.shape == (3, 2, 2)


@skip_triangulaterefine
class TestTriangulateWithFiltering:
    """Tests for triangulaterefine._triangulate_with_filtering."""

    def test_outlier_camera_is_filtered(self):
        """An outlier camera with corrupted 2D observations should be filtered out,
        and the resulting triangulation should still be close to the true point."""
        from athena.triangulaterefine import _triangulate_with_filtering

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=5)
        true_3d = np.array([0.4, -0.2, 0.3])
        ncams = 5

        pts_2d_norm = np.zeros((ncams, 1, 2), dtype=np.float64)
        pts_2d_px = np.zeros((ncams, 1, 2), dtype=np.float64)
        for c in range(ncams):
            pt_h = np.append(true_3d, 1.0)
            cam_pt = extrinsics[c, :3] @ pt_h
            pts_2d_norm[c, 0] = [cam_pt[0] / cam_pt[2], cam_pt[1] / cam_pt[2]]
            px = intrinsics[c] @ cam_pt
            pts_2d_px[c, 0] = [px[0] / px[2], px[1] / px[2]]

        # Corrupt camera 0 by adding a huge offset to its pixel observations
        pts_2d_px_corrupted = pts_2d_px.copy()
        pts_2d_px_corrupted[0, 0] += 200.0  # way off

        # Also corrupt the normalised coords for camera 0 so DLT starts badly
        pts_2d_norm_corrupted = pts_2d_norm.copy()
        pts_2d_norm_corrupted[0, 0] += 0.5

        result = _triangulate_with_filtering(
            pts_2d_norm_corrupted, pts_2d_px_corrupted,
            extrinsics, intrinsics,
            reproj_threshold=15.0, min_cams=2, max_iterations=5
        )
        assert result.shape == (1, 3)
        # Even with one corrupted camera, the filtering should yield a
        # reasonable result from the remaining 4 cameras.
        assert_allclose(result[0], true_3d, atol=0.05)

    def test_clean_data_passes_through(self):
        """With perfectly clean data, filtering should not degrade the result."""
        from athena.triangulaterefine import _triangulate_with_filtering

        intrinsics, extrinsics = _make_synthetic_cameras(n_cameras=4)
        true_3d = np.array([0.0, 0.0, 0.0])
        ncams = 4

        pts_2d_norm = np.zeros((ncams, 1, 2), dtype=np.float64)
        pts_2d_px = np.zeros((ncams, 1, 2), dtype=np.float64)
        for c in range(ncams):
            pt_h = np.append(true_3d, 1.0)
            cam_pt = extrinsics[c, :3] @ pt_h
            pts_2d_norm[c, 0] = [cam_pt[0] / cam_pt[2], cam_pt[1] / cam_pt[2]]
            px = intrinsics[c] @ cam_pt
            pts_2d_px[c, 0] = [px[0] / px[2], px[1] / px[2]]

        result = _triangulate_with_filtering(
            pts_2d_norm, pts_2d_px,
            extrinsics, intrinsics,
            reproj_threshold=15.0
        )
        assert_allclose(result[0], true_3d, atol=1e-6)


@skip_triangulaterefine
class TestSmooth3d:
    """Tests for triangulaterefine._smooth3d."""

    def test_smoothing_reduces_noise(self):
        """Smoothing a noisy 3D trajectory should reduce the overall noise level."""
        from athena.triangulaterefine import _smooth3d

        rng = np.random.RandomState(42)
        n_frames = 200
        n_landmarks = 5
        t = np.linspace(0, 2 * np.pi, n_frames)

        # Clean sinusoidal signal + noise
        clean = np.zeros((n_frames, n_landmarks, 3))
        for lm in range(n_landmarks):
            for coord in range(3):
                clean[:, lm, coord] = np.sin(t + lm * 0.5 + coord * 0.3)

        noisy = clean + rng.randn(n_frames, n_landmarks, 3) * 0.3
        smoothed = _smooth3d(noisy.copy(), fps=100, frequency_cutoff=20)

        noise_before = np.nanmean((noisy - clean) ** 2)
        noise_after = np.nanmean((smoothed - clean) ** 2)
        assert noise_after < noise_before, (
            f"Smoothing should reduce noise: MSE before={noise_before:.4f}, "
            f"after={noise_after:.4f}"
        )

    def test_output_shape_preserved(self):
        """_smooth3d should return data of the same shape as the input."""
        from athena.triangulaterefine import _smooth3d

        data = np.random.randn(100, 10, 3)
        result = _smooth3d(data.copy(), fps=60, frequency_cutoff=10)
        assert result.shape == data.shape

    def test_constant_signal_unchanged(self):
        """A perfectly constant signal should be (nearly) unchanged by smoothing."""
        from athena.triangulaterefine import _smooth3d

        data = np.ones((50, 3, 3)) * 5.0
        result = _smooth3d(data.copy(), fps=60, frequency_cutoff=10)
        assert_allclose(result, data, atol=1e-10)

    def test_handles_nan_values(self):
        """_smooth3d should not crash when the input contains NaN values."""
        from athena.triangulaterefine import _smooth3d

        data = np.random.randn(100, 3, 3)
        data[10:15, 0, :] = np.nan  # short NaN run
        result = _smooth3d(data.copy(), fps=60, frequency_cutoff=10)
        assert result.shape == data.shape
        # The short NaN run should be interpolated over (not remain NaN)
        assert not np.any(np.isnan(result[:, 0, :]))


@skip_triangulaterefine
class TestRestoreLongNanRuns:
    """Tests for triangulaterefine._restore_long_nan_runs."""

    def test_long_nan_run_restored(self):
        """NaN runs longer than min_length should be restored as NaN."""
        from athena.triangulaterefine import _restore_long_nan_runs

        original = np.ones(30)
        original[10:20] = np.nan  # 10-frame NaN run (> default min_length=5)
        filtered = np.ones(30) * 2.0  # "smoothed" version with no NaN

        result = _restore_long_nan_runs(original, filtered.copy(), min_length=5)
        # The long NaN run should be restored
        assert np.all(np.isnan(result[10:20]))
        # Other values should remain as the filtered values
        assert_allclose(result[:10], 2.0)
        assert_allclose(result[20:], 2.0)

    def test_short_nan_run_not_restored(self):
        """NaN runs shorter than or equal to min_length should remain interpolated."""
        from athena.triangulaterefine import _restore_long_nan_runs

        original = np.ones(30)
        original[10:13] = np.nan  # 3-frame NaN run (< min_length=5)
        filtered = np.ones(30) * 2.0

        result = _restore_long_nan_runs(original, filtered.copy(), min_length=5)
        # Short NaN run should NOT be restored
        assert not np.any(np.isnan(result))

    def test_exact_min_length_not_restored(self):
        """A NaN run of exactly min_length should NOT be restored
        (the check is strictly greater than)."""
        from athena.triangulaterefine import _restore_long_nan_runs

        original = np.ones(30)
        original[10:15] = np.nan  # exactly 5 frames
        filtered = np.ones(30) * 2.0

        result = _restore_long_nan_runs(original, filtered.copy(), min_length=5)
        # Run length 5 == min_length, so not restored (> not >=)
        assert not np.any(np.isnan(result))

    def test_no_nan_in_original(self):
        """When the original has no NaN, the filtered data should pass through unchanged."""
        from athena.triangulaterefine import _restore_long_nan_runs

        original = np.ones(20)
        filtered = np.ones(20) * 3.0
        result = _restore_long_nan_runs(original, filtered.copy(), min_length=5)
        assert_allclose(result, 3.0)

    def test_multiple_nan_runs(self):
        """Multiple NaN runs of different lengths should be handled independently."""
        from athena.triangulaterefine import _restore_long_nan_runs

        original = np.ones(50)
        original[5:8] = np.nan     # 3 frames (short, should stay)
        original[20:30] = np.nan   # 10 frames (long, should restore)
        original[40:45] = np.nan   # 5 frames (exactly min, should stay)
        filtered = np.ones(50) * 2.0

        result = _restore_long_nan_runs(original, filtered.copy(), min_length=5)
        # Short run: no NaN restored
        assert not np.any(np.isnan(result[5:8]))
        # Long run: NaN restored
        assert np.all(np.isnan(result[20:30]))
        # Exact min_length run: no NaN restored
        assert not np.any(np.isnan(result[40:45]))
