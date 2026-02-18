"""Shared skeleton topology, colours, and mesh rendering used by both phases.

The detection phase (``labels2d``) and triangulation phase
(``triangulaterefine``) must draw identical skeletons so that output
videos look consistent.  All shared constants and helper functions live
here to avoid duplication.
"""

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# Skeleton link & colour definitions
# ---------------------------------------------------------------------------
# Full skeleton: body (16 links) + right hand (20 links) + left hand (20 links)

SKELETON_LINKS = [
    # Body (indices 0-15)
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 54],   # left shoulder -> left hand root
    [12, 14], [14, 33],   # right shoulder -> right hand root
    # Right hand (indices 16-35)
    [33, 34], [34, 35], [35, 36], [36, 37],
    [33, 38], [38, 39], [39, 40], [40, 41],
    [33, 42], [42, 43], [43, 44], [44, 45],
    [33, 46], [46, 47], [47, 48], [48, 49],
    [33, 50], [50, 51], [51, 52], [52, 53],
    # Left hand (indices 36-55)
    [54, 55], [55, 56], [56, 57], [57, 58],
    [54, 59], [59, 60], [60, 61], [61, 62],
    [54, 63], [63, 64], [64, 65], [65, 66],
    [54, 67], [67, 68], [68, 69], [69, 70],
    [54, 71], [71, 72], [72, 73], [73, 74],
]

BODY_LINKS = SKELETON_LINKS[:16]

SKELETON_COLOURS_HEX = [
    # Body
    '#EEDE33', '#EEDE33', '#EEDE33', '#EEDE33',
    '#EEDE33', '#EEDE33', '#EEDE33', '#EEDE33',
    '#DDDDDD', '#DDDDDD', '#DDDDDD', '#DDDDDD',
    '#009988', '#009988',
    '#EE7733', '#EE7733',
    # Right hand -- pink/red gradient
    '#FDE7EF', '#FDE7EF', '#FDE7EF', '#FDE7EF',
    '#F589B1', '#F589B1', '#F589B1', '#F589B1',
    '#ED2B72', '#ED2B72', '#ED2B72', '#ED2B72',
    '#A50E45', '#A50E45', '#A50E45', '#A50E45',
    '#47061D', '#47061D', '#47061D', '#47061D',
    # Left hand -- blue gradient
    '#E5F6FF', '#E5F6FF', '#E5F6FF', '#E5F6FF',
    '#80D1FF', '#80D1FF', '#80D1FF', '#80D1FF',
    '#1AACFF', '#1AACFF', '#1AACFF', '#1AACFF',
    '#0072B3', '#0072B3', '#0072B3', '#0072B3',
    '#00314D', '#00314D', '#00314D', '#00314D',
]

BODY_COLOURS_HEX = SKELETON_COLOURS_HEX[:16]

# Mesh base colours (BGR)
MESH_RIGHT_COLOUR_BGR = (100, 100, 230)   # warm red
MESH_LEFT_COLOUR_BGR = (230, 150, 80)     # cool blue

# Face mesh landmark colour (BGR)
FACE_COLOUR_BGR = (150, 200, 100)


def hex_to_bgr(hexcode):
    """Convert a hex colour string (e.g. ``'#EEDE33'``) to a BGR tuple."""
    h = hexcode.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return rgb[::-1]


SKELETON_COLOURS_BGR = [hex_to_bgr(c) for c in SKELETON_COLOURS_HEX]


def render_mesh_overlay(img, verts_2d, verts_3d, faces, base_colour_bgr,
                        alpha=0.7):
    """Render a filled, depth-sorted, Lambertian-shaded mesh overlay.

    Parameters
    ----------
    img : np.ndarray
        BGR image to draw on (modified in-place).
    verts_2d : np.ndarray
        ``(V, 2)`` projected 2D pixel coordinates.
    verts_3d : np.ndarray
        ``(V, 3)`` vertices in some 3D space (for depth sorting / shading).
    faces : np.ndarray
        ``(F, 3)`` triangle indices.
    base_colour_bgr : tuple
        Base BGR colour for the mesh.
    alpha : float
        Blending factor (0 = transparent, 1 = opaque).
    """
    tri_3d = verts_3d[faces]
    face_depths = tri_3d[:, :, 2].mean(axis=1)

    # Lambertian shading
    v0, v1, v2 = tri_3d[:, 0], tri_3d[:, 1], tri_3d[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals = normals / norms

    light_dir = np.array([0.0, 0.0, -1.0])
    diffuse = np.abs(normals @ light_dir)
    shade = 0.35 + 0.65 * diffuse

    sort_order = np.argsort(-face_depths)

    base_b = float(base_colour_bgr[0])
    base_g = float(base_colour_bgr[1])
    base_r = float(base_colour_bgr[2])
    tri_2d = verts_2d[faces]

    overlay = img.copy()
    for idx in sort_order:
        s = shade[idx]
        colour = (int(base_b * s), int(base_g * s), int(base_r * s))
        pts = tri_2d[idx].astype(np.int32).reshape((-1, 1, 2))
        cv.fillPoly(overlay, [pts], colour)

    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_landmarks_unified(bgr_image, body_kpts, right_kpts, left_kpts,
                           face_kpts=None, mesh_right=None, mesh_left=None,
                           mesh_faces=None):
    """Draw skeleton on a BGR image using the canonical ATHENA layout.

    When mesh data is provided (HaMeR mode), renders shaded hand meshes
    instead of skeleton lines for the hands.

    Parameters
    ----------
    bgr_image : np.ndarray
        BGR image to draw on (modified in-place).
    body_kpts : list
        Body keypoints, each ``[x, y, ...]``.  Length 33.
    right_kpts : list
        Right hand keypoints, each ``[x, y, ...]``.  Length 21.
    left_kpts : list
        Left hand keypoints, each ``[x, y, ...]``.  Length 21.
    face_kpts : list, optional
        Face keypoints, each ``[x, y, ...]``.  Length 478.
    mesh_right : dict, optional
        ``{'verts_2d': (778,2), 'verts_3d': (778,3)}`` for the right hand.
    mesh_left : dict, optional
        ``{'verts_2d': (778,2), 'verts_3d': (778,3)}`` for the left hand.
    mesh_faces : np.ndarray, optional
        ``(F, 3)`` triangle indices (MANO topology).

    Returns
    -------
    np.ndarray
        The same *bgr_image* (drawn in-place).
    """
    has_mesh = mesh_faces is not None

    # Build a combined (75, 2) array:  0-32 body, 33-53 right, 54-74 left
    combined = np.full((75, 2), np.nan, dtype=np.float64)

    for i, kpt in enumerate(body_kpts):
        if kpt[0] != -1 or kpt[1] != -1:
            combined[i] = [kpt[0], kpt[1]]

    for i, kpt in enumerate(right_kpts):
        if kpt[0] != -1 or kpt[1] != -1:
            combined[33 + i] = [kpt[0], kpt[1]]

    for i, kpt in enumerate(left_kpts):
        if kpt[0] != -1 or kpt[1] != -1:
            combined[54 + i] = [kpt[0], kpt[1]]

    # Mesh overlays first (behind skeleton)
    if has_mesh:
        faces_left = mesh_faces[:, [0, 2, 1]]   # reversed winding
        if mesh_right is not None and mesh_right['verts_2d'][0, 0] != -1:
            render_mesh_overlay(bgr_image, mesh_right['verts_2d'],
                                mesh_right['verts_3d'], mesh_faces,
                                MESH_RIGHT_COLOUR_BGR)
        if mesh_left is not None and mesh_left['verts_2d'][0, 0] != -1:
            render_mesh_overlay(bgr_image, mesh_left['verts_2d'],
                                mesh_left['verts_3d'], faces_left,
                                MESH_LEFT_COLOUR_BGR)

    # Skeleton links -- skip hand links (idx >= 16) when mesh is present
    for idx, (start, end) in enumerate(SKELETON_LINKS):
        if has_mesh and idx >= 16:
            continue
        if not np.isnan(combined[start, 0]) and not np.isnan(combined[end, 0]):
            pt1 = tuple(combined[start].astype(int))
            pt2 = tuple(combined[end].astype(int))
            cv.line(bgr_image, pt1, pt2, SKELETON_COLOURS_BGR[idx], 2)

    # Face landmarks
    if face_kpts is not None:
        for kpt in face_kpts:
            if kpt[0] != -1 or kpt[1] != -1:
                pt = (int(kpt[0]), int(kpt[1]))
                cv.circle(bgr_image, pt, 1, FACE_COLOUR_BGR, -1, cv.LINE_AA)

    return bgr_image
