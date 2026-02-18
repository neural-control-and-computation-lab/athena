"""
HaMeR hand detection backend for ATHENA.

Provides 2D hand keypoint and mesh recovery as an alternative to MediaPipe
HandLandmarker.  MediaPipe HandLandmarker supplies adaptive bounding boxes
which are then fed to HaMeR for high-quality mesh recovery.

All heavy dependencies (torch, hamer) are lazily imported so that this module
can be safely imported even when those packages are not installed.

Supports CUDA, MPS (Apple Silicon), and CPU backends.

Installation:
    pip install athena-tracking[hamer]
    pip install --no-deps git+https://github.com/geopavlakos/hamer.git
"""

import sys
import types
import numpy as np


class _MockModule(types.ModuleType):
    """
    A mock module that returns a dummy object for any *public* attribute access.

    Used to satisfy HaMeR's module-level references to rendering classes
    (e.g. ``pyrender.Node`` in type annotations) without actually
    installing the rendering stack.  Dunder attributes are not mocked so
    that Python's import machinery and ``inspect`` module continue to
    work normally.
    """

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        # Return a dummy class so type annotations like List[pyrender.Node] work
        return type(name, (), {})


def _patch_compat():
    """
    Apply compatibility patches for dependencies that don't support Python 3.12+
    and/or NumPy 2.0+.

    * ``chumpy`` uses the removed ``inspect.getargspec`` (gone in Python 3.11+).
      We alias it to ``inspect.getfullargspec``.
    * ``chumpy`` imports ``np.bool``, ``np.int``, ``np.float``, ``np.complex``,
      ``np.object``, ``np.unicode``, ``np.str`` which were removed in NumPy 2.0.
      We restore them as aliases to the builtins / numpy scalar types.
    """
    import inspect
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    # Restore removed numpy aliases that chumpy expects
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = np.int_
    if not hasattr(np, 'float'):
        np.float = np.float64
    if not hasattr(np, 'complex'):
        np.complex = np.complex128
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'str'):
        np.str = np.str_
    if not hasattr(np, 'unicode'):
        np.unicode = np.str_


def _mock_rendering_deps():
    """
    Insert lightweight mock modules for rendering-only dependencies
    (pyrender, trimesh, pyopengl) into ``sys.modules`` so that HaMeR's
    import chain completes without actually installing them.

    HaMeR's ``hamer.utils.__init__`` unconditionally imports
    ``Renderer``, ``MeshRenderer``, and ``SkeletonRenderer``, which in
    turn do ``import pyrender`` at module level.  We never call any
    rendering code — only the forward pass for keypoint prediction — so
    mocking these modules is safe.

    Also applies Python 3.12+ compatibility patches (see ``_patch_compat``).

    This function is idempotent: calling it multiple times is harmless.
    """
    _patch_compat()
    for mod_name in ('pyrender', 'pyrender.constants',
                     'trimesh', 'trimesh.primitives', 'trimesh.transformations',
                     'OpenGL', 'OpenGL.GL',
                     'OpenGL.platform', 'OpenGL.platform.egl'):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _MockModule(mod_name)


# ---------------------------------------------------------------------------
# Module-level model cache (populated on first call to load_models*)
# ---------------------------------------------------------------------------
_hamer_model = None
_hamer_cfg = None

# ---------------------------------------------------------------------------
# MANO → MediaPipe joint reordering
# ---------------------------------------------------------------------------
# MANO ordering: wrist, then 4 joints per finger in order
#   thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
# MediaPipe ordering: wrist(0), thumb(1-4), index(5-8), middle(9-12),
#   ring(13-16), pinky(17-20) but with different intra-finger order
# This array is indexed as: mediapipe_kpts = mano_kpts[MANO_TO_MEDIAPIPE]
MANO_TO_MEDIAPIPE = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]


def _get_device(use_gpu=True):
    """
    Select the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Parameters:
        use_gpu (bool): Whether GPU acceleration is requested.

    Returns:
        torch.device: The selected device.
    """
    import torch

    if not use_gpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def is_available():
    """
    Check whether HaMeR and its core dependencies are importable.

    Returns:
        bool: True if torch and hamer can be imported.
    """
    try:
        import torch  # noqa: F401
        _mock_rendering_deps()
        import hamer   # noqa: F401
        return True
    except ImportError:
        return False



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _setup_cache_dir():
    """
    Ensure HaMeR's ``CACHE_DIR_HAMER`` is an absolute path pointing to a
    stable location (inside the athena package directory) rather than the
    fragile relative ``"./_DATA"``.  Also patches ``DEFAULT_CHECKPOINT``
    so that ``load_hamer()`` finds the checkpoint regardless of cwd.

    Returns:
        str: The absolute path used as cache directory.
    """
    import os
    import hamer.configs as _hcfg
    import hamer.models as _hmod

    # Use _DATA next to the athena package itself so the path is cwd-independent
    athena_pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(athena_pkg_dir)
    cache_dir = os.path.join(repo_dir, '_DATA')
    _hcfg.CACHE_DIR_HAMER = cache_dir
    _hmod.DEFAULT_CHECKPOINT = os.path.join(cache_dir, 'hamer_ckpts', 'checkpoints', 'hamer.ckpt')
    return cache_dir


def load_models(device=None, use_gpu=True):
    """
    Initialize the HaMeR hand model.  Results are cached in module-level
    globals so they are loaded only once per process.

    Parameters:
        device (torch.device, optional): Explicit device. If None, auto-detected.
        use_gpu (bool): Whether GPU acceleration is requested (used if device is None).

    Returns:
        tuple: (hamer_model, hamer_cfg, device)
    """
    global _hamer_model, _hamer_cfg

    import os
    import torch
    _mock_rendering_deps()
    from hamer.models import download_models

    cache_dir = _setup_cache_dir()
    from hamer.models import DEFAULT_CHECKPOINT  # re-import after patching

    if device is None:
        device = _get_device(use_gpu)

    if _hamer_model is not None:
        return _hamer_model, _hamer_cfg, device

    print(f"[HaMeR] Loading hand model on {device}...")

    download_models(cache_dir)

    # Check for MANO model files (requires manual download from mano.is.tue.mpg.de)
    mano_dir = os.path.join(cache_dir, 'data', 'mano')
    mano_file = os.path.join(mano_dir, 'MANO_RIGHT.pkl')
    if not os.path.exists(mano_file):
        raise FileNotFoundError(
            f"MANO model file not found at:\n  {mano_file}\n\n"
            "The MANO hand model is required by HaMeR but cannot be downloaded\n"
            "automatically due to its license terms. To obtain it:\n"
            "  1. Register at https://mano.is.tue.mpg.de\n"
            "  2. Download the MANO model (right hand)\n"
            "  3. Place MANO_RIGHT.pkl in:\n"
            f"     {mano_dir}/"
        )

    _hamer_model, _hamer_cfg = _load_hamer_no_renderer(DEFAULT_CHECKPOINT)
    _hamer_model = _hamer_model.to(device)
    _hamer_model.eval()

    print(f"[HaMeR] Hand model loaded on {device}.")
    return _hamer_model, _hamer_cfg, device


def _load_hamer_no_renderer(checkpoint_path):
    """
    Load the HaMeR model with ``init_renderer=False`` so that the mocked
    pyrender/trimesh modules are never actually called.

    This mirrors ``hamer.models.load_hamer()`` but passes the extra kwarg.
    """
    from pathlib import Path
    from hamer.configs import get_config
    from hamer.models import HAMER

    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override config for ViT backbone compatibility
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(
        checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False,
    )
    return model, model_cfg



def get_mano_faces():
    """
    Return the MANO triangle-face array (1538, 3) from the loaded model.

    The array is constant across all frames and hands — it defines the
    mesh topology (which vertices form each triangle).

    Requires that ``load_models()`` has been called first so that
    ``_hamer_model`` is populated.

    Returns:
        np.ndarray: Integer array of shape (1538, 3).

    Raises:
        RuntimeError: If no model has been loaded yet.
    """
    if _hamer_model is None:
        raise RuntimeError(
            "get_mano_faces() called before loading the HaMeR model. "
            "Call load_models() first."
        )
    return np.array(_hamer_model.mano.faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def _crop_coords_to_pixel(pred_kpts_2d, box_center, box_size, img_size,
                          is_right=True):
    """
    Transform HaMeR's normalised crop-space 2D keypoints to full-image pixel coordinates.

    HaMeR's ``pred_keypoints_2d`` values are in approximately [-0.5, 0.5], where
    (0, 0) is the centre of the bounding-box crop and the crop spans
    [-0.5, 0.5] in both axes. The mapping back to pixels is::

        pixel = pred_kpt * bbox_size + box_center

    where ``bbox_size`` is the rescaled square bounding-box side length and
    ``box_center`` is the bbox centre in pixel coordinates.

    For left hands (``is_right=False``), ``ViTDetDataset`` horizontally flips
    the input image before feeding it to HaMeR.  The model therefore predicts
    in the *flipped* image's coordinate frame, so the crop-space X offset
    points in the wrong direction relative to the original image.  Because
    ``box_center`` is returned as the **original** (unflipped) centre, we must
    negate ``pred_x`` before applying the pixel transform so that the offset
    is measured in the original image's coordinate direction::

        pred_x_corrected = -pred_x   (left hands only)

    Parameters:
        pred_kpts_2d (np.ndarray): (N, 2) keypoints in crop-normalised coordinates.
        box_center (np.ndarray): (2,) centre of the bounding box in pixel coordinates.
        box_size (float): Side length of the square bounding box in pixels.
        img_size (np.ndarray): (2,) image dimensions (width, height) for clamping.
        is_right (bool): Whether this is a right hand.  If False (left hand),
            the X component of the crop predictions is negated before the
            pixel transform.

    Returns:
        np.ndarray: (N, 2) keypoints in full-image pixel coordinates.
    """
    pred = pred_kpts_2d.copy()
    if not is_right:
        pred[:, 0] = -pred[:, 0]

    pixel_coords = pred * box_size + box_center

    # Clamp to image bounds
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, img_size[0] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, img_size[1] - 1)
    return pixel_coords


def _reorder_mano_to_mediapipe(keypoints):
    """
    Reorder keypoints from MANO joint ordering to MediaPipe joint ordering.

    Parameters:
        keypoints (np.ndarray): (21, N) array in MANO ordering.

    Returns:
        np.ndarray: (21, N) array in MediaPipe ordering.
    """
    return keypoints[MANO_TO_MEDIAPIPE]


# ---------------------------------------------------------------------------
# Vertex projection helper
# ---------------------------------------------------------------------------


def _project_vertices_to_2d(out, model_cfg):
    """
    Project 3D mesh vertices to 2D crop-normalised coordinates.

    Uses the same perspective projection that HaMeR applies to keypoints
    internally (``perspective_projection``), so the resulting 2D coordinates
    are in the same space as ``pred_keypoints_2d`` and can be fed to
    ``_crop_coords_to_pixel`` to obtain full-image pixel coordinates.

    Parameters:
        out (dict): HaMeR forward-pass output containing ``pred_vertices``,
            ``pred_cam_t``, and ``focal_length``.
        model_cfg: HaMeR model configuration (provides ``MODEL.IMAGE_SIZE``).

    Returns:
        np.ndarray: (B, 778, 2) vertex coordinates in crop-normalised space.
    """
    import torch
    from hamer.utils.geometry import perspective_projection

    pred_verts = out['pred_vertices']          # (B, 778, 3)
    pred_cam_t = out['pred_cam_t']             # (B, 3)
    focal_length = out['focal_length']         # (B, 2)

    pred_verts_2d = perspective_projection(
        pred_verts,
        translation=pred_cam_t.reshape(-1, 3),
        focal_length=focal_length.reshape(-1, 2) / model_cfg.MODEL.IMAGE_SIZE,
    )
    return pred_verts_2d.reshape(pred_verts.shape[0], -1, 2).cpu().numpy()


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def _batch_to_device(batch, device):
    """
    Move a batch dict to *device*, casting float64 tensors to float32 first.

    MPS (Apple Silicon) does not support float64.  Rather than relying on
    HaMeR's ``recursive_to`` — which blindly calls ``.to(device)`` — we
    downcast any double-precision tensors before the transfer.
    """
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:
                v = v.float()
            v = v.to(device)
        out[k] = v
    return out


def _empty_result():
    """Return an empty hand-detection result dict."""
    return {
        'left_keypoints_2d': None, 'right_keypoints_2d': None,
        'left_keypoints_3d': None, 'right_keypoints_3d': None,
        'left_vertices_3d': None, 'right_vertices_3d': None,
        'left_vertices_2d': None, 'right_vertices_2d': None,
    }


def _run_hamer_and_unpack(model, model_cfg, device, img_bgr, boxes, right_arr,
                           rescale_factor=2.0):
    """Run HaMeR on prepared bounding boxes and return the standard result dict.

    Shared post-processing for all three ``detect_hands_*`` functions.

    Parameters:
        model: HaMeR model (on device, eval mode).
        model_cfg: HaMeR model configuration.
        device (torch.device): Compute device.
        img_bgr (np.ndarray): BGR image, shape (H, W, 3).
        boxes (np.ndarray): Bounding boxes, shape (N, 4), format [x1, y1, x2, y2].
        right_arr (np.ndarray): Array of 0/1 indicating right (1) or left (0) hand.
        rescale_factor (float): Rescale factor for ViTDetDataset crop.

    Returns:
        dict: Standard result dict with keys ``{left,right}_{keypoints_2d,
              keypoints_3d, vertices_3d, vertices_2d}``.
    """
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    result = _empty_result()

    if len(boxes) == 0:
        return result

    dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right_arr,
                            rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(boxes), shuffle=False, num_workers=0)

    for batch in dataloader:
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_kpts_2d = out['pred_keypoints_2d'].cpu().numpy()
        pred_kpts_3d = out['pred_keypoints_3d'].cpu().numpy()
        pred_verts = out['pred_vertices'].cpu().numpy()
        pred_verts_2d = _project_vertices_to_2d(out, model_cfg)
        box_centers = batch['box_center'].float().cpu().numpy()
        box_sizes = batch['box_size'].float().cpu().numpy()
        img_sizes = batch['img_size'].float().cpu().numpy()
        rights = batch['right'].cpu().numpy()

        for i in range(len(pred_kpts_2d)):
            is_r = bool(rights[i])
            kpts_pixel = _crop_coords_to_pixel(
                pred_kpts_2d[i], box_centers[i], box_sizes[i], img_sizes[i],
                is_right=is_r)
            kpts_pixel = _reorder_mano_to_mediapipe(kpts_pixel)
            kpts_3d = _reorder_mano_to_mediapipe(pred_kpts_3d[i])
            verts_pixel = _crop_coords_to_pixel(
                pred_verts_2d[i], box_centers[i], box_sizes[i], img_sizes[i],
                is_right=is_r)

            prefix = 'right' if is_r else 'left'
            result[f'{prefix}_keypoints_2d'] = kpts_pixel
            result[f'{prefix}_keypoints_3d'] = kpts_3d
            result[f'{prefix}_vertices_3d'] = pred_verts[i]
            result[f'{prefix}_vertices_2d'] = verts_pixel

    return result


def detect_hands_mp_landmarks(img_bgr, left_hand_lm, right_hand_lm,
                               model, model_cfg, device,
                               padding_factor=1.5, rescale_factor=2.0):
    """
    HaMeR pipeline using MediaPipe HandLandmarker bounding boxes.

    Computes adaptive bounding boxes from MediaPipe 2D hand landmark arrays
    instead of using fixed-size boxes around wrist positions. This produces
    tighter, better-centred crops that improve HaMeR mesh quality.

    Parameters:
        img_bgr (np.ndarray): BGR image, shape (H, W, 3).
        left_hand_lm (np.ndarray or None): (21, 5) MediaPipe left hand landmarks
            in format [x_px, y_px, z, vis, pres]. Pass None or array with -1 values
            if hand not detected.
        right_hand_lm (np.ndarray or None): (21, 5) MediaPipe right hand landmarks.
        model: HaMeR model (on device, eval mode).
        model_cfg: HaMeR model configuration.
        device (torch.device): Compute device.
        padding_factor (float): How much to expand the bbox beyond the landmark extent.
        rescale_factor (float): Rescale factor for the ViTDetDataset crop.

    Returns:
        dict: Keys ``{left,right}_{keypoints_2d, keypoints_3d, vertices_3d, vertices_2d}``
            (np.ndarray or None).
    """
    H, W = img_bgr.shape[:2]

    bboxes = []
    is_right = []

    for lm, is_r in [(left_hand_lm, 0), (right_hand_lm, 1)]:
        if lm is None:
            continue
        # Check if valid (non-sentinel values)
        if lm[0, 0] == -1:
            continue

        valid = lm[:, 0] != -1
        if valid.sum() < 3:
            continue

        x_min = lm[valid, 0].min()
        x_max = lm[valid, 0].max()
        y_min = lm[valid, 1].min()
        y_max = lm[valid, 1].max()

        # Expand bbox with padding and make square
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * padding_factor
        h = (y_max - y_min) * padding_factor
        side = max(w, h)

        bbox = [
            max(0, cx - side / 2), max(0, cy - side / 2),
            min(W, cx + side / 2), min(H, cy + side / 2),
        ]
        bboxes.append(bbox)
        is_right.append(is_r)

    if not bboxes:
        return _empty_result()

    boxes = np.array(bboxes, dtype=np.float32)
    right_arr = np.array(is_right)

    return _run_hamer_and_unpack(model, model_cfg, device, img_bgr,
                                 boxes, right_arr, rescale_factor=rescale_factor)
