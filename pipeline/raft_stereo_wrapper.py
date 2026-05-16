"""
RAFT-Stereo Wrapper Module
===========================
Deep learning stereo matching using RAFT-Stereo (Lipson et al., 3DV 2021).

This module wraps the RAFT-Stereo model to provide the same interface as
the classical StereoSGBM disparity computation in disparity.py.

RAFT-Stereo works by:
1. Extracting multi-scale features from both images using CNNs
2. Building a correlation volume between left and right features
3. Iteratively refining a disparity field using GRU-based updates
4. Upsampling the refined disparity to full resolution via convex upsampling

Compared to StereoSGBM, RAFT-Stereo produces much denser and more accurate
disparity maps, especially on textureless surfaces and thin structures.

Requirements:
    - torch, torchvision
    - RAFT-Stereo source (cloned into raft_stereo/ directory)
    - Pretrained weights (models/raftstereo-middlebury.pth)
"""

import os
import sys
import numpy as np
import cv2

import config

# ── Check PyTorch availability ──────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Module-level model cache ────────────────────────────────────────────────
_model = None
_device = None


def _get_raft_stereo_path():
    """Return the path to the RAFT-Stereo root directory."""
    return os.path.join(config.PROJECT_ROOT, "raft_stereo")


def _build_args():
    """
    Build a namespace object mimicking the RAFT-Stereo argparse defaults.
    We use the Middlebury demo settings (--corr_implementation alt)
    which do NOT require the custom CUDA sampler.
    """
    from argparse import Namespace

    args = Namespace(
        # Model architecture
        hidden_dims=[128, 128, 128],
        corr_implementation="alt",          # Pure PyTorch, no CUDA kernel needed
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=False,              # Will be set based on device

        # Inference
        valid_iters=config.RAFT_STEREO_VALID_ITERS,
    )

    return args


def _load_model():
    """
    Load the RAFT-Stereo model (cached after first call).

    Returns:
        model: RAFTStereo model in eval mode
        device: torch.device (cuda or cpu)
    """
    global _model, _device

    if _model is not None:
        return _model, _device

    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed. Install with: pip install torch torchvision"
        )

    model_path = config.RAFT_STEREO_MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"RAFT-Stereo weights not found: {model_path}\n"
            "Download with: cd models && wget https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip && unzip models.zip"
        )

    # Add RAFT-Stereo core to path for imports
    raft_core_path = _get_raft_stereo_path()
    if raft_core_path not in sys.path:
        sys.path.insert(0, raft_core_path)

    from core.raft_stereo import RAFTStereo

    # Select device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        print(f"  RAFT-Stereo device: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device("cpu")
        print("  RAFT-Stereo device: CPU (this will be slow)")

    args = _build_args()

    # Enable mixed precision on GPU for memory efficiency
    if _device.type == "cuda":
        args.mixed_precision = True

    # Load model — weights are saved with DataParallel wrapper
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    state_dict = torch.load(model_path, map_location=_device, weights_only=False)
    model.load_state_dict(state_dict)

    model = model.module
    model.to(_device)
    model.eval()

    _model = model
    print(f"  RAFT-Stereo model loaded: {os.path.basename(model_path)}")

    return _model, _device


def _prepare_image(image_bgr, device):
    """
    Convert a BGR uint8 image to the tensor format RAFT-Stereo expects.

    Args:
        image_bgr: numpy array (H, W, 3) BGR uint8
        device: torch.device

    Returns:
        tensor: (1, 3, H, W) float32 tensor in RGB order
    """
    # BGR → RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # HWC → CHW, add batch dimension
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
    tensor = tensor[None].to(device)

    return tensor


def compute_disparity_raft(rect_left, rect_right):
    """
    Compute disparity map using RAFT-Stereo deep learning model.

    Drop-in replacement for disparity.compute_disparity(). Returns the
    same format so downstream stages (depth map, point cloud) work unchanged.

    Args:
        rect_left: Rectified left image (BGR, uint8, HxWx3)
        rect_right: Rectified right image (BGR, uint8, HxWx3)

    Returns:
        disparity_raw: Raw disparity map (float32, H×W)
        disparity_filtered: Same as raw — RAFT-Stereo output is already
                           smooth and does not need WLS filtering
    """
    model, device = _load_model()

    h_orig, w_orig = rect_left.shape[:2]
    print(f"  Input size: {w_orig}×{h_orig}")

    # ── Resize for VRAM safety (RTX 3050 = 4GB) ────────────────────────
    max_h = config.RAFT_STEREO_MAX_HEIGHT
    max_w = config.RAFT_STEREO_MAX_WIDTH
    scale = 1.0

    if h_orig > max_h or w_orig > max_w:
        scale = min(max_h / h_orig, max_w / w_orig)
        new_h = int(h_orig * scale)
        new_w = int(w_orig * scale)
        left_resized = cv2.resize(rect_left, (new_w, new_h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(rect_right, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Resized to {new_w}×{new_h} for VRAM safety (scale={scale:.3f})")
    else:
        left_resized = rect_left
        right_resized = rect_right

    # ── Convert to tensors ──────────────────────────────────────────────
    image1 = _prepare_image(left_resized, device)
    image2 = _prepare_image(right_resized, device)

    # ── Pad to be divisible by 32 ──────────────────────────────────────
    # RAFT-Stereo requires dimensions divisible by 2^n_downsample * 8 = 32
    raft_core_path = _get_raft_stereo_path()
    if raft_core_path not in sys.path:
        sys.path.insert(0, raft_core_path)
    from core.utils.utils import InputPadder

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    # ── Run inference ───────────────────────────────────────────────────
    iters = config.RAFT_STEREO_VALID_ITERS

    with torch.no_grad():
        _, flow_up = model(image1, image2, iters=iters, test_mode=True)

    # Unpad
    flow_up = padder.unpad(flow_up).squeeze()

    # RAFT-Stereo outputs negative disparity (left → right flow)
    # Convert to positive disparity to match SGBM convention
    disparity = -flow_up.cpu().numpy()

    # Clamp any negative values (invalid)
    disparity = np.maximum(disparity, 0).astype(np.float32)

    # ── Upscale if we resized earlier ───────────────────────────────────
    if scale < 1.0:
        disparity = cv2.resize(disparity, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        # Scale disparity values proportionally
        disparity = disparity / scale
        print(f"  Upscaled disparity back to {w_orig}×{h_orig}")

    valid_count = np.sum(disparity > 0)
    total = disparity.size
    if valid_count > 0:
        valid_vals = disparity[disparity > 0]
        print(f"  Disparity range: [{valid_vals.min():.1f}, {valid_vals.max():.1f}]")
    print(f"  Valid disparity pixels: {valid_count}/{total} ({valid_count / total * 100:.1f}%)")

    # Free GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Return both raw and filtered (same for RAFT-Stereo — no WLS needed)
    return disparity, disparity
