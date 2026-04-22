"""
Configuration parameters for the 3D Reconstruction pipeline.
All tunable parameters are centralized here for easy experimentation.
"""

import os
import numpy as np

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LEFT_IMAGE_PATH = os.path.join(IMAGES_DIR, "left.jpg")
RIGHT_IMAGE_PATH = os.path.join(IMAGES_DIR, "right.jpg")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Feature Detection (SIFT)
# ──────────────────────────────────────────────
SIFT_N_FEATURES = 0            # 0 = detect all features
SIFT_N_OCTAVE_LAYERS = 3       # Number of layers in each octave
SIFT_CONTRAST_THRESHOLD = 0.04 # Filter out weak features
SIFT_EDGE_THRESHOLD = 10       # Filter out edge-like features
SIFT_SIGMA = 1.6               # Gaussian sigma for the first octave

# ──────────────────────────────────────────────
# Feature Matching (FLANN)
# ──────────────────────────────────────────────
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5                # Number of KD-trees for indexing
FLANN_CHECKS = 50              # Number of checks during search
LOWE_RATIO_THRESHOLD = 0.75   # Lowe's ratio test threshold
MIN_MATCH_COUNT = 10           # Minimum matches required to proceed

# ──────────────────────────────────────────────
# Epipolar Geometry
# ──────────────────────────────────────────────
RANSAC_REPROJ_THRESHOLD = 1.0  # RANSAC inlier threshold (pixels)
RANSAC_CONFIDENCE = 0.999      # RANSAC confidence level

# ──────────────────────────────────────────────
# Stereo Matching (StereoSGBM)
# ──────────────────────────────────────────────
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 128     # Must be divisible by 16
SGBM_BLOCK_SIZE = 5            # Odd number, typically 3-11
SGBM_P1 = 8 * 3 * 5 ** 2      # Penalty on disparity changes (small)
SGBM_P2 = 32 * 3 * 5 ** 2     # Penalty on disparity changes (large)
SGBM_DISP12_MAX_DIFF = 1       # Max allowed difference in L-R check
SGBM_UNIQUENESS_RATIO = 10     # Margin in % for best match uniqueness
SGBM_SPECKLE_WINDOW_SIZE = 100 # Max size of smooth disparity regions
SGBM_SPECKLE_RANGE = 2         # Max disparity variation in speckle region
SGBM_PRE_FILTER_CAP = 63
SGBM_MODE = 1                  # cv2.StereoSGBM_MODE_SGBM_3WAY

# WLS Filter parameters
WLS_LAMBDA = 8000              # Regularization parameter
WLS_SIGMA = 1.5                # Sensitivity parameter

# ──────────────────────────────────────────────
# Camera Intrinsics (estimated / default)
# ──────────────────────────────────────────────
# These are estimated defaults. For uncalibrated images,
# focal length is approximated as max(image_width, image_height).
# Override these if you have actual calibration data.
USE_CALIBRATED = False         # Set True if you have real K matrix

# Default intrinsic matrix (will be updated based on image size)
def get_default_camera_matrix(image_shape):
    """
    Estimate a default camera intrinsic matrix K.
    Focal length is approximated as the max dimension of the image.
    Principal point is assumed to be the image center.
    """
    h, w = image_shape[:2]
    focal_length = max(h, w)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1 ]
    ], dtype=np.float64)
    return K

# ──────────────────────────────────────────────
# Point Cloud
# ──────────────────────────────────────────────
POINT_CLOUD_OUTPUT = os.path.join(OUTPUT_DIR, "point_cloud.ply")
DEPTH_MAP_OUTPUT = os.path.join(OUTPUT_DIR, "depth_map.png")
DISPARITY_MAP_OUTPUT = os.path.join(OUTPUT_DIR, "disparity_map.png")
MATCHES_OUTPUT = os.path.join(OUTPUT_DIR, "matches.jpg")
EPIPOLAR_OUTPUT = os.path.join(OUTPUT_DIR, "epipolar_lines.jpg")
RECTIFIED_OUTPUT = os.path.join(OUTPUT_DIR, "rectified.jpg")

# Point cloud filtering
PC_DEPTH_MIN = 0               # Minimum depth to include (0 = auto)
PC_DEPTH_MAX = 10000           # Maximum depth to include
PC_STATISTICAL_OUTLIER_NB = 20 # Neighbors for statistical outlier removal
PC_STATISTICAL_OUTLIER_STD = 2.0  # Std ratio for outlier removal
