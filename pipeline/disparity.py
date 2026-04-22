"""
Disparity Map Module
====================
Computes the disparity map from a rectified stereo pair using StereoSGBM.

Disparity = x_left - x_right (horizontal pixel offset between matching points).
Depth is inversely proportional to disparity: depth = (f * baseline) / disparity.
"""

import cv2
import numpy as np
import config


def compute_disparity(rect_left, rect_right):
    """
    Compute disparity map using StereoSGBM with WLS filtering.
    
    Args:
        rect_left: Rectified left image (BGR)
        rect_right: Rectified right image (BGR)
    
    Returns:
        disparity_raw: Raw disparity map (float32)
        disparity_filtered: WLS-filtered disparity map (float32)
    """
    # Convert to grayscale for matching
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # Improve contrast with CLAHE (helps stereo matching a lot)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_left = clahe.apply(gray_left)
    gray_right = clahe.apply(gray_right)

    # Dynamically determine numDisparities based on image width
    # Rule of thumb: ~25% of the image width, rounded up to multiple of 16
    w = gray_left.shape[1]
    num_disp = max(16, ((w // 4) // 16) * 16)
    num_disp = min(num_disp, config.SGBM_NUM_DISPARITIES)  # respect config cap

    block_size = config.SGBM_BLOCK_SIZE
    channels = 3  # original images are BGR

    # Create left matcher (StereoSGBM)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=config.SGBM_MIN_DISPARITY,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * channels * block_size ** 2,
        P2=32 * channels * block_size ** 2,
        disp12MaxDiff=config.SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=config.SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=config.SGBM_SPECKLE_WINDOW_SIZE,
        speckleRange=config.SGBM_SPECKLE_RANGE,
        preFilterCap=config.SGBM_PRE_FILTER_CAP,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY
    )

    # Create right matcher for WLS filter
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Compute disparities
    disp_left = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disp_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

    print(f"  numDisparities used: {num_disp}")
    print(f"  Raw disparity range: [{disp_left.min():.1f}, {disp_left.max():.1f}]")

    # WLS (Weighted Least Squares) filter for smoother results
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(config.WLS_LAMBDA)
    wls_filter.setSigmaColor(config.WLS_SIGMA)

    disparity_filtered = wls_filter.filter(disp_left, gray_left, None, disp_right)

    # Replace any NaN/Inf values from WLS filter with 0 (invalid)
    disparity_filtered = np.nan_to_num(disparity_filtered, nan=0.0, posinf=0.0, neginf=0.0)

    # If WLS filter produced no valid data, fall back to raw disparity
    if not np.any(disparity_filtered > 0):
        print("  WLS filter produced no valid disparities, using raw disparity")
        disparity_filtered = np.nan_to_num(disp_left, nan=0.0, posinf=0.0, neginf=0.0)

    valid_count = np.sum(disparity_filtered > 0)
    total = disparity_filtered.size
    print(f"  Filtered disparity range: [{disparity_filtered[disparity_filtered > 0].min():.1f}, {disparity_filtered.max():.1f}]" if valid_count > 0 else "  No valid filtered disparities")
    print(f"  Valid disparity pixels: {valid_count}/{total} ({valid_count/total*100:.1f}%)")

    return disp_left, disparity_filtered


def normalize_disparity(disparity):
    """
    Normalize disparity map to 0-255 range for visualization.
    
    Args:
        disparity: Disparity map (float32)
    
    Returns:
        normalized: Normalized disparity (uint8)
    """
    # Replace NaN/Inf and only consider positive disparities as valid
    disp_clean = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
    valid = disp_clean > 0

    if not valid.any():
        return np.zeros(disparity.shape[:2], dtype=np.uint8)

    # Normalize to 0-255 using only valid range
    d_min = float(disp_clean[valid].min())
    d_max = float(disp_clean[valid].max())

    if d_max <= d_min:
        return np.zeros(disparity.shape[:2], dtype=np.uint8)

    normalized = np.zeros(disparity.shape[:2], dtype=np.uint8)
    normalized[valid] = np.clip(
        ((disp_clean[valid] - d_min) / (d_max - d_min) * 255), 0, 255
    ).astype(np.uint8)

    return normalized


def colorize_disparity(disparity):
    """
    Apply a colormap to the disparity map for better visualization.
    
    Args:
        disparity: Disparity map (float32)
    
    Returns:
        colored: Colorized disparity map (BGR, uint8)
    """
    normalized = normalize_disparity(disparity)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    # Black out invalid areas
    colored[normalized == 0] = 0
    return colored
