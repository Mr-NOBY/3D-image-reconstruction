"""
Stereo Rectification Module
============================
Transforms the image pair so that epipolar lines become horizontal.

Two strategies:
  1. Calibrated path: uses recovered R, t with estimated K → proper Q matrix
  2. Uncalibrated path: uses stereoRectifyUncalibrated → homographies only
"""

import cv2
import numpy as np
import config


def rectify_uncalibrated(img_left, img_right, pts_left, pts_right, F, mask=None):
    """Perform uncalibrated stereo rectification using the Fundamental Matrix."""
    h, w = img_left.shape[:2]

    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
        pts_l = pts_left[mask_flat]
        pts_r = pts_right[mask_flat]
    else:
        pts_l = pts_left
        pts_r = pts_right

    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_l.reshape(-1, 2), pts_r.reshape(-1, 2),
        F, imgSize=(w, h)
    )

    if not retval:
        raise RuntimeError("Uncalibrated stereo rectification failed.")

    rect_left = cv2.warpPerspective(img_left, H1, (w, h))
    rect_right = cv2.warpPerspective(img_right, H2, (w, h))

    print(f"  Uncalibrated rectification successful")
    return rect_left, rect_right, H1, H2


def rectify_calibrated(img_left, img_right, K, dist_coeffs, R, t):
    """
    Perform calibrated stereo rectification with Q matrix output.
    
    Even with estimated (not ground-truth) intrinsics, this path produces
    a geometrically valid Q matrix, which is critical for accurate 3D
    reprojection via cv2.reprojectImageTo3D().
    """
    h, w = img_left.shape[:2]
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist_coeffs, K, dist_coeffs, (w, h), R, t,
        alpha=0,        # 0 = crop to valid pixels only
        newImageSize=(w, h)
    )

    # Compute the rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1
    )

    # Apply the maps to rectify the images
    rect_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

    print(f"  Calibrated rectification successful")
    print(f"  Q matrix diagonal: [{Q[0,0]:.4f}, {Q[1,1]:.4f}, {Q[2,2]:.4f}, {Q[3,3]:.4f}]")
    return rect_left, rect_right, Q


def build_Q_matrix_uncalibrated(img_shape, K=None):
    """Build an approximate Q matrix for uncalibrated reconstruction."""
    if K is None:
        K = config.get_default_camera_matrix(img_shape)

    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    Tx = 1.0

    Q = np.array([
        [1, 0, 0,      -cx],
        [0, 1, 0,      -cy],
        [0, 0, 0,       f ],
        [0, 0, -1/Tx,   0 ]
    ], dtype=np.float64)
    return Q


def draw_rectified(rect_left, rect_right, num_lines=20):
    """Side-by-side rectified images with horizontal alignment lines."""
    h, w = rect_left.shape[:2]
    vis = np.hstack([rect_left, rect_right])
    step = h // (num_lines + 1)
    for i in range(1, num_lines + 1):
        y = i * step
        cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)
    return vis
