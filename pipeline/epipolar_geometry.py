"""
Epipolar Geometry Module
========================
Computes the Fundamental Matrix (F) and optionally the Essential Matrix (E)
from matched point correspondences between two views.

Theory:
    The Fundamental Matrix F encodes the epipolar constraint:
        x'^T · F · x = 0
    where x and x' are corresponding points in homogeneous coordinates.
    
    F maps a point in one image to its epipolar line in the other image.
    All epipolar lines in one image converge at the epipole — the projection
    of the other camera's center.
    
    The Essential Matrix E relates to F through the camera intrinsics K:
        E = K'^T · F · K
    
    E can be decomposed into relative rotation R and translation t between
    the two camera positions:
        E = [t]_× · R
    
    This decomposition gives us the camera poses needed for triangulation.
"""

import cv2
import numpy as np
import config


def compute_fundamental_matrix(pts_left, pts_right):
    """
    Estimate the Fundamental Matrix using RANSAC.
    
    RANSAC (Random Sample Consensus) iteratively:
    1. Selects a random subset of 8 point correspondences
    2. Computes F from those 8 points (8-point algorithm)
    3. Counts how many other correspondences agree (inliers)
    4. Keeps the F with the most inliers
    
    Args:
        pts_left: Matched points in left image (N, 2)
        pts_right: Matched points in right image (N, 2)
    
    Returns:
        F: Fundamental Matrix (3, 3)
        mask: Inlier mask (N, 1) — 1 for inliers, 0 for outliers
    """
    F, mask = cv2.findFundamentalMat(
        pts_left, pts_right,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
        confidence=config.RANSAC_CONFIDENCE
    )
    
    inlier_count = np.sum(mask) if mask is not None else 0
    total_count = len(pts_left)
    inlier_ratio = inlier_count / total_count * 100
    
    print(f"  Fundamental Matrix estimated")
    print(f"  Inliers: {inlier_count}/{total_count} ({inlier_ratio:.1f}%)")
    
    return F, mask


def compute_essential_matrix(F, K):
    """
    Compute the Essential Matrix from the Fundamental Matrix and camera intrinsics.
    
    E = K^T · F · K
    
    Args:
        F: Fundamental Matrix (3, 3)
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        E: Essential Matrix (3, 3)
    """
    E = K.T @ F @ K
    
    # Enforce the constraint that E has two equal singular values
    U, S, Vt = np.linalg.svd(E)
    S = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
    E = U @ np.diag(S) @ Vt
    
    print(f"  Essential Matrix computed from F and K")
    
    return E


def recover_camera_pose(E, pts_left, pts_right, K):
    """
    Decompose the Essential Matrix to recover relative camera pose (R, t).
    
    cv2.recoverPose() tests all 4 possible decompositions (2 rotations × 2 
    translations) and returns the one where the triangulated points are in 
    front of both cameras (positive depth — the cheirality check).
    
    Args:
        E: Essential Matrix (3, 3)
        pts_left: Inlier points in left image (N, 2)
        pts_right: Inlier points in right image (N, 2)
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        R: Rotation matrix (3, 3) from camera 1 to camera 2
        t: Translation vector (3, 1) — unit vector, direction only
        good_count: Number of points that pass the cheirality check
    """
    good_count, R, t, pose_mask = cv2.recoverPose(E, pts_left, pts_right, K)
    
    print(f"  Camera pose recovered")
    print(f"  Points passing cheirality check: {good_count}")
    
    return R, t, good_count


def compute_epipolar_lines(pts, F, which_image):
    """
    Compute epipolar lines for points in one image as seen in the other.
    
    For a point x in image 1, the epipolar line in image 2 is:
        l' = F · x    (which_image=2)
    
    For a point x' in image 2, the epipolar line in image 1 is:
        l = F^T · x'  (which_image=1)
    
    Args:
        pts: Points in one image (N, 2)
        F: Fundamental Matrix (3, 3)
        which_image: Which image the lines should appear in (1 or 2)
    
    Returns:
        lines: Epipolar lines (N, 3) — each line as [a, b, c]: ax + by + c = 0
    """
    pts_hom = pts.reshape(-1, 1, 2)
    lines = cv2.computeCorrespondEpilines(pts_hom, which_image, F)
    lines = lines.reshape(-1, 3)
    return lines


def draw_epipolar_lines(img_left, img_right, pts_left, pts_right, F, num_lines=30):
    """
    Draw epipolar lines on both images for visualization.
    
    For each selected point pair:
    - Draw the point on one image and its epipolar line on the other
    - Use consistent colors so matching pairs are visually linked
    
    Args:
        img_left: Left image (BGR)
        img_right: Right image (BGR)
        pts_left: Points in left image (N, 2)
        pts_right: Points in right image (N, 2)
        F: Fundamental Matrix (3, 3)
        num_lines: Number of epipolar lines to draw
    
    Returns:
        vis_left: Left image with epipolar lines and points
        vis_right: Right image with epipolar lines and points
    """
    vis_left = img_left.copy()
    vis_right = img_right.copy()
    
    h_l, w_l = vis_left.shape[:2]
    h_r, w_r = vis_right.shape[:2]
    
    # Select a subset of points to draw
    indices = np.random.choice(len(pts_left), min(num_lines, len(pts_left)), replace=False)
    
    selected_left = pts_left[indices]
    selected_right = pts_right[indices]
    
    # Compute epipolar lines in right image from left points
    lines_right = compute_epipolar_lines(selected_left, F, which_image=1)
    # Compute epipolar lines in left image from right points
    lines_left = compute_epipolar_lines(selected_right, F, which_image=2)
    
    for i in range(len(indices)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw on left image: point + epipolar line from right
        a, b, c = lines_left[i]
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = w_l, int(-(c + a * w_l) / b) if b != 0 else 0
        cv2.line(vis_left, (x0, y0), (x1, y1), color, 1)
        pt = tuple(selected_left[i].astype(int))
        cv2.circle(vis_left, pt, 5, color, -1)
        
        # Draw on right image: point + epipolar line from left
        a, b, c = lines_right[i]
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = w_r, int(-(c + a * w_r) / b) if b != 0 else 0
        cv2.line(vis_right, (x0, y0), (x1, y1), color, 1)
        pt = tuple(selected_right[i].astype(int))
        cv2.circle(vis_right, pt, 5, color, -1)
    
    return vis_left, vis_right
