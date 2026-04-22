"""
Feature Matching Module
=======================
Matches SIFT descriptors between two images using FLANN (Fast Library for
Approximate Nearest Neighbors) and filters results with Lowe's Ratio Test.

Lowe's Ratio Test:
    For each descriptor in image 1, find the 2 nearest matches in image 2.
    Accept the match only if:  distance(best) / distance(second_best) < threshold
    
    This eliminates ambiguous matches where multiple descriptors are
    equally similar, keeping only distinctive, reliable correspondences.
"""

import cv2
import numpy as np
import config


def match_features(desc_left, desc_right):
    """
    Match descriptors using FLANN-based matcher with Lowe's ratio test.
    
    Args:
        desc_left: Descriptors from the left image (N, 128)
        desc_right: Descriptors from the right image (M, 128)
    
    Returns:
        good_matches: List of cv2.DMatch objects that pass the ratio test
    
    Raises:
        ValueError: If not enough matches are found
    """
    # FLANN parameters for SIFT (float descriptors → KD-Tree)
    index_params = dict(
        algorithm=config.FLANN_INDEX_KDTREE,
        trees=config.FLANN_TREES
    )
    search_params = dict(checks=config.FLANN_CHECKS)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find 2 nearest neighbors for ratio test
    matches = flann.knnMatch(desc_left, desc_right, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < config.LOWE_RATIO_THRESHOLD * n.distance:
            good_matches.append(m)
    
    print(f"  Total matches: {len(matches)}")
    print(f"  Good matches (after ratio test): {len(good_matches)}")
    
    if len(good_matches) < config.MIN_MATCH_COUNT:
        raise ValueError(
            f"Not enough matches found: {len(good_matches)} < {config.MIN_MATCH_COUNT}. "
            "Try adjusting LOWE_RATIO_THRESHOLD or using images with more texture."
        )
    
    return good_matches


def extract_matched_points(kp_left, kp_right, matches):
    """
    Extract matched point coordinates from keypoints and match objects.
    
    Args:
        kp_left: Keypoints from the left image
        kp_right: Keypoints from the right image
        matches: List of cv2.DMatch objects
    
    Returns:
        pts_left: numpy array of shape (N, 2) — matched points in left image
        pts_right: numpy array of shape (N, 2) — matched points in right image
    """
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches])
    
    return pts_left, pts_right


def draw_matches(img_left, kp_left, img_right, kp_right, matches, max_draw=200):
    """
    Draw feature matches between two images for visualization.
    
    Args:
        img_left: Left image (BGR)
        kp_left: Keypoints from left image
        img_right: Right image (BGR)
        kp_right: Keypoints from right image
        matches: List of cv2.DMatch objects
        max_draw: Maximum number of matches to draw (for clarity)
    
    Returns:
        vis: Visualization image with match lines drawn
    """
    # Sort matches by distance (best first)
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    draw_count = min(max_draw, len(sorted_matches))
    
    vis = cv2.drawMatches(
        img_left, kp_left,
        img_right, kp_right,
        sorted_matches[:draw_count], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return vis
