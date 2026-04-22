"""
Feature Detection Module
========================
Detects keypoints and computes descriptors using SIFT (Scale-Invariant Feature Transform).

SIFT works by:
1. Building a scale-space pyramid (Difference of Gaussians)
2. Detecting local extrema as candidate keypoints
3. Refining keypoint location and filtering weak/edge responses
4. Assigning dominant orientation to each keypoint
5. Computing a 128-dimensional descriptor for each keypoint

The descriptors are invariant to scale, rotation, and partially to
illumination changes — making SIFT ideal for matching across viewpoints.
"""

import cv2
import numpy as np
import config


def load_image_pair(left_path=None, right_path=None):
    """
    Load a pair of stereo images.
    
    Args:
        left_path: Path to the left image (default: config path)
        right_path: Path to the right image (default: config path)
    
    Returns:
        img_left: Left image (BGR)
        img_right: Right image (BGR)
    
    Raises:
        FileNotFoundError: If either image cannot be loaded
    """
    left_path = left_path or config.LEFT_IMAGE_PATH
    right_path = right_path or config.RIGHT_IMAGE_PATH
    
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    
    if img_left is None:
        raise FileNotFoundError(f"Could not load left image: {left_path}")
    if img_right is None:
        raise FileNotFoundError(f"Could not load right image: {right_path}")
    
    print(f"  Left image:  {img_left.shape[1]}x{img_left.shape[0]} px")
    print(f"  Right image: {img_right.shape[1]}x{img_right.shape[0]} px")
    
    return img_left, img_right


def detect_features(image, name="image"):
    """
    Detect SIFT keypoints and compute descriptors for a single image.
    
    Args:
        image: Input image (BGR or grayscale)
        name: Name label for logging
    
    Returns:
        keypoints: List of cv2.KeyPoint objects
        descriptors: numpy array of shape (N, 128) — SIFT descriptors
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create SIFT detector with configured parameters
    sift = cv2.SIFT_create(
        nfeatures=config.SIFT_N_FEATURES,
        nOctaveLayers=config.SIFT_N_OCTAVE_LAYERS,
        contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
        edgeThreshold=config.SIFT_EDGE_THRESHOLD,
        sigma=config.SIFT_SIGMA
    )
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    print(f"  {name}: {len(keypoints)} keypoints detected")
    
    return keypoints, descriptors


def detect_features_pair(img_left, img_right):
    """
    Detect features in both images of the stereo pair.
    
    Args:
        img_left: Left image (BGR)
        img_right: Right image (BGR)
    
    Returns:
        kp_left, desc_left: Keypoints and descriptors for left image
        kp_right, desc_right: Keypoints and descriptors for right image
    """
    kp_left, desc_left = detect_features(img_left, "Left")
    kp_right, desc_right = detect_features(img_right, "Right")
    
    return kp_left, desc_left, kp_right, desc_right


def draw_keypoints(image, keypoints, name="keypoints"):
    """
    Draw detected keypoints on an image for visualization.
    
    Args:
        image: Input image (BGR)
        keypoints: List of cv2.KeyPoint objects
        name: Label for the output
    
    Returns:
        vis: Image with keypoints drawn (with size and orientation)
    """
    vis = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    return vis
