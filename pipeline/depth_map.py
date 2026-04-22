"""
Depth Map Module
================
Converts disparity values to depth values.
Depth = (focal_length * baseline) / disparity
"""

import cv2
import numpy as np
import config


def disparity_to_depth(disparity, K=None, baseline=1.0):
    """
    Convert disparity map to depth map.
    
    Args:
        disparity: Disparity map (float32)
        K: Camera intrinsic matrix (optional)
        baseline: Distance between cameras (default 1.0 for relative depth)
    
    Returns:
        depth_map: Depth map (float32)
    """
    # Only positive disparities are valid
    valid_mask = disparity > 0

    if K is not None:
        focal_length = K[0, 0]
    else:
        focal_length = max(disparity.shape[:2])

    depth_map = np.zeros_like(disparity, dtype=np.float32)
    depth_map[valid_mask] = (focal_length * baseline) / disparity[valid_mask]

    # Clamp extreme values
    depth_map = np.clip(depth_map, 0, config.PC_DEPTH_MAX)

    valid_count = np.sum(valid_mask)
    if valid_count > 0:
        valid_depths = depth_map[valid_mask & (depth_map > 0)]
        if len(valid_depths) > 0:
            print(f"  Depth range: [{valid_depths.min():.2f}, {np.percentile(valid_depths, 99):.2f}]")
            print(f"  Valid pixels: {valid_count} / {disparity.size} ({valid_count/disparity.size*100:.1f}%)")
    else:
        print("  Warning: no valid depth values")

    return depth_map


def colorize_depth(depth_map):
    """
    Apply a colormap to the depth map for visualization.
    Closer objects appear warmer, farther objects appear cooler.
    
    Args:
        depth_map: Depth map (float32)
    
    Returns:
        colored: Colorized depth map (BGR, uint8)
    """
    valid = depth_map > 0
    depth_vis = np.zeros(depth_map.shape, dtype=np.uint8)

    if valid.any():
        valid_depths = depth_map[valid]
        # Use percentiles to avoid outlier distortion
        d_min = np.percentile(valid_depths, 2)
        d_max = np.percentile(valid_depths, 98)

        if d_max > d_min:
            # Normalize to [0, 255]
            normalized = np.clip(depth_map, d_min, d_max)
            normalized = ((normalized - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            # Invert: closer = brighter (higher value)
            normalized = 255 - normalized
            normalized[~valid] = 0
            depth_vis = normalized

    colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0

    return colored
