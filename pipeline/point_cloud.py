"""
Point Cloud Module
==================
Generates a 3D point cloud from the disparity map using cv2.reprojectImageTo3D().
Exports the result as a PLY file for visualization in Open3D or web viewers.
"""

import cv2
import numpy as np
import config

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. PLY export will use manual writer.")


def reproject_to_3d(disparity, Q, color_image):
    """
    Reproject disparity map to 3D point cloud.
    
    Uses the Q matrix to map (x, y, disparity) -> (X, Y, Z).
    
    Args:
        disparity: Disparity map (float32)
        Q: Disparity-to-depth matrix (4, 4)
        color_image: Original color image for point colors (BGR)
    
    Returns:
        points_3d: Array of 3D points (N, 3)
        colors: Array of RGB colors (N, 3) normalized to [0, 1]
    """
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # Create mask for valid points — must have:
    # 1. Positive disparity
    # 2. Finite coordinates
    # 3. Reasonable Z range (reject extreme outliers)
    disp_valid = disparity > 0
    finite_mask = (
        np.isfinite(points_3d[:, :, 0]) &
        np.isfinite(points_3d[:, :, 1]) &
        np.isfinite(points_3d[:, :, 2])
    )

    # Use percentile-based Z filtering to reject outliers
    if disp_valid.any() and finite_mask.any():
        combined = disp_valid & finite_mask
        z_vals = points_3d[:, :, 2][combined]
        if len(z_vals) > 100:
            z_low = np.percentile(z_vals, 5)
            z_high = np.percentile(z_vals, 95)
            z_mask = (points_3d[:, :, 2] >= z_low) & (points_3d[:, :, 2] <= z_high)
        else:
            z_mask = np.abs(points_3d[:, :, 2]) < config.PC_DEPTH_MAX
    else:
        z_mask = np.abs(points_3d[:, :, 2]) < config.PC_DEPTH_MAX

    mask = disp_valid & finite_mask & z_mask

    # Also ensure the color image pixel is not pure black (often invalid areas)
    color_sum = np.sum(color_image, axis=2)
    mask = mask & (color_sum > 10)

    # Extract valid points and their colors
    valid_points = points_3d[mask]
    valid_colors = color_image[mask]

    # Convert BGR to RGB and normalize to [0, 1]
    valid_colors = valid_colors[:, ::-1].astype(np.float64) / 255.0

    # -- Normalize point cloud geometry --
    # With estimated (uncalibrated) camera intrinsics, the Q matrix produces
    # 3D coordinates where depth (Z) can be on a completely different scale
    # than X and Y. This creates the "flat billboard" effect where the point
    # cloud looks correct from the front but collapses when rotated.
    #
    # Fix: center the cloud and normalize each axis independently so that
    # the spatial extent in X, Y, and Z are comparable. This preserves the
    # relative structure while making depth visually meaningful.
    if len(valid_points) > 100:
        centroid = np.median(valid_points, axis=0)
        valid_points = valid_points - centroid

        # Scale each axis so the interquartile range is normalized
        for axis in range(3):
            q25 = np.percentile(valid_points[:, axis], 10)
            q75 = np.percentile(valid_points[:, axis], 90)
            iqr = q75 - q25
            if iqr > 1e-6:
                valid_points[:, axis] /= iqr

    print(f"  Valid 3D points: {len(valid_points)}")

    return valid_points, valid_colors


def filter_point_cloud(points, colors):
    """
    Filter the point cloud using statistical outlier removal.
    
    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3)
    
    Returns:
        filtered_points: Filtered 3D points
        filtered_colors: Filtered colors
    """
    if not HAS_OPEN3D or len(points) == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Statistical outlier removal
    pcd_filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=config.PC_STATISTICAL_OUTLIER_NB,
        std_ratio=config.PC_STATISTICAL_OUTLIER_STD
    )

    filtered_points = np.asarray(pcd_filtered.points)
    filtered_colors = np.asarray(pcd_filtered.colors)

    removed = len(points) - len(filtered_points)
    print(f"  Outlier removal: {removed} points removed, {len(filtered_points)} remaining")

    return filtered_points, filtered_colors


def save_ply(points, colors, filepath=None):
    """
    Save point cloud as PLY file.
    
    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) normalized [0, 1]
        filepath: Output path (default: config path)
    """
    filepath = filepath or config.POINT_CLOUD_OUTPUT

    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filepath, pcd)
    else:
        _save_ply_manual(points, colors, filepath)

    print(f"  Point cloud saved: {filepath} ({len(points)} points)")


def _save_ply_manual(points, colors, filepath):
    """Manually write a PLY file without Open3D."""
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")
