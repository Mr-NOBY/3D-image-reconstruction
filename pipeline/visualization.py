"""
Visualization Module
====================
Open3D interactive viewer and utility functions for saving visualizations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def view_point_cloud(ply_path=None, points=None, colors=None):
    """
    Open an interactive 3D viewer for the point cloud.
    
    Args:
        ply_path: Path to PLY file (used if points/colors not provided)
        points: 3D points array (N, 3)
        colors: RGB colors array (N, 3) normalized [0, 1]
    """
    if not HAS_OPEN3D:
        print("Open3D not available. Use the web viewer instead.")
        return
    
    if points is not None and colors is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif ply_path:
        pcd = o3d.io.read_point_cloud(ply_path)
    else:
        ply_path = config.POINT_CLOUD_OUTPUT
        pcd = o3d.io.read_point_cloud(ply_path)
    
    print(f"  Displaying {len(pcd.points)} points...")
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="3D Reconstruction - Point Cloud",
        width=1280, height=720,
        point_show_normal=False
    )


def save_visualization(image, filepath, title=None):
    """Save a visualization image with optional title."""
    if title:
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        cv2.imwrite(filepath, image)
    
    print(f"  Saved: {filepath}")


def create_pipeline_summary(img_left, img_right, matches_vis, epipolar_vis,
                            rectified_vis, disparity_vis, depth_vis):
    """
    Create a summary grid of all pipeline stages.
    
    Args:
        All intermediate visualization images
    
    Returns:
        summary: Combined summary image
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle("3D Reconstruction Pipeline Summary", fontsize=18, fontweight='bold')
    
    # Input images
    combined_input = np.hstack([
        cv2.resize(img_left, (400, 300)),
        cv2.resize(img_right, (400, 300))
    ])
    axes[0, 0].imshow(cv2.cvtColor(combined_input, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Input Stereo Pair")
    axes[0, 0].axis('off')
    
    # Feature matches
    if matches_vis is not None:
        axes[0, 1].imshow(cv2.cvtColor(cv2.resize(matches_vis, (800, 300)), cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("2. Feature Matches (SIFT + FLANN)")
        axes[0, 1].axis('off')
    
    # Epipolar lines
    if epipolar_vis is not None:
        axes[1, 0].imshow(cv2.cvtColor(cv2.resize(epipolar_vis, (800, 300)), cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("3. Epipolar Lines")
        axes[1, 0].axis('off')
    
    # Rectified images
    if rectified_vis is not None:
        axes[1, 1].imshow(cv2.cvtColor(cv2.resize(rectified_vis, (800, 300)), cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("4. Rectified Stereo Pair")
        axes[1, 1].axis('off')
    
    # Disparity map
    if disparity_vis is not None:
        axes[2, 0].imshow(cv2.cvtColor(cv2.resize(disparity_vis, (400, 300)), cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title("5. Disparity Map (StereoSGBM + WLS)")
        axes[2, 0].axis('off')
    
    # Depth map
    if depth_vis is not None:
        axes[2, 1].imshow(cv2.cvtColor(cv2.resize(depth_vis, (400, 300)), cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title("6. Depth Map")
        axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    summary_path = config.OUTPUT_DIR + "/pipeline_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Pipeline summary saved: {summary_path}")
    return summary_path
