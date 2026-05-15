"""
3D Reconstruction from Two Images
===================================
Main entry point that orchestrates the full stereo reconstruction pipeline.

Usage:
    python main.py
    python main.py --left path/to/left.jpg --right path/to/right.jpg
    python main.py --no-viewer      # Skip Open3D viewer
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline.feature_detection import load_image_pair, detect_features_pair, draw_keypoints
from pipeline.feature_matching import match_features, extract_matched_points, draw_matches
from pipeline.epipolar_geometry import (
    compute_fundamental_matrix, compute_essential_matrix,
    recover_camera_pose, draw_epipolar_lines
)
from pipeline.rectification import (
    rectify_uncalibrated, rectify_calibrated,
    build_Q_matrix_uncalibrated, draw_rectified
)
from pipeline.disparity import compute_disparity, colorize_disparity
from pipeline.depth_map import disparity_to_depth, colorize_depth
from pipeline.point_cloud import reproject_to_3d, filter_point_cloud, save_ply
from pipeline.visualization import (
    view_point_cloud, save_visualization, create_pipeline_summary
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D Reconstruction from Two Images")
    parser.add_argument("--left", type=str, default=config.LEFT_IMAGE_PATH,
                        help="Path to the left image")
    parser.add_argument("--right", type=str, default=config.RIGHT_IMAGE_PATH,
                        help="Path to the right image")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Skip Open3D interactive viewer")
    return parser.parse_args()


def print_banner():
    print("=" * 60)
    print("   3D RECONSTRUCTION FROM TWO IMAGES")
    print("   Stereo Vision Pipeline")
    print("=" * 60)
    print()


def run_pipeline(left_path, right_path, open_viewer=True):
    """
    Run the full 3D reconstruction pipeline.
    
    Args:
        left_path: Path to left image
        right_path: Path to right image
        open_viewer: Whether to open the Open3D viewer at the end
    
    Returns:
        dict with all intermediate results
    """
    results = {}
    
    # ─────────────────────────────────────────
    # Stage 1: Load Images
    # ─────────────────────────────────────────
    print("[1/8] Loading images...")
    img_left, img_right = load_image_pair(left_path, right_path)
    results['img_left'] = img_left
    results['img_right'] = img_right
    print()
    
    # ─────────────────────────────────────────
    # Stage 2: Feature Detection
    # ─────────────────────────────────────────
    print("[2/8] Detecting features (SIFT)...")
    kp_left, desc_left, kp_right, desc_right = detect_features_pair(img_left, img_right)
    
    # Keypoints visualization
    kp_vis_left = draw_keypoints(img_left, kp_left, "Left")
    kp_vis_right = draw_keypoints(img_right, kp_right, "Right")
    results['keypoints_vis'] = np.hstack([
        cv2.resize(kp_vis_left, (0, 0), fx=0.5, fy=0.5),
        cv2.resize(kp_vis_right, (0, 0), fx=0.5, fy=0.5),
    ])
    print()
    
    # ─────────────────────────────────────────
    # Stage 3: Feature Matching
    # ─────────────────────────────────────────
    print("[3/8] Matching features (FLANN + Lowe's ratio test)...")
    good_matches = match_features(desc_left, desc_right)
    pts_left, pts_right = extract_matched_points(kp_left, kp_right, good_matches)
    
    # Save matches visualization
    matches_vis = draw_matches(img_left, kp_left, img_right, kp_right, good_matches)
    save_visualization(matches_vis, config.MATCHES_OUTPUT)
    results['matches_vis'] = matches_vis
    print()
    
    # ─────────────────────────────────────────
    # Stage 4: Epipolar Geometry
    # ─────────────────────────────────────────
    print("[4/8] Computing epipolar geometry...")
    F, mask = compute_fundamental_matrix(pts_left, pts_right)
    results['F'] = F
    results['mask'] = mask
    
    # Compute camera matrix and Essential Matrix
    K = config.get_default_camera_matrix(img_left.shape)
    E = compute_essential_matrix(F, K)
    
    # Filter inlier points
    mask_flat = mask.ravel().astype(bool)
    pts_left_inlier = pts_left[mask_flat]
    pts_right_inlier = pts_right[mask_flat]
    
    # Recover camera pose
    R, t, good_count = recover_camera_pose(E, pts_left_inlier, pts_right_inlier, K)
    results['R'] = R
    results['t'] = t
    
    # Save epipolar lines visualization
    epi_left, epi_right = draw_epipolar_lines(
        img_left, img_right, pts_left_inlier, pts_right_inlier, F
    )
    epipolar_vis = np.hstack([epi_left, epi_right])
    save_visualization(epipolar_vis, config.EPIPOLAR_OUTPUT)
    results['epipolar_vis'] = epipolar_vis
    print()
    
    # ─────────────────────────────────────────
    # Stage 5: Stereo Rectification
    # ─────────────────────────────────────────
    print("[5/8] Rectifying stereo pair...")
    # Always use calibrated rectification since we recovered R, t from the
    # Essential matrix. This produces a proper Q matrix for 3D reprojection.
    # (Even with estimated intrinsics, this is far more accurate than the
    #  uncalibrated path which can't produce a valid Q matrix.)
    rect_left, rect_right, Q = rectify_calibrated(
        img_left, img_right, K, None, R, t
    )
    
    results['rect_left'] = rect_left
    results['rect_right'] = rect_right
    results['Q'] = Q
    
    # Save rectified visualization
    rectified_vis = draw_rectified(rect_left, rect_right)
    save_visualization(rectified_vis, config.RECTIFIED_OUTPUT)
    results['rectified_vis'] = rectified_vis
    print()
    
    # ─────────────────────────────────────────
    # Stage 6: Disparity Map
    # ─────────────────────────────────────────
    print("[6/8] Computing disparity map (StereoSGBM + WLS)...")
    disp_raw, disp_filtered = compute_disparity(rect_left, rect_right)
    results['disparity'] = disp_filtered
    
    # Save disparity visualization
    disp_vis = colorize_disparity(disp_filtered)
    save_visualization(disp_vis, config.DISPARITY_MAP_OUTPUT)
    results['disparity_vis'] = disp_vis
    print()
    
    # ─────────────────────────────────────────
    # Stage 7: Depth Map
    # ─────────────────────────────────────────
    print("[7/8] Computing depth map...")
    depth = disparity_to_depth(disp_filtered, K)
    results['depth'] = depth
    
    depth_vis = colorize_depth(depth)
    save_visualization(depth_vis, config.DEPTH_MAP_OUTPUT)
    results['depth_vis'] = depth_vis
    print()
    
    # ─────────────────────────────────────────
    # Stage 8: 3D Point Cloud
    # ─────────────────────────────────────────
    print("[8/8] Generating 3D point cloud...")
    points_3d, colors_3d = reproject_to_3d(disp_filtered, Q, rect_left)
    
    # Filter outliers
    if len(points_3d) > 0:
        points_3d, colors_3d = filter_point_cloud(points_3d, colors_3d)
    
    # Save PLY
    save_ply(points_3d, colors_3d)
    results['points_3d'] = points_3d
    results['colors_3d'] = colors_3d
    print()
    
    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    print("=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Feature matches:  {len(good_matches)}")
    print(f"  Inlier matches:   {np.sum(mask)}")
    print(f"  3D points:        {len(points_3d)}")
    print(f"  Output directory:  {config.OUTPUT_DIR}")
    print()
    print("  Output files:")
    for f in [config.MATCHES_OUTPUT, config.EPIPOLAR_OUTPUT,
              config.RECTIFIED_OUTPUT, config.DISPARITY_MAP_OUTPUT,
              config.DEPTH_MAP_OUTPUT, config.POINT_CLOUD_OUTPUT]:
        exists = "✓" if os.path.exists(f) else "✗"
        print(f"    {exists} {os.path.basename(f)}")
    print()
    
    # Create summary
    create_pipeline_summary(
        img_left, img_right, matches_vis, epipolar_vis,
        rectified_vis, disp_vis, depth_vis
    )
    
    # Open viewer
    if open_viewer and len(points_3d) > 0:
        print("Opening 3D viewer...")
        view_point_cloud(points=points_3d, colors=colors_3d)
    
    return results


def main():
    print_banner()
    args = parse_args()
    
    open_viewer = not args.no_viewer
    
    try:
        results = run_pipeline(args.left, args.right, open_viewer=open_viewer)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please place your stereo images in the 'images/' directory:")
        print(f"  Left image:  {config.LEFT_IMAGE_PATH}")
        print(f"  Right image: {config.RIGHT_IMAGE_PATH}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
