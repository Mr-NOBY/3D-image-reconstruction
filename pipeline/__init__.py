"""
3D Reconstruction Pipeline
===========================
Modular pipeline for stereo 3D reconstruction from two images.

Modules:
    - feature_detection: SIFT keypoint detection & description
    - feature_matching: FLANN matching with Lowe's ratio test
    - epipolar_geometry: Fundamental/Essential matrix estimation
    - rectification: Stereo rectification (calibrated & uncalibrated)
    - disparity: StereoSGBM disparity map computation
    - depth_map: Disparity to depth conversion
    - point_cloud: 3D point cloud generation & export
    - visualization: Open3D viewer & intermediate result display
"""
