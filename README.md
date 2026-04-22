# 3D Reconstruction from Two Images

> Reconstruct a 3D point cloud and depth map from two images of the same scene taken from different viewpoints using stereo vision and epipolar geometry.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your images
#    images/left.jpg  — first viewpoint
#    images/right.jpg — second viewpoint

# 3. Run the pipeline
python main.py

# Run with custom images
python main.py --left path/to/img1.jpg --right path/to/img2.jpg

# Skip the 3D viewer (just save outputs)
python main.py --no-viewer
```

### Output Files

| File | Description |
|------|-------------|
| `output/matches.jpg` | Feature matches between the two images |
| `output/epipolar_lines.jpg` | Epipolar lines visualization |
| `output/rectified.jpg` | Rectified stereo pair with alignment lines |
| `output/disparity_map.png` | Colorized disparity map |
| `output/depth_map.png` | Colorized depth map |
| `output/point_cloud.ply` | 3D point cloud (viewable in MeshLab, Open3D, or the web viewer) |
| `output/pipeline_summary.png` | Combined summary of all stages |

### Web Viewer

Open `web_viewer/index.html` in a browser and load the generated PLY file to interactively explore the 3D reconstruction.

---

## Pipeline Overview

```
Input Images → Feature Detection → Feature Matching → Epipolar Geometry
    → Stereo Rectification → Disparity Map → Depth Map → 3D Point Cloud
```

---

## Mathematical Foundation

### 1. Feature Detection (SIFT)

**Scale-Invariant Feature Transform** detects keypoints that are invariant to scale and rotation.

**Scale-space construction** — the image is convolved with Gaussian kernels at increasing scales:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

where $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/(2\sigma^2)}$

**Difference of Gaussians (DoG)** — keypoints are detected as extrema in:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

Each keypoint gets a **128-dimensional descriptor** computed from gradient orientations in its local neighborhood.

---

### 2. Feature Matching

Descriptors are matched using **k-nearest neighbors** (k=2) with the FLANN library.

**Lowe's Ratio Test** filters ambiguous matches:

$$\frac{d(f_1, f_2^{(1)})}{d(f_1, f_2^{(2)})} < \tau$$

where $f_2^{(1)}$ and $f_2^{(2)}$ are the best and second-best matches, and $\tau = 0.75$.

A low ratio means the best match is significantly better than the second best — indicating a distinctive, reliable correspondence.

---

### 3. Epipolar Geometry

#### The Epipolar Constraint

Given two cameras observing the same 3D point $\mathbf{X}$, the projections $\mathbf{x}$ and $\mathbf{x}'$ in the two images satisfy:

$$\mathbf{x}'^T F \mathbf{x} = 0$$

where $F$ is the **Fundamental Matrix** (3×3, rank 2, 7 degrees of freedom).

#### Geometric Interpretation

- **Epipole**: The projection of one camera's center into the other camera's image
- **Epipolar line**: For a point $\mathbf{x}$ in image 1, the line $\mathbf{l}' = F\mathbf{x}$ in image 2 contains the corresponding point $\mathbf{x}'$
- **Epipolar plane**: The plane defined by the 3D point and both camera centers

#### Fundamental Matrix $F$

$F$ is estimated from point correspondences using the **8-point algorithm** with RANSAC:

1. Select 8 random correspondences
2. Solve the linear system $\mathbf{x}'^T F \mathbf{x} = 0$ for F
3. Enforce rank-2 constraint via SVD: set smallest singular value to 0
4. Count inliers and repeat

#### Essential Matrix $E$

When camera intrinsics $K$ are known:

$$E = K'^T F K$$

$E$ encodes only the relative rotation and translation between cameras:

$$E = [t]_\times R$$

where $[t]_\times$ is the skew-symmetric matrix of the translation vector.

#### Camera Pose Recovery

$E$ is decomposed via SVD into $R$ (rotation) and $t$ (translation direction):

$$E = U \Sigma V^T$$

This yields 4 possible solutions (2 rotations × 2 translation directions). The correct one is selected by the **cheirality check** — triangulated points must have positive depth in both cameras.

---

### 4. Stereo Rectification

Rectification transforms both images so that:
- Epipolar lines become **horizontal** (parallel to the image x-axis)
- Corresponding points lie on the **same row** (same y-coordinate)

This reduces the 2D correspondence search to a 1D horizontal search, enabling efficient disparity computation.

**Uncalibrated rectification** computes homographies $H_1, H_2$ such that:

$$H_1^{-T} F H_2^{-1}$$

has the form of a rectified fundamental matrix (epipoles at infinity).

---

### 5. Disparity Map

**Disparity** is the horizontal pixel difference between corresponding points in the rectified pair:

$$d = x_L - x_R$$

**StereoSGBM** (Semi-Global Block Matching) minimizes an energy function:

$$E(D) = \sum_p \left( C(p, D_p) + \sum_{q \in N_p} P_1 \cdot T[|D_p - D_q| = 1] + \sum_{q \in N_p} P_2 \cdot T[|D_p - D_q| > 1] \right)$$

where:
- $C(p, D_p)$ — matching cost at pixel $p$ for disparity $D_p$
- $P_1$ — penalty for small disparity changes (smooth surfaces)
- $P_2$ — penalty for large disparity changes (depth edges)

The **WLS filter** refines the disparity map by combining left and right disparities with an edge-aware weighted least squares filter.

---

### 6. Depth Map

Depth is computed from disparity:

$$Z = \frac{f \cdot B}{d}$$

where:
- $Z$ — depth (distance from camera)
- $f$ — focal length (pixels)
- $B$ — baseline (distance between cameras)
- $d$ — disparity (pixels)

Key insight: **depth is inversely proportional to disparity** — closer objects have larger disparity.

---

### 7. 3D Point Cloud

The **Q matrix** maps image coordinates and disparity to 3D world coordinates:

$$\begin{bmatrix} X \\ Y \\ Z \\ W \end{bmatrix} = Q \cdot \begin{bmatrix} x \\ y \\ d \\ 1 \end{bmatrix}$$

The 3D point is then $(X/W, Y/W, Z/W)$.

Each point is colored using the corresponding pixel from the original image, producing a colored point cloud that can be exported as a PLY file and viewed in 3D.

---

## Project Structure

```
CV_Project/
├── main.py                      # Main pipeline entry point
├── config.py                    # All tunable parameters
├── requirements.txt             # Python dependencies
├── pipeline/
│   ├── feature_detection.py     # SIFT keypoint detection
│   ├── feature_matching.py      # FLANN matching + ratio test
│   ├── epipolar_geometry.py     # F/E matrix, camera pose
│   ├── rectification.py         # Stereo rectification
│   ├── disparity.py             # StereoSGBM disparity
│   ├── depth_map.py             # Depth computation
│   ├── point_cloud.py           # 3D reprojection + PLY export
│   └── visualization.py         # Open3D viewer + plots
├── web_viewer/
│   ├── index.html               # Three.js 3D viewer
│   ├── style.css                # Viewer styling
│   └── viewer.js                # Viewer logic
├── images/                      # Input stereo pair
│   ├── left.jpg
│   └── right.jpg
└── output/                      # Generated outputs
```

## Tips for Best Results

1. **Image quality**: Use sharp, well-lit images with plenty of texture
2. **Overlap**: Ensure significant overlap (>60%) between views
3. **Baseline**: The distance between viewpoints should be 5-15% of the distance to the scene
4. **Avoid**: Pure rotation (translation is needed for depth), textureless surfaces, repetitive patterns
5. **Tuning**: Adjust `SGBM_NUM_DISPARITIES` and `SGBM_BLOCK_SIZE` in `config.py` for your images

## Dependencies

- **OpenCV** (with contrib) — feature detection, stereo matching, geometry
- **NumPy** — matrix operations
- **Matplotlib** — 2D visualization
- **Open3D** — 3D point cloud visualization and export
- **SciPy** — optional advanced filtering
