# 3D Image Reconstruction Pipeline: In-Depth Technical Documentation

This document provides a rigorous, line-by-line, and mathematically detailed explanation of the 3D Image Reconstruction project. It explains not just *what* the code does, but the underlying computer vision theories, algorithms, and mathematical formulations at every step of the process.

## Architecture Overview

The system processes two 2D images (a stereo pair captured from slightly different horizontal perspectives) into a fully interactive 3D point cloud. The pipeline execution flow is orchestrated by `main.py` (`run_pipeline` function) and uses OpenCV and Open3D. It consists of 8 distinct sequential stages:

1. **Image Loading** (`load_image_pair`)
2. **Feature Detection** (`detect_features_pair`)
3. **Feature Matching** (`match_features`)
4. **Epipolar Geometry Computation** (`compute_fundamental_matrix`, `recover_camera_pose`)
5. **Stereo Rectification** (`rectify_calibrated` or `rectify_uncalibrated`)
6. **Disparity Map Generation** (`compute_disparity` or `compute_disparity_raft`)
7. **Depth Map Conversion** (`disparity_to_depth`)
8. **3D Point Cloud Reprojection & Filtering** (`reproject_to_3d`, `filter_point_cloud`)

---

## Detailed Implementation Breakdown

### Stage 1: Image Loading
**File:** `pipeline/feature_detection.py`

The pipeline starts by loading the left and right stereo images into memory.
```python
img_left = cv2.imread(left_path)
img_right = cv2.imread(right_path)
```
- **Implementation Detail:** Images are loaded in `BGR` format by default in OpenCV. The stereo pair must have the exact same spatial resolution ($W \times H$). If there is a slight misconfiguration in the camera setup, subsequent rectification steps will attempt to correct it, but the initial images must be of identical dimensions.

### Stage 2: Feature Detection (SIFT)
**File:** `pipeline/feature_detection.py` | **Algorithm:** Scale-Invariant Feature Transform

Before the images can be compared structurally, we need to find "points of interest" (keypoints) that are highly distinctive and invariant to scale, rotation, and illumination.

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(
    nfeatures=config.SIFT_N_FEATURES,
    nOctaveLayers=config.SIFT_N_OCTAVE_LAYERS,
    contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
    edgeThreshold=config.SIFT_EDGE_THRESHOLD,
    sigma=config.SIFT_SIGMA
)
keypoints, descriptors = sift.detectAndCompute(gray, None)
```

**Line-by-Line Intuition:**
1. **Grayscale Conversion:** SIFT analyzes intensity gradients, so color information is discarded to focus purely on structural luminance.
2. **Difference of Gaussians (DoG):** SIFT creates a "scale-space pyramid". The image is repeatedly blurred (using Gaussian filters defined by `sigma`) and scaled down. The difference between adjacent blurred images (DoG) is computed to identify edges and corners at various scales.
3. **Local Extrema Detection:** A pixel is marked as a keypoint if it is a local maximum or minimum compared to its 26 neighbors (8 in the current scale, 9 in the scale above, 9 in the scale below).
4. **Thresholding (`contrastThreshold`, `edgeThreshold`):** Weak keypoints (low contrast) and points located on strong linear edges (which are ambiguous to match) are discarded.
5. **Orientation Assignment:** The dominant gradient direction around the keypoint is calculated. This allows the descriptor to be rotationally invariant.
6. **Descriptors:** For each keypoint, a $16 \times 16$ neighborhood is analyzed. It is divided into $4 \times 4$ sub-blocks. For each block, an 8-bin orientation histogram is created, resulting in a $4 \times 4 \times 8 = 128$-dimensional floating-point vector. This is the **descriptor**.

### Stage 3: Feature Matching (FLANN & Lowe's Ratio Test)
**File:** `pipeline/feature_matching.py` | **Algorithm:** Approximate Nearest Neighbors

Once descriptors are extracted, we must find matching pairs between the left and right images.

```python
index_params = dict(algorithm=config.FLANN_INDEX_KDTREE, trees=config.FLANN_TREES)
search_params = dict(checks=config.FLANN_CHECKS)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_left, desc_right, k=2)
```

**Line-by-Line Intuition:**
1. **FLANN:** A brute-force matching of $N$ left descriptors to $M$ right descriptors is $O(N \times M)$ which is too slow. FLANN uses a KD-Tree (K-Dimensional Tree) to partition the 128-dimensional descriptor space, allowing for $O(N \log M)$ lookup times.
2. **KNN Match (`k=2`):** For every descriptor in the left image, we ask FLANN for the **top 2** closest matches in the right image.

```python
good_matches = []
for m, n in matches:
    if m.distance < config.LOWE_RATIO_THRESHOLD * n.distance:
        good_matches.append(m)
```
**Lowe's Ratio Test (`LOWE_RATIO_THRESHOLD = 0.75`):**
- Let $D(best)$ be `m.distance` and $D(second\_best)$ be `n.distance`.
- If an object is unique, $D(best)$ will be very small, and $D(second\_best)$ will be large.
- If an object is repetitive (e.g., a brick wall), $D(best)$ and $D(second\_best)$ will be almost identical.
- We only keep the match if the best match is significantly closer (less than 75% the distance) than the second-best match. This eliminates ambiguous mappings.

### Stage 4: Epipolar Geometry & Camera Pose
**File:** `pipeline/epipolar_geometry.py`

This is the mathematical core of stereo vision. We use the matched points to understand the physical relationship between the two camera positions.

```python
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 
                                 ransacReprojThreshold=1.0, confidence=0.999)
```
**The Fundamental Matrix ($F$):**
- The epipolar constraint states that for a point $x$ in the left image, its corresponding point $x'$ in the right image MUST lie on a specific line (the epipolar line). Mathematically: $x'^T F x = 0$.
- $F$ is a $3 \times 3$ matrix of rank 2. It maps a 2D point in one image to a 1D line in the other.
- **RANSAC (Random Sample Consensus):** Matches contain outliers. RANSAC randomly picks 8 point-pairs, computes a candidate $F$, and counts how many other points satisfy $x'^T F x \approx 0$ (within `ransacReprojThreshold`). It repeats this to find the mathematically most supported matrix. The `mask` array flags the inliers.

```python
E = K.T @ F @ K
U, S, Vt = np.linalg.svd(E)
S = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
E = U @ np.diag(S) @ Vt
```
**The Essential Matrix ($E$):**
- While $F$ operates in pixel coordinates, $E$ operates in normalized camera coordinates.
- We compute it using the intrinsic camera matrix $K$.
- We enforce the mathematical property that $E$ must have exactly two equal non-zero singular values by applying Singular Value Decomposition (SVD), averaging the top two values, and zeroing the third.

```python
good_count, R, t, pose_mask = cv2.recoverPose(E, pts_left, pts_right, K)
```
**Pose Recovery:**
- $E$ mathematically decomposes into a Rotation matrix ($R$) and a Translation vector ($t$) such that $E = [t]_\times R$.
- There are 4 possible mathematical combinations of $R$ and $t$. OpenCV triangulates points for all 4 solutions and applies the **cheirality check**: selecting the one solution where the triangulated 3D points fall *in front* of both camera lenses ($Z > 0$).

### Stage 5: Stereo Rectification
**File:** `pipeline/rectification.py`

Searching for matching pixels across entire 2D images is inefficient. Rectification physically warps the images so that the cameras appear perfectly parallel. Thus, a pixel at $(x, y)$ in the left image is guaranteed to match a pixel at $(x - d, y)$ in the right image. The search becomes 1-Dimensional.

**Calibrated Path:**
```python
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist_coeffs, K, dist_coeffs, (w, h), R, t)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
rect_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```
- Computes rectification transforms `R1` (left) and `R2` (right).
- Computes $P1, P2$ (new projection matrices).
- Yields the critical $Q$ matrix (Reprojection matrix) that maps $(x, y, disparity)$ directly to 3D $(X,Y,Z)$.

**Uncalibrated Path:**
```python
retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l, pts_r, F, imgSize=(w, h))
rect_left = cv2.warpPerspective(img_left, H1, (w, h))
```
- When $K$ is unknown or estimated, we use $F$ directly to find 2D homographies ($H1, H2$) that warp the images to force epipolar lines to be horizontal.
- The $Q$ matrix has to be approximated manually based on the image center and estimated focal length.

### Stage 6: Disparity Map Calculation
**Files:** `pipeline/disparity.py` and `pipeline/raft_stereo_wrapper.py`

We now find the disparity $d = x_{left} - x_{right}$ for every single pixel. The pipeline supports two methods: a classical approach and a deep-learning approach.

#### Method A: StereoSGBM (Classical)
**Algorithm:** Semi-Global Block Matching + WLS Filter

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_left = clahe.apply(gray_left)
```
- **CLAHE:** Applied to enhance local contrast, bringing out details in flat textures so SGBM can track them.

```python
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,          # Search range
    blockSize=block_size,             # Window size
    P1=8 * 3 * block_size ** 2,       # Smoothness penalty
    P2=32 * 3 * block_size ** 2,      # Discontinuity penalty
    uniquenessRatio=15,
    speckleWindowSize=200,
    mode=cv2.STEREO_SGBM_MODE_HH
)
```
**Semi-Global Block Matching (SGBM):**
- Minimizes a global cost function using dynamic programming along multiple 1D paths (horizontal, vertical, diagonal).
- `P1` penalizes small disparity changes (sloped surfaces).
- `P2` strongly penalizes large disparity changes (depth edges/occlusions).
- `speckleWindowSize`: Discards tiny blobs of disparities as noise.

```python
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)
disparity_filtered = wls_filter.filter(disp_left, gray_left, None, disp_right)
```
**WLS (Weighted Least Squares) Filtering:**
- StereoSGBM often leaves "holes" or noisy edges.
- We compute right-to-left disparity as well (`disp_right`), and the WLS filter mathematically cross-checks the left and right maps. It smooths the depth map while strictly preserving the sharp edges detected in the original color image (`gray_left`).

#### Method B: RAFT-Stereo (Deep Learning)
**Algorithm:** Recurrent All-Pairs Field Transforms for Stereo Matching

```python
model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
model.load_state_dict(torch.load(model_path))
...
_, flow_up = model(image1, image2, iters=iters, test_mode=True)
disparity = -flow_up.cpu().numpy()
```
**Line-by-Line Intuition:**
1. **Feature Extraction:** Instead of using hand-crafted features or raw pixel intensities, the images pass through a shared Convolutional Neural Network (CNN) backbone that extracts multi-scale features.
2. **Correlation Volume:** A 1D correlation volume is computed between the left and right feature maps. Since the images are rectified, the network only needs to search along the horizontal epipolar line.
3. **Iterative GRU Updates (`iters=config.RAFT_STEREO_VALID_ITERS`):** A Convolutional Gated Recurrent Unit (ConvGRU) starts with an initial disparity of zero and iteratively looks up the correlation volume to refine the disparity field.
4. **Convex Upsampling:** The GRU operates at a lower resolution (1/8th scale) for speed. A learned convex combination mask is used to smoothly upsample the disparity back to the original full image resolution.

- **VRAM Management:** `raft_stereo_wrapper.py` dynamically resizes inputs (`max_h`, `max_w`) before passing them to the model to prevent Out-Of-Memory errors on standard GPUs, then scales the disparity values proportionally back up.
- **Why RAFT-Stereo?** Unlike StereoSGBM which struggles with textureless regions and occlusions (yielding ~73% pixel coverage), RAFT-Stereo infers dense matching globally, resulting in 100% pixel coverage and much higher accuracy on thin structures.

### Stage 7: Depth Map Conversion
**File:** `pipeline/depth_map.py`

```python
depth_map = (focal_length * baseline) / disparity
```
**The Math of Depth:**
- Given focal length $f$, baseline $B$ (distance between cameras), and disparity $d$.
- Due to similar triangles, depth $Z = \frac{f \cdot B}{d}$.
- Notice the inverse relationship: a larger pixel shift (high disparity) implies the object is very close. If disparity is 0, the object is infinitely far away.

### Stage 8: 3D Point Cloud Generation
**File:** `pipeline/point_cloud.py`

We transform our 2.5D disparity map into a true 3D spatial geometry.

```python
points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
```
**Reprojection Matrix ($Q$):**
The multiplication occurs as follows:
$$ \begin{bmatrix} X \\ Y \\ Z \\ W \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & -c_x \\ 0 & 1 & 0 & -c_y \\ 0 & 0 & 0 & f \\ 0 & 0 & \frac{-1}{B} & \frac{c_x - c_x'}{B} \end{bmatrix} \begin{bmatrix} x \\ y \\ d \\ 1 \end{bmatrix} $$
The final 3D coordinates are $(X/W, Y/W, Z/W)$.

```python
# Geometry Normalization
centroid = np.median(valid_points, axis=0)
valid_points = valid_points - centroid
for axis in range(3):
    iqr = np.percentile(valid_points[:, axis], 90) - np.percentile(valid_points[:, axis], 10)
    valid_points[:, axis] /= iqr
```
**Geometry Normalization (Fixing the Billboard Effect):**
When using uncalibrated images, the approximated $Q$ matrix can scale the Z axis disproportionately compared to X and Y. This makes the 3D map look like a flat painting. By dividing each spatial axis by its Interquartile Range (IQR), we force X, Y, and Z to share similar dynamic scales, resulting in a realistically proportional 3D volume.

```python
pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
```
**Statistical Outlier Removal (Open3D):**
For every point, it computes the average distance to its 50 nearest neighbors. The mean and standard deviation of all these distances are calculated across the entire cloud. Any point whose average neighbor distance falls outside `mean + 1.0 * std` is considered an outlier (floating noise) and deleted.

```python
_save_ply_manual(...)
```
**PLY File Export:**
The `.ply` (Polygon File Format) header is written via ASCII describing the vertex list: $X, Y, Z$ (floats) and $R, G, B$ (unsigned characters). The viewer (Open3D) uses this to render the colored mesh.

---

## The App Layer: Gradio UI & Config State

- **`config.py`**: Acts as a stateful memory bank. Variables like `SGBM_BLOCK_SIZE` and `WLS_LAMBDA` dictate the aggressiveness of the algorithms.
- **`app.py`**: Uses Gradio blocks. When the "Run Pipeline" button is clicked, `run_pipeline_ui` intercepts the web sliders, dynamically overwrites the global variables in `config.py` (e.g., `config.WLS_SIGMA = float(wls_sigma)`), and triggers `main.py`. This provides real-time algorithmic tuning without requiring code changes or CLI restarts. The `io.StringIO` context manager hijacks Python's `sys.stdout` to pipe the terminal logs directly into the web UI's text box.
