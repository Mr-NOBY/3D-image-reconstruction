"""
3D Reconstruction -- Gradio UI
==============================
Interactive web interface for the stereo 3D reconstruction pipeline.

Wraps main.py's run_pipeline() to provide image upload, step-by-step
visualization, parameter tuning, and Open3D 3D point cloud viewing.

Usage:
    python app.py
"""

import os
import sys
import io
import contextlib
import threading

import cv2
import numpy as np
import gradio as gr

# Open3D requires X11 session type on Wayland systems.
# Must be set in the shell BEFORE launching: export XDG_SESSION_TYPE=x11
if os.environ.get("XDG_SESSION_TYPE") != "x11":
    print("[WARNING] Open3D 3D viewer requires X11. Run this first:")
    print("          export XDG_SESSION_TYPE=x11")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main import run_pipeline
from pipeline.visualization import view_point_cloud


def bgr_to_rgb(img):
    """Convert OpenCV BGR image to RGB for Gradio display."""
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_pipeline_ui(
    left_img_path, right_img_path,
    rectification_mode,
    block_size, num_disparities, uniqueness_ratio,
    speckle_window, wls_lambda, wls_sigma,
    outlier_neighbors, outlier_std,
    progress=gr.Progress(),
):
    """
    Run the full 3D reconstruction pipeline with user-tuned parameters.
    Overrides config values before each run.
    """
    if left_img_path is None or right_img_path is None:
        raise gr.Error("Please upload both left and right images before running.")

    # Override config with slider values
    config.SGBM_BLOCK_SIZE = int(block_size)
    config.SGBM_P1 = 8 * 3 * int(block_size) ** 2
    config.SGBM_P2 = 32 * 3 * int(block_size) ** 2
    config.SGBM_NUM_DISPARITIES = int(num_disparities)
    config.SGBM_UNIQUENESS_RATIO = int(uniqueness_ratio)
    config.SGBM_SPECKLE_WINDOW_SIZE = int(speckle_window)
    config.WLS_LAMBDA = float(wls_lambda)
    config.WLS_SIGMA = float(wls_sigma)
    config.PC_STATISTICAL_OUTLIER_NB = int(outlier_neighbors)
    config.PC_STATISTICAL_OUTLIER_STD = float(outlier_std)

    # Capture stdout from the pipeline for the log
    log_capture = io.StringIO()

    progress(0.05, desc="Running pipeline...")

    use_calibrated = (rectification_mode == "Calibrated")

    with contextlib.redirect_stdout(log_capture):
        results = run_pipeline(
            left_img_path, right_img_path,
            open_viewer=False,
            use_calibrated=use_calibrated,
        )

    progress(0.95, desc="Preparing outputs...")

    # Extract visualizations from results dict (all BGR -- convert to RGB)
    keypoints_img = bgr_to_rgb(results.get('keypoints_vis'))
    matches_img = bgr_to_rgb(results.get('matches_vis'))
    epipolar_img = bgr_to_rgb(results.get('epipolar_vis'))
    rectified_img = bgr_to_rgb(results.get('rectified_vis'))
    disparity_img = bgr_to_rgb(results.get('disparity_vis'))
    depth_img = bgr_to_rgb(results.get('depth_vis'))

    # Point cloud info
    n_points = len(results.get('points_3d', []))
    ply_path = config.POINT_CLOUD_OUTPUT
    status = f"Point cloud ready -- {n_points} points\nSaved to: {ply_path}"

    progress(1.0, desc="Done!")

    return (
        keypoints_img,
        matches_img,
        epipolar_img,
        rectified_img,
        disparity_img,
        depth_img,
        status,
        log_capture.getvalue(),
    )


def launch_open3d_viewer():
    """
    Launch Open3D interactive viewer for the generated point cloud.
    Runs in a separate thread so it doesn't block the Gradio server.
    """
    ply_path = config.POINT_CLOUD_OUTPUT
    if not os.path.isfile(ply_path):
        raise gr.Error("No point cloud found. Run the pipeline first!")

    def _open():
        view_point_cloud(ply_path=ply_path)

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    return "Open3D viewer launched -- check your desktop!"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* -- Global ------------------------------------------------ */
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* -- Header ------------------------------------------------ */
#app-title {
    text-align: center;
    padding: 1.2rem 0 0.4rem;
}
#app-title h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
#app-subtitle {
    text-align: center;
    opacity: 0.65;
    font-size: 0.95rem;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}

/* -- Upload cards ------------------------------------------ */
.upload-section {
    border: 2px dashed rgba(102, 126, 234, 0.35);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.25s ease;
}
.upload-section:hover {
    border-color: rgba(102, 126, 234, 0.7);
}

/* -- Run button -------------------------------------------- */
#run-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#run-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* -- Step cards --------------------------------------------- */
.step-card {
    border-radius: 14px;
    overflow: hidden;
}
.step-card img {
    border-radius: 10px;
}

/* -- Log area ---------------------------------------------- */
#log-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    background: #1a1b26 !important;
    color: #a9b1d6 !important;
    border-radius: 12px !important;
}

/* -- 3D viewer button -------------------------------------- */
#open3d-btn {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#open3d-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
}
#viewer-status textarea {
    text-align: center !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

/* -- Parameter panel --------------------------------------- */
.param-panel {
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 14px;
    padding: 0.8rem;
    background: rgba(102, 126, 234, 0.03);
}
"""

CUSTOM_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

# Pre-load sample images if they exist
SAMPLE_LEFT = config.LEFT_IMAGE_PATH if os.path.isfile(config.LEFT_IMAGE_PATH) else None
SAMPLE_RIGHT = config.RIGHT_IMAGE_PATH if os.path.isfile(config.RIGHT_IMAGE_PATH) else None


def build_ui():
    with gr.Blocks(
        title="3D Reconstruction Pipeline",
    ) as demo:

        # -- Header ------------------------------------------------
        gr.HTML(
            '<div id="app-title"><h1>3D Reconstruction Pipeline</h1></div>'
            '<div id="app-subtitle">Upload a stereo image pair  |  View each processing step  |  Explore the 3D result</div>'
        )

        # -- Upload Section ----------------------------------------
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="upload-section"):
                left_input = gr.Image(
                    label="Left Image",
                    type="filepath",
                    value=SAMPLE_LEFT,
                    height=280,
                    sources=["upload", "clipboard"],
                )
            with gr.Column(elem_classes="upload-section"):
                right_input = gr.Image(
                    label="Right Image",
                    type="filepath",
                    value=SAMPLE_RIGHT,
                    height=280,
                    sources=["upload", "clipboard"],
                )

        # -- Parameter Tuning Panel --------------------------------
        with gr.Accordion("Parameter Tuning", open=False):
            gr.HTML(
                '<p style="opacity:0.6; font-size:0.85rem; margin:0 0 0.5rem;">'
                'Adjust these to improve reconstruction quality for different images. '
                'Hover over the <b>?</b> icon on each slider for details.</p>'
            )

            with gr.Row():
                with gr.Column(elem_classes="param-panel"):
                    gr.HTML('<p style="font-weight:600; margin:0 0 0.3rem;">Rectification</p>')
                    rd_rectify_mode = gr.Radio(
                        choices=["Calibrated", "Uncalibrated"],
                        value="Calibrated",
                        label="Rectification Mode",
                        info="Calibrated: uses estimated camera intrinsics (K) + recovered pose (R, t). "
                             "Uncalibrated: uses only the Fundamental matrix + matched points -- "
                             "try this if the 3D looks flat or distorted.",
                    )

                    gr.HTML('<p style="font-weight:600; margin:0.8rem 0 0.3rem;">Stereo Matching (SGBM)</p>')
                    sl_block_size = gr.Slider(
                        minimum=3, maximum=21, step=2,
                        value=config.SGBM_BLOCK_SIZE,
                        label="Block Size",
                        info="Window size for pixel matching. Larger = smoother but less detail. Use 5-7 for textured scenes, 11-15 for smooth objects.",
                    )
                    sl_num_disp = gr.Slider(
                        minimum=16, maximum=256, step=16,
                        value=config.SGBM_NUM_DISPARITIES,
                        label="Num Disparities",
                        info="Max disparity search range. Increase for wider baselines or closer objects. Must be divisible by 16.",
                    )
                    sl_uniqueness = gr.Slider(
                        minimum=5, maximum=30, step=1,
                        value=config.SGBM_UNIQUENESS_RATIO,
                        label="Uniqueness Ratio",
                        info="How much better the best match must be vs second-best (%). Higher = fewer but more reliable matches.",
                    )
                    sl_speckle = gr.Slider(
                        minimum=50, maximum=500, step=25,
                        value=config.SGBM_SPECKLE_WINDOW_SIZE,
                        label="Speckle Window",
                        info="Max size of isolated noise patches to remove. Larger = more aggressive noise cleanup.",
                    )

                with gr.Column(elem_classes="param-panel"):
                    gr.HTML('<p style="font-weight:600; margin:0 0 0.3rem;">WLS Filter (Smoothing)</p>')
                    sl_wls_lambda = gr.Slider(
                        minimum=1000, maximum=20000, step=500,
                        value=config.WLS_LAMBDA,
                        label="WLS Lambda",
                        info="Smoothing strength. Higher = smoother disparity map. Lower = preserves more detail but keeps more noise.",
                    )
                    sl_wls_sigma = gr.Slider(
                        minimum=0.5, maximum=3.0, step=0.1,
                        value=config.WLS_SIGMA,
                        label="WLS Sigma",
                        info="Edge sensitivity. Lower = stronger edge preservation. Higher = smoother across edges.",
                    )

                    gr.HTML('<p style="font-weight:600; margin:0.8rem 0 0.3rem;">Point Cloud Filtering</p>')
                    sl_outlier_nb = gr.Slider(
                        minimum=10, maximum=100, step=5,
                        value=config.PC_STATISTICAL_OUTLIER_NB,
                        label="Outlier Neighbors",
                        info="How many neighbors to check for outlier detection. More neighbors = more robust but slower.",
                    )
                    sl_outlier_std = gr.Slider(
                        minimum=0.3, maximum=3.0, step=0.1,
                        value=config.PC_STATISTICAL_OUTLIER_STD,
                        label="Outlier Std Ratio",
                        info="Strictness of outlier removal. Lower = more aggressive (removes more floating points).",
                    )

        # -- Run Button --------------------------------------------
        with gr.Row():
            run_btn = gr.Button(
                "Run Reconstruction Pipeline",
                variant="primary",
                elem_id="run-btn",
                size="lg",
            )

        # -- Pipeline Steps Output ---------------------------------
        gr.HTML('<hr style="border: none; border-top: 1px solid rgba(127,127,127,0.2); margin: 1.2rem 0;">')

        with gr.Accordion("Pipeline Steps", open=True):
            with gr.Row():
                with gr.Column(elem_classes="step-card"):
                    out_keypoints = gr.Image(
                        label="1 - SIFT Keypoints",
                        interactive=False,
                    )
                with gr.Column(elem_classes="step-card"):
                    out_matches = gr.Image(
                        label="2 - Feature Matches",
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(elem_classes="step-card"):
                    out_epipolar = gr.Image(
                        label="3 - Epipolar Lines",
                        interactive=False,
                    )
                with gr.Column(elem_classes="step-card"):
                    out_rectified = gr.Image(
                        label="4 - Rectified Pair",
                        interactive=False,
                    )

            with gr.Row():
                with gr.Column(elem_classes="step-card"):
                    out_disparity = gr.Image(
                        label="5 - Disparity Map",
                        interactive=False,
                    )
                with gr.Column(elem_classes="step-card"):
                    out_depth = gr.Image(
                        label="6 - Depth Map",
                        interactive=False,
                    )

        # -- 3D Viewer ---------------------------------------------
        gr.HTML('<hr style="border: none; border-top: 1px solid rgba(127,127,127,0.2); margin: 1.2rem 0;">')

        with gr.Accordion("3D Point Cloud", open=True):
            out_3d_status = gr.Textbox(
                label="Point Cloud Status",
                interactive=False,
                elem_id="viewer-status",
                lines=2,
            )
            open3d_btn = gr.Button(
                "Open 3D Viewer (Open3D)",
                elem_id="open3d-btn",
                size="lg",
            )
            open3d_msg = gr.Textbox(
                label="Viewer Status",
                interactive=False,
                visible=False,
            )

        # -- Logs --------------------------------------------------
        with gr.Accordion("Pipeline Log", open=False):
            out_log = gr.Textbox(
                label="Console Output",
                lines=18,
                max_lines=40,
                interactive=False,
                elem_id="log-box",
            )

        # -- Wiring ------------------------------------------------
        run_btn.click(
            fn=run_pipeline_ui,
            inputs=[
                left_input, right_input,
                rd_rectify_mode,
                sl_block_size, sl_num_disp, sl_uniqueness,
                sl_speckle, sl_wls_lambda, sl_wls_sigma,
                sl_outlier_nb, sl_outlier_std,
            ],
            outputs=[
                out_keypoints,
                out_matches,
                out_epipolar,
                out_rectified,
                out_disparity,
                out_depth,
                out_3d_status,
                out_log,
            ],
            show_progress="full",
        )

        open3d_btn.click(
            fn=launch_open3d_viewer,
            inputs=[],
            outputs=[open3d_msg],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
