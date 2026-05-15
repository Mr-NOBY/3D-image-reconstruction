"""
3D Reconstruction -- Gradio UI
==============================
Interactive web interface for the stereo 3D reconstruction pipeline.

Wraps main.py's run_pipeline() to provide image upload, step-by-step
visualization, and Open3D 3D point cloud viewing.

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


def run_pipeline_ui(left_img_path, right_img_path, progress=gr.Progress()):
    """
    Run the full 3D reconstruction pipeline via main.py and return all
    intermediate visualizations plus the PLY path.
    """
    if left_img_path is None or right_img_path is None:
        raise gr.Error("Please upload both left and right images before running.")

    # Capture stdout from the pipeline for the log
    log_capture = io.StringIO()

    progress(0.05, desc="Running pipeline...")

    with contextlib.redirect_stdout(log_capture):
        results = run_pipeline(left_img_path, right_img_path, open_viewer=False)

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
            inputs=[left_input, right_input],
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
