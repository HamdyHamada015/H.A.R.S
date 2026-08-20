import os

_GRADIO_TEMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_temp")
os.makedirs(_GRADIO_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TEMP

import cv2
import math
import time
import shutil
import uuid
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

from distance_estimation import (
    KNOWN_WIDTHS,
    approximate_focal_length,
)

from risk_engine import (
    draw_object_result,
    draw_path_zone,
    process_object_from_bbox,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_DIR = BASE_DIR / "inputs"

OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("AI COLLISION RISK SYSTEM")
print("=" * 50)
print(f"Model: {MODEL_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Gradio temp directory: {_GRADIO_TEMP}")
print("=" * 50)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model weights not found at: {MODEL_PATH}\n"
        f"Place your trained 'best.pt' file next to this script."
    )

try:
    model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded successfully.")
except Exception as e:
    print("ERROR: Could not load YOLO model.")
    print(e)
    raise

# ============================================================
# GLOBAL STATE
# ============================================================

previous_states = {}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_filename(extension=".mp4"):
    """Generate a unique filename."""
    return f"collision_{uuid.uuid4().hex}{extension}"


def copy_input_video(video_path):
    if video_path is None:
        return None

    if isinstance(video_path, dict):
        possible_paths = [video_path.get("path"), video_path.get("name")]
        source = next((p for p in possible_paths if p and os.path.exists(str(p))), None)

        if source is None:
            raise FileNotFoundError("Could not locate uploaded video.")
    else:
        source = str(video_path)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Input video does not exist:\n{source}")

    extension = Path(source).suffix.lower() or ".mp4"
    destination = INPUT_DIR / safe_filename(extension)

    last_error = None
    for attempt in range(5):
        try:
            shutil.copy2(source, destination)
            last_error = None
            break
        except PermissionError as e:
            last_error = e
            time.sleep(0.4)

    if last_error is not None:
        raise last_error

    return str(destination)


def make_output_path():
    return str(OUTPUT_DIR / safe_filename(".mp4"))


def draw_text_with_outline(img, text, pos, font_scale, color, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 5, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_header(frame, status, speed):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 145), (15, 18, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    draw_text_with_outline(frame, "AI COLLISION RISK SYSTEM", (30, 40), 0.95, (255, 255, 255), 2)
    draw_text_with_outline(frame, f"SPEED  {int(speed)} km/h", (30, 88), 0.85, (255, 255, 255), 2)

    status_colors = {
        "SAFE": (0, 220, 100),
        "WARNING": (0, 220, 255),
        "HIGH RISK": (0, 120, 255),
        "CRITICAL / BRAKE": (0, 0, 255),
    }
    color = status_colors.get(status, (255, 255, 255))
    draw_text_with_outline(frame, f"STATUS  {status}", (w - 420, 88), 0.85, color, 2)


def draw_bottom_bar(frame, objects_count):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 48), (w, h), (10, 12, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    draw_text_with_outline(frame, f"TRACKED OBJECTS: {objects_count}", (25, h - 15), 0.55, (220, 220, 220), 1)
    draw_text_with_outline(frame, "MONOCULAR DISTANCE ESTIMATION", (w - 330, h - 15), 0.50, (180, 180, 180), 1)

# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video_pipeline(video_path, start_speed, crop_dashboard=False, progress=gr.Progress()):
    
    global previous_states
    previous_states = {}

    if video_path is None:
        raise gr.Error("Please upload or record a video first.")

    cap = None
    out = None

    try:
        progress(0.02, desc="Preparing video...")
        local_input = copy_input_video(video_path)

        if local_input is None:
            raise gr.Error("Unable to read the selected video.")

        cap = cv2.VideoCapture(local_input)

        if not cap.isOpened():
            raise gr.Error("OpenCV could not open the video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0 or math.isnan(fps):
            fps = 30.0
        fps = min(max(fps, 10), 60)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise gr.Error("Could not read video dimensions. The file may be corrupted.")

        output_path = make_output_path()
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise gr.Error("Could not create output video writer.")

        focal_length = approximate_focal_length(image_width_px=width)

        current_speed = float(start_speed)
        current_speed = max(0, min(current_speed, 60))

        dt = 1.0 / fps
        frame_timestamp = 0.0

        status_rank = {
            "SAFE": 0,
            "WARNING": 1,
            "HIGH RISK": 2,
            "CRITICAL / BRAKE": 3,
        }

        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            if crop_dashboard:
                h, w, _ = frame.shape
                frame = frame[:int(h * 0.65), :]

            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except Exception as e:
                print(f"YOLO error on frame {frame_number}: {e}")
                results = []

            frame = draw_path_zone(frame, frame_width=width)

            worst_status = "SAFE"
            tracked_count = 0

            # ------------------------------------------------
            # Process detections 
            # ------------------------------------------------
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_object = results[0].boxes

                if boxes_object.id is not None:
                    boxes = boxes_object.xyxy.cpu().numpy()
                    track_ids = boxes_object.id.int().cpu().tolist()
                    classes = boxes_object.cls.int().cpu().tolist()

                    active_ids = set()

                    for i, box in enumerate(boxes):
                        track_id = track_ids[i]
                        active_ids.add(track_id)
                        class_id = classes[i]
                        raw_name = model.names.get(class_id, "Car")

                        class_name = raw_name.split("-", 1)[-1] if "-" in raw_name else raw_name
                        lookup_class = class_name if class_name in KNOWN_WIDTHS else "Car"

                        detection = {
                            "track_id": track_id,
                            "class": lookup_class,
                            "bbox": box.tolist(),
                            "timestamp": frame_timestamp,
                        }

                        try:
                            #تم تمرير frame_width=width هنا للالتزام بأبعاد الصورة الحقيقية
                            result = process_object_from_bbox(
                                detection,
                                focal_length=focal_length,
                                frame_width=width,
                                frame_height=height,
                                previous_state=previous_states.get(track_id),
                            )
                        except Exception as e:
                            print(f"Risk engine error: {e}")
                            result = None

                        if result is None:
                            continue

                        if not result.get("in_path", False):
                            continue

                        tracked_count += 1
                        previous_states[track_id] = result
                        frame = draw_object_result(frame, box.tolist(), result)
                        status = result.get("status", "SAFE")

                        if status in status_rank and status_rank[status] > status_rank[worst_status]:
                            worst_status = status

                    # Remove stale objects no longer tracked
                    for stale_id in list(previous_states.keys()):
                        if stale_id not in active_ids:
                            del previous_states[stale_id]

            # ------------------------------------------------
            # Dynamic speed simulation
            # ------------------------------------------------
            if worst_status == "CRITICAL / BRAKE":
                current_speed = max(0, current_speed - (10.0 / fps))
            elif worst_status == "HIGH RISK":
                current_speed = max(0, current_speed - (6.0 / fps))
            elif worst_status == "WARNING":
                current_speed = max(0, current_speed - (3.0 / fps))
            else:
                current_speed = min(60, current_speed + (1.5 / fps))

            draw_header(frame, worst_status, current_speed)
            draw_bottom_bar(frame, tracked_count)

            out.write(frame)

            if total_frames > 0:
                progress_value = frame_number / total_frames
                progress(min(0.98, progress_value), desc=f"Analyzing frame {frame_number}/{total_frames}")

            frame_number += 1
            frame_timestamp += dt

        cap.release()
        cap = None
        out.release()
        out = None

        if not os.path.exists(output_path):
            raise gr.Error("Output video was not created.")

        if os.path.getsize(output_path) <= 0:
            raise gr.Error("Output video is empty.")

        last_open_error = None
        for attempt in range(10):
            try:
                with open(output_path, "rb") as f:
                    f.read(1024)
                last_open_error = None
                break
            except PermissionError as e:
                last_open_error = e
                time.sleep(0.3)

        if last_open_error is not None:
            raise gr.Error("Output file locked by another process.")

        progress(1.0, desc="Analysis completed.")
        return output_path

    except gr.Error:
        raise
    except Exception as e:
        print(f"Processing error: {e}")
        raise gr.Error(f"Processing failed: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()


def reset_system():
    global previous_states
    previous_states = {}

    return (
        None,
        60,
        "System ready. Upload or record a video.",
    )


CUSTOM_CSS = """
:root {
    --accent: #4f8cff;
    --accent-soft: rgba(79, 140, 255, 0.15);
    --safe: #22c55e;
    --warn: #eab308;
    --high: #f97316;
    --crit: #ef4444;
    --panel-border: rgba(255, 255, 255, 0.08);
}

body, .gradio-container {
    background: radial-gradient(circle at 15% 0%, #16213a 0%, #0a0e17 45%, #05070b 100%) !important;
    color: #eef1f6 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: auto !important;
}

.hero {
    padding: 34px 42px;
    border-radius: 22px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(24, 33, 56, 0.95), rgba(8, 11, 18, 0.98));
    border: 1px solid var(--panel-border);
    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.45);
}

.hero-title {
    font-size: 40px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin: 0 !important;
    background: linear-gradient(90deg, #ffffff, #b9c8ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #9aa7bd !important;
    font-size: 15.5px !important;
    margin-top: 10px !important;
    max-width: 760px;
    line-height: 1.5;
}

.card {
    background: rgba(15, 20, 32, 0.9) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25);
    transition: border-color 0.25s ease;
}

.card:hover {
    border-color: rgba(79, 140, 255, 0.25) !important;
}

.primary-btn {
    min-height: 52px !important;
    border-radius: 12px !important;
    font-size: 15.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px;
}

button.primary {
    background: linear-gradient(135deg, #4f8cff, #2f5fd6) !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(79, 140, 255, 0.25) !important;
}

button.secondary {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid var(--panel-border) !important;
}

.video-container {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid var(--panel-border) !important;
}

label {
    font-weight: 600 !important;
    color: #cdd6e6 !important;
}

textarea, input {
    border-radius: 10px !important;
}

.status-safe { border-left: 4px solid var(--safe) !important; }
.status-warn { border-left: 4px solid var(--warn) !important; }
.status-high { border-left: 4px solid var(--high) !important; }
.status-crit { border-left: 4px solid var(--crit) !important; }

.footer {
    text-align: center;
    color: #5c6779;
    padding: 30px 10px 18px 10px;
    font-size: 12.5px;
    letter-spacing: 0.3px;
}

.footer-title {
    color: #aeb9cc;
    font-weight: 600;
    font-size: 13px;
}

.footer-tech {
    margin-top: 5px;
    color: #68758a;
    font-size: 12px;
}

.team-title {
    margin-top: 22px;
    margin-bottom: 11px;
    color: #cdd6e6;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.team-links {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.team-links a {
    display: inline-block;
    color: #8fb3ff !important;
    text-decoration: none !important;
    padding: 8px 14px;
    border: 1px solid rgba(79, 140, 255, 0.20);
    border-radius: 999px;
    background: rgba(79, 140, 255, 0.06);
    transition: all 0.2s ease;
    font-size: 12.5px;
    font-weight: 600;
}

.team-links a:hover {
    color: #ffffff !important;
    background: rgba(79, 140, 255, 0.16);
    border-color: rgba(79, 140, 255, 0.45);
    transform: translateY(-1px);
}

"""

with gr.Blocks(title="H.A.R.S", theme=gr.themes.Base()) as app:

    gr.HTML(
        """
        <div class="hero">
            <div class="hero-title">🚘 H.A.R.S</div>
            <div class="hero-subtitle">
                Intelligent computer vision system for object tracking,
                monocular distance estimation, path analysis and
                real-time collision risk assessment.
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### 🎥 Input Source\nUpload a driving video or record directly from your webcam.")

            video_input = gr.File(
                label="Upload Video",
                file_types=["video"],
                type="filepath",
                elem_classes="video-container",
            )

            gr.Markdown("### ⚙️ Simulation Settings")

            speed_slider = gr.Slider(
                minimum=0,
                maximum=60,
                value=60,
                step=1,
                label="Initial Vehicle Speed",
                info="Starting speed used by the risk simulation.",
            )
            crop_checkbox = gr.Checkbox(label="Crop Dashboard (For interior videos)", value=False)

            with gr.Row():
                analyze_btn = gr.Button("🚀 START COLLISION ANALYSIS", variant="primary", elem_classes="primary-btn")
                reset_btn = gr.Button("↻ RESET", variant="secondary", elem_classes="primary-btn")

            status_box = gr.Textbox(
                label="System Status",
                value="System ready. Upload or record a video.",
                interactive=False,
            )

        with gr.Column(scale=1.35, elem_classes="card"):
            gr.Markdown("### 🧠 Collision Analysis\nReal-time detection and risk visualization.")

            video_output = gr.Video(
                label="Collision Analysis Result",
                autoplay=False,
                format="mp4",
                elem_classes="video-container",
            )

            gr.Markdown(
                "#### Detection Pipeline\n"
                "**YOLO Detection → ByteTrack → Distance Estimation → Path Analysis → Risk Engine**"
            )

    with gr.Row():
        with gr.Column(elem_classes="card status-safe"):
            gr.Markdown("### 🟢 SAFE\nNo significant collision risk detected.")
        with gr.Column(elem_classes="card status-warn"):
            gr.Markdown("### 🟡 WARNING\nObject detected in a potentially unsafe area.")
        with gr.Column(elem_classes="card status-high"):
            gr.Markdown("### 🟠 HIGH RISK\nCollision probability is increasing.")
        with gr.Column(elem_classes="card status-crit"):
            gr.Markdown("### 🔴 CRITICAL / BRAKE\nImmediate braking response recommended.")

    gr.HTML(
        """
        <div class="footer">
            <div class="footer-title">H.A.R.S • Human-Aware Robotic Systems</div>
            <div class="footer-tech">YOLO + ByteTrack + Monocular Distance Estimation</div>
            <div class="team-title">Project Team</div>
            <div class="team-links">
                <a href="https://www.linkedin.com/in/hamdy-hamada-7b5453320" target="_blank" rel="noopener noreferrer">Hamdy Hamada</a>
                <a href="https://www.linkedin.com/in/omar-adeeb-275588361" target="_blank" rel="noopener noreferrer">Omar Adeeb</a>
                <a href="https://www.linkedin.com/in/sherief-ahmed-547b1b332" target="_blank" rel="noopener noreferrer">Sherif Ahmed Hamed</a>
                <a href="https://www.linkedin.com/in/mahmoud-ali-eng" target="_blank" rel="noopener noreferrer">Mahmoud Ali Mahmoud</a>
            </div>
        </div>
        """
    )

    def start_processing(video, speed, crop_dash, progress=gr.Progress()):
        result = process_video_pipeline(video, speed, crop_dashboard=crop_dash, progress=progress)
        return result, "Analysis completed successfully."

    analyze_btn.click(
        fn=start_processing,
        inputs=[video_input, speed_slider, crop_checkbox],
        outputs=[video_output, status_box],
    )
    reset_btn.click(
        fn=reset_system,
        inputs=[],
        outputs=[video_output, speed_slider, status_box],
    )

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
    )