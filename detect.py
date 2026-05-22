"""
detect.py — Real-Time Fire & Smoke Detection Engine
=====================================================
This module is the core of the detection pipeline. It:

  1. Opens a video source (webcam / RTSP / IP cam / video file).
  2. Feeds each frame to a YOLOv8 model for inference.
  3. Draws bounding boxes, labels, and confidence scores on detections.
  4. Saves screenshots automatically on every new hazard event.
  5. Enforces a cooldown timer so alerts aren't spammed.
  6. Calls alert.dispatch_alerts() to send SMS + email.
  7. Renders a live annotated feed with FPS overlay.

Class labels expected from the model
--------------------------------------
  0 → fire
  1 → smoke
(These match the public fire-smoke YOLOv8 dataset label order.)

Usage
-----
    from detect import run_detection
    run_detection(source=0)           # webcam
    run_detection(source="rtsp://…")  # RTSP stream
    run_detection(source="video.mp4") # video file
"""

import cv2
import time
import os
from datetime import datetime

import numpy as np

# Ultralytics YOLOv8 — lazy import so the rest of the system still loads
# even if torch isn't installed (useful for running alert.py alone in tests)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – detection disabled.")

from alert import dispatch_alerts

# ════════════════════════════════════════════════════════════════
# Configuration constants  (tweak without touching logic below)
# ════════════════════════════════════════════════════════════════

# Path to the trained YOLOv8 weights file.
# Drop your custom fire/smoke .pt file into models/ and update this.
MODEL_PATH = os.path.join("models", "fire_smoke_yolov8.pt")

# Fallback to the pretrained nano model if custom weights are absent.
# Replace with "yolov8s.pt" / "yolov8m.pt" for better accuracy at the
# cost of speed.
FALLBACK_MODEL = "yolov8n.pt"

# Minimum confidence to draw a box (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.45

# Minimum confidence before triggering an alert (slightly higher than draw)
ALERT_CONFIDENCE_THRESHOLD = 0.55

# Seconds to wait before sending another alert for the same hazard
ALERT_COOLDOWN_SECONDS = 30

# Directory to save screenshots on detection
SCREENSHOT_DIR = "screenshots"

# Maximum frame width for display (resizes large RTSP feeds to save CPU)
DISPLAY_WIDTH = 960

# Class names  (index must match model's data.yaml)
CLASS_NAMES = {0: "Fire", 1: "Smoke"}

# Bounding box colors per class  (BGR)
CLASS_COLORS = {
    0: (0,   60,  255),   # vivid red-orange → Fire
    1: (130, 130, 130),   # mid grey         → Smoke
}

# ════════════════════════════════════════════════════════════════
# Helper: save an annotated screenshot to disk
# ════════════════════════════════════════════════════════════════

def save_screenshot(frame: np.ndarray, hazard_type: str) -> str:
    """
    Saves `frame` as a JPEG in the screenshots/ directory.

    Returns the absolute path of the saved file so it can be included
    in alert messages.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{hazard_type.replace(' ', '_')}_{ts}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"[SCREENSHOT] Saved → {filepath}")
    return os.path.abspath(filepath)


# ════════════════════════════════════════════════════════════════
# Helper: draw overlay text on the frame
# ════════════════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray, fps: float, alert_active: bool) -> None:
    """
    Draws a heads-up display on the frame in-place:
      • FPS counter (top-left)
      • Live status badge (top-right)
      • ⚠ HAZARD DETECTED ribbon (bottom, when alert_active)
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark bar across the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

    # LIVE badge
    badge_txt = "● LIVE"
    (tw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, badge_txt, (w - tw - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

    # Hazard ribbon at the bottom
    if alert_active:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 40), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "⚠  HAZARD DETECTED — ALERT SENT",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ════════════════════════════════════════════════════════════════
# Helper: annotate a single detection on the frame
# ════════════════════════════════════════════════════════════════

def draw_detection(frame: np.ndarray, box, cls_id: int, conf: float) -> None:
    """
    Draws a bounding box + label with confidence score for one detection.

    Parameters
    ----------
    frame  : BGR frame (modified in-place)
    box    : xyxy tensor / array  [x1, y1, x2, y2]
    cls_id : integer class index
    conf   : confidence score  0.0 – 1.0
    """
    x1, y1, x2, y2 = map(int, box)
    label   = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
    color   = CLASS_COLORS.get(cls_id, (255, 255, 255))
    caption = f"{label}  {conf:.0%}"

    # Outer rectangle (thick)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Label background pill
    (tw, th), baseline = cv2.getTextSize(
        caption, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 6
    cv2.rectangle(frame,
                  (x1, y1 - th - baseline - pad * 2),
                  (x1 + tw + pad * 2, y1),
                  color, -1)

    cv2.putText(frame, caption,
                (x1 + pad, y1 - baseline - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ════════════════════════════════════════════════════════════════
# Main detection loop
# ════════════════════════════════════════════════════════════════

def run_detection(
    source=0,
    model_path: str = MODEL_PATH,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    alert_conf_threshold: float = ALERT_CONFIDENCE_THRESHOLD,
    cooldown: int = ALERT_COOLDOWN_SECONDS,
    show_window: bool = True,
    save_output: bool = True,
) -> None:
    """
    Main entry point — starts the real-time detection loop.

    Parameters
    ----------
    source              : int (0 = webcam) | str (path/URL)
    model_path          : path to YOLOv8 .pt weights
    conf_threshold      : min confidence to draw bounding box
    alert_conf_threshold: min confidence to trigger alert
    cooldown            : seconds between consecutive alerts
    show_window         : whether to open cv2.imshow window
    save_output         : whether to save annotated video to output/
    """

    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        return

    # ── Load model ───────────────────────────────────────────
    if os.path.exists(model_path):
        print(f"[MODEL] Loading custom weights: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"[MODEL] Custom weights not found at '{model_path}'.")
        print(f"[MODEL] Falling back to pretrained '{FALLBACK_MODEL}' …")
        print("[MODEL] NOTE: The pretrained model detects 80 COCO classes,")
        print("[MODEL]       not fire/smoke. Train or download custom weights.")
        model = YOLO(FALLBACK_MODEL)

    # ── Open video source ────────────────────────────────────
    print(f"[VIDEO] Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        print("        • For webcam  : use source=0")
        print("        • For RTSP    : use source='rtsp://user:pass@ip/stream'")
        print("        • For video   : use source='path/to/video.mp4'")
        return

    # ── Output video writer (optional) ───────────────────────
    video_writer = None
    if save_output:
        os.makedirs("output", exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("output", f"detection_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w, h))
        print(f"[OUTPUT] Saving annotated video → {out_path}")

    # ── Alert state ──────────────────────────────────────────
    last_alert_time: float = 0.0   # epoch seconds of last alert
    alert_ribbon_until: float = 0.0  # show ribbon for N seconds after alert

    print("\n[SYSTEM] Detection started. Press  Q  to quit.\n")

    prev_time = time.time()

    # ══════════════════════════════════════════════════════════
    # Frame loop
    # ══════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()

        # Handle end of file or lost connection gracefully
        if not ret:
            print("[VIDEO] Stream ended or frame read failed. Restarting …")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video files
            time.sleep(0.5)
            cap = cv2.VideoCapture(source)          # reconnect streams
            continue

        # ── Resize for consistent display speed ─────────────
        h_orig, w_orig = frame.shape[:2]
        if w_orig > DISPLAY_WIDTH:
            scale  = DISPLAY_WIDTH / w_orig
            frame  = cv2.resize(frame, (DISPLAY_WIDTH, int(h_orig * scale)))

        # ── YOLOv8 inference ─────────────────────────────────
        results = model(frame, conf=conf_threshold, verbose=False)

        # ── Parse detections ─────────────────────────────────
        detected_classes: list[str] = []
        max_confidence: float = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])

                # Only process known fire/smoke classes
                if cls_id not in CLASS_NAMES:
                    continue

                draw_detection(frame, box.xyxy[0], cls_id, conf)
                detected_classes.append(CLASS_NAMES[cls_id])
                max_confidence = max(max_confidence, conf)

        # ── Determine hazard type string ─────────────────────
        has_fire  = "Fire"  in detected_classes
        has_smoke = "Smoke" in detected_classes

        if has_fire and has_smoke:
            hazard_type = "Fire & Smoke"
        elif has_fire:
            hazard_type = "Fire"
        elif has_smoke:
            hazard_type = "Smoke"
        else:
            hazard_type = None

        # ── Alert logic with cooldown ─────────────────────────
        now = time.time()
        should_alert = (
            hazard_type is not None
            and max_confidence >= alert_conf_threshold
            and (now - last_alert_time) >= cooldown
        )

        if should_alert:
            last_alert_time    = now
            alert_ribbon_until = now + 5   # show ribbon for 5 s

            # Save screenshot FIRST so path is available in alert
            screenshot_path = save_screenshot(frame, hazard_type)

            # Fire SMS + email in sequence (consider threading for production)
            dispatch_alerts(hazard_type, max_confidence, screenshot_path)

        # ── HUD overlay ──────────────────────────────────────
        current_time = time.time()
        fps          = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time    = current_time
        alert_active = now < alert_ribbon_until
        draw_hud(frame, fps, alert_active)

        # ── Write to output video ────────────────────────────
        if video_writer is not None:
            video_writer.write(frame)

        # ── Display window ───────────────────────────────────
        if show_window:
            cv2.imshow("FireSmokeDetection — Real-Time Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or Esc
                print("[SYSTEM] Quit signal received.")
                break

    # ── Cleanup ──────────────────────────────────────────────
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Detection stopped. Resources released.")
