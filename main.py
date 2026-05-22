"""
main.py — FireSmokeDetection System Entry Point
================================================
Run this file to start the real-time fire and smoke detection pipeline.

Quick start
-----------
    # Webcam (default)
    python main.py

    # Specific webcam index
    python main.py --source 1

    # Video file
    python main.py --source path/to/video.mp4

    # RTSP / IP camera stream
    python main.py --source "rtsp://admin:password@192.168.1.100/stream1"

    # CCTV HTTP stream
    python main.py --source "http://192.168.1.101/video"

    # Custom model weights
    python main.py --model models/my_fire_model.pt

    # Headless mode (no window — useful on servers / Raspberry Pi)
    python main.py --no-window

    # Adjust confidence thresholds
    python main.py --conf 0.5 --alert-conf 0.65

    # Change alert cooldown (seconds)
    python main.py --cooldown 60

CLI Flags
---------
    --source       Video source. Int for webcam index, string for path/URL.
    --model        Path to YOLOv8 .pt weights file.
    --conf         Min confidence to draw bounding box (default 0.45).
    --alert-conf   Min confidence to trigger alert    (default 0.55).
    --cooldown     Seconds between consecutive alerts (default 30).
    --no-window    Run headless (no cv2.imshow display).
    --no-save      Do not save annotated video to output/.
"""

import argparse
import sys
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Fire & Smoke Detection System (YOLOv8 + OpenCV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Video source ─────────────────────────────────────────
    parser.add_argument(
        "--source",
        default=0,
        help=(
            "Video source. Use 0/1/2 for webcam, or provide a file path, "
            "RTSP URL, or HTTP stream URL. (default: 0 = webcam)"
        ),
    )

    # ── Model path ───────────────────────────────────────────
    parser.add_argument(
        "--model",
        default=os.path.join("models", "fire_smoke_yolov8.pt"),
        help="Path to YOLOv8 .pt weights. Falls back to yolov8n.pt if not found.",
    )

    # ── Detection thresholds ─────────────────────────────────
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Minimum confidence to draw a bounding box (0.0–1.0, default 0.45).",
    )
    parser.add_argument(
        "--alert-conf",
        type=float,
        default=0.55,
        dest="alert_conf",
        help="Minimum confidence to trigger an SMS/email alert (default 0.55).",
    )

    # ── Alert cooldown ───────────────────────────────────────
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Seconds between consecutive alerts to avoid spam (default 30).",
    )

    # ── Display / output flags ───────────────────────────────
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run headless without cv2.imshow (useful on remote servers).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated video to output/.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Convert source to int if it looks like a webcam index ──
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print("=" * 60)
    print("  🔥  FireSmokeDetection AI System  🔥")
    print("=" * 60)
    print(f"  Source       : {source}")
    print(f"  Model        : {args.model}")
    print(f"  Conf (draw)  : {args.conf}")
    print(f"  Conf (alert) : {args.alert_conf}")
    print(f"  Cooldown     : {args.cooldown}s")
    print(f"  Window       : {'No (headless)' if args.no_window else 'Yes'}")
    print(f"  Save output  : {'No' if args.no_save else 'Yes'}")
    print("=" * 60)
    print()

    # ── Import and run detection pipeline ───────────────────
    try:
        from detect import run_detection
        run_detection(
            source=source,
            model_path=args.model,
            conf_threshold=args.conf,
            alert_conf_threshold=args.alert_conf,
            cooldown=args.cooldown,
            show_window=not args.no_window,
            save_output=not args.no_save,
        )
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user (Ctrl+C). Exiting …")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        raise


if __name__ == "__main__":
    main()
