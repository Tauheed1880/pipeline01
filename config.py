"""Central configuration for the RTSP vehicle-detection pipeline.

Every value can be overridden with an environment variable so the same code
runs unchanged on a workstation, on-site, or inside Docker.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _get_class_ids():
    raw = os.getenv("VEHICLE_CLASS_IDS", "2,3,5,7").strip().lower()
    if raw in {"", "all", "none", "*"}:
        return None
    return {int(x) for x in raw.split(",") if x.strip().isdigit()}


# --- Video source ----------------------------------------------------------
# Your camera serves an HLS stream. The page http://10.1.2.251:8888/traffic/
# is only a player; the real video URL is the .m3u8 playlist below.
# (A real rtsp://... URL also works -- the reader picks options per scheme.)
RTSP_URL = os.getenv("RTSP_URL", "http://10.1.2.251:8888/traffic/index.m3u8")
RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "2.0"))

# --- Camera label (requirement #5 -> always cam 01) ------------------------
CAMERA_NUMBER = os.getenv("CAMERA_NUMBER", "cam 01")

# --- Web server ------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# --- Models (requirement: use files stored in models/) ---------------------
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", str(MODELS_DIR / "yolo26m.pt"))
PLATE_MODEL = os.getenv("PLATE_MODEL", str(MODELS_DIR / "license_plate_detector.pt"))

# --- Inference -------------------------------------------------------------
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
VEHICLE_CONF = float(os.getenv("VEHICLE_CONF", "0.40"))
PLATE_CONF = float(os.getenv("PLATE_CONF", "0.30"))
# COCO vehicle ids by default (car, motorcycle, bus, truck). Use "all" to keep
# every class (needed when you plug in a custom vehicle model).
VEHICLE_CLASS_IDS = _get_class_ids()
DEVICE = os.getenv("DEVICE", "") or None  # None = auto, or "cpu", "0"

# --- Low-latency stream output --------------------------------------------
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "75"))

# --- Tracking / recording --------------------------------------------------
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.30"))
MAX_MISSED = int(os.getenv("MAX_MISSED", "30"))
# Record a vehicle once a plate is found, or after this many tracked frames.
RECORD_GRACE_FRAMES = int(os.getenv("RECORD_GRACE_FRAMES", "12"))

# --- Output (requirement #6/#7 snapshots + JSON records) -------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
PLATE_DIR = OUTPUT_DIR / "plates"
RECORDS_FILE = Path(os.getenv("RECORDS_FILE", str(OUTPUT_DIR / "detections.json")))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
