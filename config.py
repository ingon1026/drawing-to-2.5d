"""
drawing-2p5d — Configuration
"""

# === Model ===
MODEL_PATH = "models/magic_touch.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "interactive_segmenter/magic_touch/float32/1/magic_touch.tflite"
)

# === Normalization ===
GAUSSIAN_BLUR_KSIZE = 51  # shadow removal (must be odd)

# === Segmentation ===
CONFIDENCE_THRESHOLD = 0.1  # category_mask binary threshold

# === Postprocessing ===
MORPH_KERNEL_SIZE = 5
MORPH_CLOSE_ITER = 3  # fill gaps
MORPH_OPEN_ITER = 2   # remove noise
MIN_CONTOUR_AREA_RATIO = 0.001  # minimum component area / image area

# === SAM2 Auto-Mask ===
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = "models/sam2.1_hiera_tiny.pt"
SAM2_POINTS_PER_SIDE = 16     # grid density for auto-mask
SAM2_IOU_THRESH = 0.7
SAM2_STABILITY_THRESH = 0.8
SAM2_MIN_AREA = 500           # minimum mask region area in pixels
SAM2_FRAME_DIFF_THRESH = 15.0 # mean pixel diff to trigger re-segmentation

# === Depth Estimation ===
DEPTH_MODEL_NAME = "Intel/dpt-hybrid-midas"
NORMAL_STRENGTH = 1.0  # normal map intensity (higher = more dramatic)

# === Export ===
OUTPUT_DIR = "output"
MASK_FILENAME = "mask.png"
OBJECT_FILENAME = "object.png"
DEPTH_FILENAME = "depth.png"
NORMAL_FILENAME = "normal.png"
DEBUG_OVERLAY_FILENAME = "debug_overlay.png"
