"""
AR-BOOK Drawing Classifier — Configuration
"""

# === Classes ===
CLASSES = ["animal", "candy", "flower", "human", "weapon"]  # alphabetical (matches image_dataset_from_directory)
NUM_CLASSES = len(CLASSES)

# === Quick Draw sub-classes → super-class ===
QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
QUICKDRAW_MAP = {
    "animal": {"tiger": 2000, "pig": 2000, "rabbit": 2000, "cat": 2000},
    "candy":  {"lollipop": 2000, "cookie": 2000, "cake": 2000, "ice cream": 2000},
    "flower": {"flower": 2000},
    "human":  {"face": 1600, "yoga": 400},
    "weapon": {"sword": 500, "knife": 500, "axe": 500, "rifle": 2000},
}

# === ImageNet-Sketch: WordNet ID → super-class ===
IMAGENET_SKETCH_MAP = {
    "animal": {
        "n02129604": "tiger",
        "n02396427": "hog",
        "n02325366": "wood_rabbit",
        "n02129165": "lion",
    },
    "candy": {
        "n07614500": "ice_cream",
        "n07615774": "ice_lolly",
        "n03089624": "confectionery",
    },
    "flower": {
        "n11939491": "daisy",
        "n12057211": "yellow_ladys_slipper",
        "n11879895": "rapeseed",
    },
    # human: no ImageNet-Sketch mapping (Quick Draw only)
    "weapon": {
        "n04090263": "rifle",
        "n03498962": "hatchet",
    },
}

# === Data ===
VAL_RATIO = 0.2

# === Image ===
IMG_SIZE = 224

# === Training (2-phase) ===
BATCH_SIZE = 64
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-4

# === Paths ===
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
TFLITE_DIR = "outputs/tflite"
