import os
from pathlib import Path


DATA_ROOT = Path(os.getenv("WIND_MAMBA_DATA_ROOT", "processed_data"))
VESSEL_IDS = (
    "sd1031", "sd1033", "sd1036", "sd1040", "sd1041",
    "sd1042", "sd1057", "sd1069", "sd1083", "sd1091",
)

ALL_BOAT_FILES = [str(DATA_ROOT / f"processed_{vessel}.csv") for vessel in VESSEL_IDS]

TARGET_BOAT_FILES = [str(DATA_ROOT / f"processed_{vessel}.csv") for vessel in ("sd1042", "sd1091")]

SOURCE_BOAT_FILES = [file_path for file_path in ALL_BOAT_FILES if file_path not in TARGET_BOAT_FILES]

BOAT_ID_MAP = {file_path: idx for idx, file_path in enumerate(ALL_BOAT_FILES)}

EXPERIMENT_VERSION = "transfer_two_targets_v1"

SEEDS = (42, 43, 44, 45, 46)
SEQ_LEN = 36
PRED_LEN = 6
INPUT_DIM = 10
HIDDEN_DIM = 96
D_STATE = 16
MAMBA_LAYERS = 3
DIRECTION_LOSS_WEIGHT = 1.0
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20
WINDOW_STRIDE = 1
SOURCE_TRAINING_WS_P95 = 10.580439745930407


def boat_tag(file_path):
    return Path(file_path).stem.replace("processed_", "").lower()


def boat_ids_for(file_paths):
    return [BOAT_ID_MAP[file_path] for file_path in file_paths]
