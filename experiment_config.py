import os
from pathlib import Path


# Raw Saildrone mission files can be obtained from the NOAA PMEL ERDDAP/NetCDF
# data service. This repository expects users to preprocess those raw files into
# the CSV schema consumed by data_provider.py.
NOAA_PMEL_ERDDAP_URL = "https://data.pmel.noaa.gov/pmel/erddap/info/index.html?page=1&itemsPerPage=1000"

# By default, processed CSV files are expected under ./processed_data.
# Set WIND_MAMBA_DATA_ROOT to your local processed-data directory if needed.
DATA_ROOT = Path(os.getenv("WIND_MAMBA_DATA_ROOT", "processed_data"))

VESSEL_IDS = [
    "sd1031",
    "sd1033",
    "sd1036",
    "sd1040",
    "sd1041",
    "sd1042",
    "sd1057",
    "sd1069",
    "sd1083",
    "sd1091",
]

TARGET_VESSEL_IDS = ["sd1042", "sd1091"]

ALL_BOAT_FILES = [str(DATA_ROOT / f"processed_{vessel_id}.csv") for vessel_id in VESSEL_IDS]

TARGET_BOAT_FILES = [str(DATA_ROOT / f"processed_{vessel_id}.csv") for vessel_id in TARGET_VESSEL_IDS]

SOURCE_BOAT_FILES = [file_path for file_path in ALL_BOAT_FILES if file_path not in TARGET_BOAT_FILES]

BOAT_ID_MAP = {file_path: idx for idx, file_path in enumerate(ALL_BOAT_FILES)}

EXPERIMENT_VERSION = "transfer_two_targets_v1"

# ASOC benchmark protocol. Environment variables in main.py may override the
# shape parameters for diagnostic runs, while these values define the reported
# configuration.
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
