from pathlib import Path


ALL_BOAT_FILES = [
    "processed_data/processed_sd1031.csv",
    "processed_data/processed_sd1033.csv",
    "processed_data/processed_sd1036.csv",
    "processed_data/processed_sd1040.csv",
    "processed_data/processed_sd1041.csv",
    "processed_data/processed_sd1042.csv",
    "processed_data/processed_sd1057.csv",
    "processed_data/processed_sd1069.csv",
    "processed_data/processed_sd1083.csv",
    "processed_data/processed_sd1091.csv",
]

TARGET_BOAT_FILES = [
    "processed_data/processed_sd1042.csv",
    "processed_data/processed_sd1091.csv",
]

SOURCE_BOAT_FILES = [file_path for file_path in ALL_BOAT_FILES if file_path not in TARGET_BOAT_FILES]

BOAT_ID_MAP = {file_path: idx for idx, file_path in enumerate(ALL_BOAT_FILES)}

EXPERIMENT_VERSION = "transfer_two_targets_v1"


def boat_tag(file_path):
    return Path(file_path).stem.replace("processed_", "").lower()


def boat_ids_for(file_paths):
    return [BOAT_ID_MAP[file_path] for file_path in file_paths]
