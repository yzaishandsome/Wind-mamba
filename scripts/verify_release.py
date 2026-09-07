"""Check the integrity and portability of the public reproduction package."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "environment.yml",
    "configs/asoc_revision.json",
    "preprocessing/preprocess_saildrone.py",
    "preprocessing/DATA_ACQUISITION.md",
    "preprocessing/retained_segments.csv",
    "preprocessing/processed_file_manifest.csv",
    "reproduction/submitted/main.py",
    "reproduction/submitted/run_baselines.py",
    "reproduction/submitted/run_statistical_baselines.py",
    "reproduction/submitted/run_ablation.py",
    "reproduction/submitted/run_loss_ablation.py",
    "reproduction/revision/run_task1_dpfmformer.py",
    "reproduction/revision/run_task2_spectral_phase.py",
    "reproduction/revision/run_task3_residual_bounds.py",
    "reproduction/revision/run_task4_weighted_loss.py",
    "reproduction/revision/run_task5_circular_3seed.py",
    "reproduction/revision/run_horizon_and_bootstrap.py",
    "reproduction/revision/run_conformal_clean.py",
    "scripts/generate_tables.py",
    "scripts/generate_fig9.py",
    "results/revision/fig9_candidate_sd1042.npz",
]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml"}
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\")


def require_paths() -> None:
    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).exists()]
    if missing:
        raise RuntimeError("Missing required release files:\n" + "\n".join(missing))


def check_configuration() -> None:
    config = json.loads((ROOT / "configs/asoc_revision.json").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    revision = config["revision"]

    assert dataset["source_vessels"] == [
        "sd1031", "sd1033", "sd1036", "sd1040",
        "sd1041", "sd1057", "sd1069", "sd1083",
    ]
    assert dataset["target_vessels"] == ["sd1042", "sd1091"]
    assert (dataset["train_ratio"], dataset["validation_ratio"], dataset["test_ratio"]) == (0.6, 0.2, 0.2)
    assert (dataset["sequence_length"], dataset["prediction_length"], dataset["window_stride"]) == (36, 6, 1)
    assert (model["input_dimension"], model["hidden_dimension"], model["mamba_layers"], model["state_dimension"]) == (10, 96, 3, 16)
    assert training["seeds"] == [42, 43, 44, 45, 46]
    assert revision["source_training_ws_p95_mps"] == 10.580439745930407


def check_manifests() -> None:
    with (ROOT / "preprocessing/retained_segments.csv").open(newline="", encoding="utf-8") as stream:
        retained = list(csv.DictReader(stream))
    with (ROOT / "preprocessing/processed_file_manifest.csv").open(newline="", encoding="utf-8") as stream:
        processed = list(csv.DictReader(stream))
    assert len(retained) == 10
    assert len(processed) == 10
    assert all(len(row["sha256"]) == 64 for row in processed)


def check_portability() -> None:
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if WINDOWS_PATH.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise RuntimeError("Local absolute Windows paths remain in:\n" + "\n".join(offenders))


def main() -> None:
    require_paths()
    check_configuration()
    check_manifests()
    check_portability()
    print("Release integrity check passed.")


if __name__ == "__main__":
    main()
