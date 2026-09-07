"""Check the integrity and portability of the public reproduction package."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
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


def check_table8_mapping() -> None:
    path = ROOT / "results/submitted/repeated_core_ablation_raw.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant_key"], int(row["seed"]))].append(row)

    expected = {
        "full": {
            "mean": (0.8898172784355733, 0.6071562684901959, 8.676276048186427, 0.8754231000348532),
            "std": (0.0027337858056444, 0.0026511779184095, 0.0964216920426681, 0.0008263072444933),
        },
        "mlp_decoder": {
            "mean": (0.8849118330357462, 0.5985332922532457, 8.504349406387409, 0.8760917307258935),
            "std": (0.0053207489889797, 0.0031808998222584, 0.0577482139010790, 0.0018334427770434),
        },
    }
    columns = ("test_ws_rmse", "test_ws_mae", "test_wd_mae", "test_r2")
    for variant, reference in expected.items():
        seed_means = []
        for seed in (42, 43, 44, 45, 46):
            target_rows = grouped[(variant, seed)]
            assert sorted(row["target"] for row in target_rows) == ["sd1042", "sd1091"]
            seed_means.append(tuple(statistics.mean(float(row[column]) for row in target_rows) for column in columns))
        observed_means = tuple(statistics.mean(values) for values in zip(*seed_means))
        observed_stds = tuple(statistics.stdev(values) for values in zip(*seed_means))
        assert all(
            math.isclose(value, expected_value, abs_tol=1e-12)
            for value, expected_value in zip(observed_means, reference["mean"])
        )
        assert all(
            math.isclose(value, expected_value, abs_tol=1e-12)
            for value, expected_value in zip(observed_stds, reference["std"])
        )


def check_experimental_thresholds() -> None:
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or path.resolve() == Path(__file__).resolve():
            continue
        if "10.59" in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise RuntimeError("Obsolete all-data 10.59 threshold remains in executable code:\n" + "\n".join(offenders))


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
    check_table8_mapping()
    check_experimental_thresholds()
    check_portability()
    print("Release integrity check passed.")


if __name__ == "__main__":
    main()
