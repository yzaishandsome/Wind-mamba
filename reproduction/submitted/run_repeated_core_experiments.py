import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_STEPS = ["edgewind", "baselines", "ablation", "loss", "replacement"]
EXPERIMENT_VERSION = "transfer_two_targets_v1"
DEFAULT_SUFFIX_PREFIX = "rev2_seed"


def parse_csv_env(name, default_values):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return list(default_values)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def seed_suffix(seed):
    prefix = os.getenv("EDGEWIND_REPEAT_SUFFIX_PREFIX", DEFAULT_SUFFIX_PREFIX).strip()
    return f"{prefix}_{seed}"


def expected_output(step, suffix):
    if step == "edgewind":
        return Path("weights") / EXPERIMENT_VERSION / "edgewind" / suffix / "edgewind_summary.csv"
    if step == "baselines":
        return Path("weights") / EXPERIMENT_VERSION / "baselines_v2" / suffix / "baseline_training_summary.csv"
    if step == "ablation":
        return Path("comparison_outputs") / EXPERIMENT_VERSION / "ablation" / suffix / "ablation_summary.csv"
    if step == "loss":
        return Path("comparison_outputs") / EXPERIMENT_VERSION / "loss_ablation" / suffix / "loss_ablation_summary.csv"
    if step == "replacement":
        return (
            Path("comparison_outputs")
            / EXPERIMENT_VERSION
            / "mamba_transformer_replacement"
            / suffix
            / "mamba_transformer_replacement_summary.csv"
        )
    raise ValueError(f"Unknown repeated-experiment step: {step}")


def command_for_step(step):
    if step == "edgewind":
        return [sys.executable, "main.py"]
    if step == "baselines":
        return [sys.executable, "run_baselines.py"]
    if step == "ablation":
        return [sys.executable, "run_ablation.py"]
    if step == "loss":
        return [sys.executable, "run_loss_ablation.py"]
    if step == "replacement":
        return [sys.executable, "run_mamba_transformer_replacement.py"]
    raise ValueError(f"Unknown repeated-experiment step: {step}")


def step_env(base_env, step):
    env = dict(base_env)
    env.setdefault("EDGEWIND_MAMBA_BACKEND", "custom")
    env.setdefault("EDGEWIND_LOSS_MODE", "smoothl1_dircos")
    env.setdefault("EDGEWIND_EXTREME_WS_THRESHOLD", "10.580439745930407")
    if step == "edgewind":
        env.setdefault("EDGEWIND_RUN_NO_TRANSFER", "0")
    if step == "baselines":
        env.setdefault("EDGEWIND_RUN_NO_TRANSFER", "0")
        env.setdefault(
            "EDGEWIND_BASELINE_MODELS",
            "bp,cnnlstm,tcnlstm,informer,transformer,autoformer,timesnet,stdmamba",
        )
    return env


def run_step(seed, step, force=False):
    suffix = seed_suffix(seed)
    output_path = expected_output(step, suffix)
    if output_path.exists() and not force:
        print(f"[skip] seed={seed} step={step} already exists: {output_path}")
        return

    log_dir = Path("logs") / "repeated_core"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{suffix}_{step}.out.log"
    err_log = log_dir / f"{suffix}_{step}.err.log"

    env = os.environ.copy()
    env["EDGEWIND_SEED"] = str(seed)
    env["EDGEWIND_OUTPUT_SUFFIX"] = suffix
    env = step_env(env, step)

    cmd = command_for_step(step)
    print(f"[run] seed={seed} step={step}: {' '.join(cmd)}")
    with out_log.open("w", encoding="utf-8") as stdout_file, err_log.open("w", encoding="utf-8") as stderr_file:
        subprocess.run(cmd, check=True, env=env, stdout=stdout_file, stderr=stderr_file)


def main():
    seeds = [int(item) for item in parse_csv_env("EDGEWIND_REPEAT_SEEDS", DEFAULT_SEEDS)]
    steps = parse_csv_env("EDGEWIND_REPEAT_STEPS", DEFAULT_STEPS)
    force = os.getenv("EDGEWIND_REPEAT_FORCE", "0") == "1"

    print(f"Repeated seeds: {seeds}")
    print(f"Repeated steps: {steps}")
    print(f"Force rerun: {force}")

    for seed in seeds:
        for step in steps:
            run_step(seed, step, force=force)


if __name__ == "__main__":
    main()
