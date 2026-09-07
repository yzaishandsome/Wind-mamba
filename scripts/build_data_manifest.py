"""Create record-count, time-range, and SHA-256 metadata for processed CSVs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("processed_data"))
    parser.add_argument("--output", type=Path, default=Path("preprocessing/processed_file_manifest.csv"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.data_root.glob("processed_sd*.csv")):
        frame = pd.read_csv(path, usecols=lambda column: column in {"time", "is_interpolated"})
        times = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        interpolated = frame.get("is_interpolated", pd.Series(dtype=bool))
        rows.append(
            {
                "file_name": path.name,
                "records": len(frame),
                "start_time_utc": times.min().isoformat(),
                "end_time_utc": times.max().isoformat(),
                "interpolated_records": int(interpolated.fillna(False).astype(bool).sum()),
                "sha256": sha256(path),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No processed_sd*.csv files found under {args.data_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
