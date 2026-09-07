# Saildrone data acquisition

The repository does not redistribute NOAA mission files. Download the relevant
Saildrone products from the [NOAA PMEL ERDDAP catalog](https://data.pmel.noaa.gov/pmel/erddap/info/index.html?page=1&itemsPerPage=1000)
and export the ten vessel files as CSV:

`sd1031.csv`, `sd1033.csv`, `sd1036.csv`, `sd1040.csv`, `sd1041.csv`,
`sd1042.csv`, `sd1057.csv`, `sd1069.csv`, `sd1083.csv`, and `sd1091.csv`.

Place them under `raw_data/`. The exported files must contain a timestamp plus
the variables listed in `preprocess_saildrone.py`. NOAA ERDDAP CSV exports often
contain a units row immediately after the header; the reported preprocessing
script skips that row.

Run:

```bash
python preprocessing/preprocess_saildrone.py --raw-dir raw_data --output-dir processed_data
python scripts/build_data_manifest.py --data-root processed_data
```

`retained_segments.csv` records the exact retained temporal ranges used in the
study. `processed_file_manifest.csv` records the corresponding record counts and
SHA-256 checksums so that a local preprocessing run can be verified.
