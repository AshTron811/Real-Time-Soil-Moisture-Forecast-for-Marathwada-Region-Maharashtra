# Real‑Time Soil Moisture Forecast for Marathwada Region (Maharashtra)

**Project:** Soil moisture forecasting and downscaling pipeline for the Marathwada region.

**Author:** Ashutosh Sharma

---

## Overview

This repository contains an end‑to‑end pipeline for preparing environmental predictors, downscaling SMAP soil‑moisture to higher resolution using a Random Forest, and running a forecasting baseline using an LSTM model. The code is organized into standalone scripts for data export (from Earth Engine), downscaling, and model training/evaluation.

---

## Repository structure

Files at the repository root (observed):

* `chirps_export.py` — CHIRPS precipitation export / prep script
* `gldas_export.py` — GLDAS variables export / prep script
* `gldas_predictors.csv` — sample / extracted GLDAS predictors
* `modis_export.py` — MODIS export / prep script
* `srtm_export.py` — SRTM export / static predictors script
* `srtm_samples.csv` — sample SRTM-derived samples
* `rf_downscaling.py` — Random Forest downscaling of SMAP to 500 m
* `lstm_forecasting.py` — LSTM forecasting training & inference script
* `lstm_model.h5` — example / saved LSTM model weights
* `requirements.txt` — Python dependencies
* `last_trained_date.txt` — text file storing the last training date
* `__pycache__/` — Python bytecode cache

> If any files are missing locally after cloning, run `git pull` or check the repository web view.

---

## Requirements

Python 3.8+ and the packages listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The project expects access to Google Earth Engine for the export scripts.

---

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/AshTron811/Real-Time-Soil-Moisture-Forecast-for-Marathwada-Region-Maharashtra.git
cd Real-Time-Soil-Moisture-Forecast-for-Marathwada-Region-Maharashtra
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Authenticate Google Earth Engine (if not already):

```bash
pip install earthengine-api
earthengine authenticate
```

4. Export predictor data (example):

```bash
python gldas_export.py --start 2024-05-01 --end 2025-05-01
python modis_export.py --start 2024-05-01 --end 2025-05-01
python chirps_export.py --start 2024-05-01 --end 2025-05-01
python srtm_export.py
```

5. Run RF downscaling to create 500 m SM estimates:

```bash
python rf_downscaling.py --district Aurangabad --start 2024-05-01 --end 2025-05-01
```

6. Train / run the LSTM forecasting baseline:

```bash
python lstm_forecasting.py --train --district Aurangabad --epochs 50
```

Adjust script flags as required — check each script's `--help` or header comments for exact CLI arguments.

---

## Output & results

* Scripts save intermediate CSVs and model artifacts in the working directory (check each script for the output paths).
* `lstm_model.h5` is an example saved model; `last_trained_date.txt` contains the most recent training date recorded by the scripts.

---

## Tips & troubleshooting

* Earth Engine quotas: exporting long time series can take time — batch your exports and reuse cached files where possible.
* If any script fails due to missing credentials or API limits, authenticate Earth Engine and confirm network access.
* For reproducible results, use a consistent Python environment (virtualenv/conda) and the included `requirements.txt`.

---

## Contact

For questions or issues, open an issue on the repository or contact the author (see GitHub profile).
