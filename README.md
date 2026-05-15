# BBO-DRL: Bio-Inspired Adaptive Task Offloading for IoT-Edge-Cloud Healthcare Networks

A hybrid Bombardier Beetle Optimizer / Deep Q-Network scheduling system for real-time, Criticality-Index-aware task offloading in IoMT deployments, developed as part of a Q1 IEEE J-BHI submission.

---

## What This Repository Contains

| Directory / File | Contents |
|---|---|
| `docs/Patent.md` | Full provisional patent text: architecture claims, system overview, background |
| `docs/Plan.md` | Strategic roadmap from patent to journal publication; mathematical model references |
| `code/` | Complete Python simulation framework — algorithms, data ingestion, Monte Carlo runner, XAI |
| `data/` | Placeholder directory for the four publicly available datasets (not committed to repo) |
| `simulators/` | Supplementary simulation scripts and environment configuration files |
| `latex/` | LaTeX manuscript source, figures, and bibliography |
| `figures/` | Architecture and trade-off diagrams (Figure 1–3 referenced in patent) |
| `submission/` | Cover letter, complexity analysis supplement, data availability statement, reviewer anticipation notes |
| `README.md` | This file |

---

## Dependencies

Python 3.10 or later is required. Install all dependencies with:

```bash
pip install -r code/requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `wfdb` (MIT-BIH parsing), `openpyxl` (Mendeley XLSX), `tqdm`, `pyyaml`. Optional: `shap`, `scikit-learn` (required for XAI module). No PyTorch or TensorFlow dependency — the DQN is implemented in pure NumPy.

---

## Data Setup

Four publicly available datasets are required. Download them and place them in the `data/` directory as follows:

```
data/
  MIT-BIH/
      100.dat  100.hea  101.dat  101.hea  ...  (48 records)
      Download: https://physionet.org/content/mitdb/1.0.0/
      Command:  wfdb.dl_database('mitdb', 'data/MIT-BIH/')

  Mendeley-IoMT/
      patients_data_with_alerts.xlsx
      Download: https://doi.org/10.17632/7dpr3yvptn.1

  CICIoMT2024/
      <CSV traffic trace files>
      Download: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

  MedSec-25/
      <CSV files from Kaggle download>
      Download: https://www.kaggle.com/datasets/abdullah001234/medsec-25-iomt-cybersecurity-dataset
```

The data ingestion pipeline handles all format differences between datasets. MIT-BIH records are read using the `wfdb` library; Mendeley data is read from XLSX using `openpyxl`; CICIoMT2024 and MedSec-25 are read from CSV using `pandas`.

---

## Quick Start: Sanity Check

Run the integration test from the project root to verify that all six schedulers execute correctly on 500 synthetic tasks without requiring any dataset download:

```bash
python code/run_test.py
```

Expected output (approximate values — exact numbers vary by platform):

```
Algorithm      Avg Lat (ms)  Avg Energy (mJ)  Privacy Risk  SLA Viols
--------------------------------------------------------------------
LocalOnly             38.42           0.0821        0.0000   127/500
CloudOnly            142.71           0.0023        0.8800    83/500
PSO                   61.34           0.0412        0.3200    44/500
ACO                   58.19           0.0398        0.3100    41/500
HS-HHO                55.87           0.0381        0.2900    36/500
BBO-DRL               39.23           0.0301        0.2100    18/500

All schedulers completed successfully.
```

---

## Running Full Experiments

The Monte Carlo experiment runner executes all six algorithms across six task scales (100 to 10,000 tasks) with 30 independent runs per configuration. From the project root:

```bash
# Full experiment suite (~15-25 minutes on a modern laptop)
python code/src/simulation/monte_carlo.py

# Fast mode: 5 runs per configuration instead of 30
python code/src/simulation/monte_carlo.py --n_runs 5

# Custom output directory
python code/src/simulation/monte_carlo.py --output results/experiment_v2 --n_runs 10
```

Results are saved as JSON files in `results/` (or your specified output directory). Figures are generated in `latex/figures/`.

---

## Running the XAI Analysis

The XAI module trains a SHAP-explainable CI predictor on the Mendeley IoMT dataset and generates three publication-quality figures. The Mendeley dataset must be in place before running this step.

```bash
python code/src/analysis/xai_ci_module.py
```

Optional arguments:
```bash
python code/src/analysis/xai_ci_module.py \
    --data-dir data \
    --results-dir results \
    --figures-dir latex/figures
```

Outputs:
- `results/shap_feature_importance.json` — SHAP mean absolute values per feature
- `latex/figures/fig_shap_beeswarm.pdf` / `.png` — SHAP beeswarm plot
- `latex/figures/fig_shap_bar.pdf` / `.png` — Mean |SHAP| bar chart by feature group
- `latex/figures/fig_ci_distribution.pdf` / `.png` — CI score histogram with cumulative CDF

---

## Directory Structure

```
Bio-Inspired Adaptive Task Offloading System/
├── code/
│   ├── src/
│   │   ├── algorithms/         # BBO-DRL, PSO, ACO, HS-HHO, LocalOnly, CloudOnly
│   │   ├── core/               # Task, network, cost function, hardware profiles
│   │   ├── data_ingestion/     # Dataset parsers and synthetic event generator
│   │   ├── simulation/         # Monte Carlo runner, environment, metrics, topology
│   │   └── analysis/           # XAI / SHAP CI module
│   ├── run_test.py             # Quick integration test (no datasets needed)
│   ├── run_data_ingestion.py   # Ingest all four datasets into unified format
│   └── requirements.txt
├── data/                       # (not committed) — place downloaded datasets here
├── docs/
│   ├── Patent.md               # Full provisional patent text
│   └── Plan.md                 # Journal publication roadmap
├── figures/                    # Architecture diagrams (Figure 1-3)
├── latex/                      # Manuscript LaTeX source and figures
├── simulators/                 # Supplementary simulation scripts
├── submission/                 # Cover letter, complexity analysis, data statement
└── README.md
```

---

## Recommended Running Order for a Clean Experiment

1. **Install dependencies:** `pip install -r code/requirements.txt`
2. **Download datasets:** Follow Data Setup instructions above
3. **Ingest datasets:** `python code/run_data_ingestion.py` — creates `data/processed/`
4. **Run experiments:** `python code/src/simulation/monte_carlo.py --n_runs 30`
5. **Run XAI analysis:** `python code/src/analysis/xai_ci_module.py`
6. **View figures:** Open `latex/figures/` — all PDFs and PNGs for manuscript figures

---

## Citing This Work

If you use this code or build on the BBO-DRL algorithm, please cite:

**Preprint / Manuscript (upon acceptance):**
```
@article{bbodrl2026,
  title   = {BBO-DRL: A Hybrid Bombardier Beetle Optimizer with Deep Reinforcement
             Learning for Adaptive Task Offloading in IoT-Edge-Cloud Healthcare Networks},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

**Provisional Patent (system architecture):**
```
Bio-Inspired Adaptive Task Offloading System for Energy-Efficient IoT-Edge-Cloud
Healthcare Continuum. Provisional Patent Application. 2025.
```

**Bombardier Beetle Optimizer (algorithm basis):**
```
arXiv:2510.17005 (October 2025) — Bombardier Beetle Optimizer: original algorithm
description and CEC 2017 benchmark validation.
```

---

## License

MIT License. See `LICENSE` for details.

This project uses four publicly available datasets, each subject to its own access terms (PhysioNet Restricted Health Data License, CC BY 4.0, CIC Terms of Use, Kaggle Community Data License). Consult the respective repository pages before redistribution of data files.
