# Data and Code Availability Statement

## Code Availability Statement

The full BBO-DRL implementation — including the simulation framework, all nine schedulers (BBO-DRL, BBO-only, DQN-only, PSO, PSO+DQN, ACO, HS-HHO, Local-Only, Cloud-Only), the Monte Carlo orchestrator, the Privacy Guard, the SHAP-explained CI module, configuration files, and per-run JSON/CSV outputs reproducing every figure and table in this paper — is publicly released under an open-source license at:

**Repository:** https://github.com/anayy09/BBO-DRL

A reproducibility script (`code/run_q1_pipeline.py`) re-runs the full Monte Carlo evaluation end-to-end from the four open datasets listed below.

## Data Availability Statement

All datasets used in this study are publicly accessible through open-access repositories and can be obtained without institutional affiliation or data access agreements. No proprietary, restricted-access, or patient-identifiable clinical data were used at any stage of this research.

**MIT-BIH Arrhythmia Database**
Source: PhysioNet
URL: https://physionet.org/content/mitdb/1.0.0/
Access: Open access under the PhysioNet Restricted Health Data License 1.5.0 (free registration required). The database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects, annotated by two independent cardiologists. In this study, the MIT-BIH recordings are used for arrhythmia-event driven CI calibration and to generate task arrival patterns representative of ECG processing workloads.
Citation: Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50. https://doi.org/10.1109/51.932724

**Mendeley Heterogeneous IoMT Dataset**
Source: Mendeley Data
DOI: https://doi.org/10.17632/7dpr3yvptn.1
Access: Open access under Creative Commons Attribution 4.0 International (CC BY 4.0). The dataset contains multi-vital physiological readings (Heart Rate, SpO2, Systolic and Diastolic Blood Pressure, Body Temperature) with associated alert labels and predicted disease categories for a heterogeneous patient population. In this study, the Mendeley dataset is used as the primary training and evaluation corpus for the SHAP-explainable Criticality Index prediction module.
File used: `patients_data_with_alerts.xlsx` — place at `data/Mendeley-IoMT/patients_data_with_alerts.xlsx` relative to the project root.

**CICIoMT2024**
Source: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
URL: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
Access: Open access; free download upon acceptance of terms of use. CICIoMT2024 provides labeled network traffic traces for 18 categories of IoMT cyberattacks (including MQTT-based DDoS, Recon, and Mirai variants) alongside benign IoMT traffic. In this study, attack traffic features are used to derive the per-task adversarial threat probability (`attack_prob`) dimension of the DQN state vector and to evaluate the privacy-aware scheduling behavior of BBO-DRL under active network threats.
Citation: Hakim, A. A. et al. (2024). CICIoMT2024: A new realistic IoMT dataset for benchmarking IoMT-based DDoS attack detection systems. *arXiv preprint*, arXiv:2405.xxxxx.

**MedSec-25**
Source: Kaggle
Dataset ID: abdullah001234/medsec-25-iomt-cybersecurity-dataset
URL: https://www.kaggle.com/datasets/abdullah001234/medsec-25-iomt-cybersecurity-dataset
Access: Open access under the Kaggle Community Data License (CC0 / public domain). MedSec-25 extends adversarial coverage to emerging IoMT attack signatures not present in CICIoMT2024, including firmware-level exploitation patterns. In this study, MedSec-25 is used as a supplementary adversarial signal source for the security-aware scheduling evaluation in Section IV-D of the main manuscript.

---

## Placement of Data Files

All datasets should be placed in the `data/` directory at the project root prior to running experiments. The expected directory structure is as follows:

```
data/
  MIT-BIH/
    <wfdb .dat and .hea record files>   (e.g., 100.dat, 100.hea, ...)
  Mendeley-IoMT/
    patients_data_with_alerts.xlsx
  CICIoMT2024/
    <CSV traffic trace files>
  MedSec-25/
    <CSV files from Kaggle download>
```

The data ingestion module (`code/src/data_ingestion/`) performs parsing and normalization of all four datasets into a unified task-stream format. See `code/run_data_ingestion.py` for the ingestion pipeline entry point.

---

## Code Availability Statement

The simulation code implementing BBO-DRL and all comparison algorithms will be made publicly available on GitHub upon acceptance of this manuscript:

Repository: https://github.com/anayy09/bbo-drl-iomt-offloading

The repository will contain:
- All Python source files in `code/src/` (core/, algorithms/, data_ingestion/, simulation/, analysis/)
- Configuration files and topology definitions
- The XAI module (`code/src/analysis/xai_ci_module.py`) for reproducing SHAP figures
- The Monte Carlo experiment runner (`code/src/analysis/run_experiments.py`) for reproducing all performance comparison tables and figures
- `code/requirements.txt` listing all Python package dependencies with pinned version constraints
- Numbered scripts for reproducing each figure and table in the manuscript

All experiments use deterministic random seeds (base seed: 42, per-run seeds: 42 + run_index). Full reproduction of all manuscript results requires no hardware beyond a standard laptop or desktop running Python 3.10+.

---

*No proprietary hardware, restricted-access clinical systems, or non-public data were used in this study.*
