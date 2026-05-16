# BBO-DRL: Bio-Inspired Adaptive Task Offloading for IoT-Edge-Cloud Healthcare Networks

A hybrid Bombardier Beetle Optimizer / Deep Q-Network scheduling system for privacy-aware, Criticality-Index-adaptive task offloading in IoMT deployments. Submitted to IEEE Journal of Biomedical and Health Informatics.

---

## Repository Layout

```
Bio-Inspired Adaptive Task Offloading System/
├── code/
│   ├── src/
│   │   ├── algorithms/         # All 9 schedulers (see below)
│   │   ├── analysis/           # Monte Carlo runner, ablations, figures, XAI
│   │   ├── core/               # Task, cost function, hardware profiles, network model
│   │   ├── data_ingestion/     # Dataset parsers and synthetic event generator
│   │   ├── simulation/         # Event-driven environment, topology, metrics
│   │   └── config.py           # Global hyperparameters and algorithm registry
│   ├── run_q1_pipeline.py      # Main orchestrator — runs all experiment steps
│   ├── regen_figures.py        # Regenerate publication figures from saved results
│   ├── run_data_ingestion.py   # Ingest all four datasets into unified format
│   ├── run_test.py             # Quick integration test (no datasets needed)
│   └── requirements.txt
├── configs/
│   └── simulation_config.yaml  # Hardware parameters, topology, MC protocol
├── data/                       # (not committed) — place downloaded datasets here
├── docs/
│   ├── Patent.md               # Full provisional patent text
│   ├── Plan.md                 # Journal publication roadmap
│   ├── Fix.md                  # Round-1 revision log (implemented)
│   └── Fix2.md                 # Round-2 revision log (in progress)
├── figures/                    # Architecture diagrams referenced in patent
├── latex/
│   ├── manuscript.tex          # J-BHI submission source
│   └── figures/                # Publication-quality PDFs and PNGs (fig1–fig10)
├── results/                    # JSON and CSV outputs from completed runs
├── submission/
│   ├── cover_letter.md
│   ├── complexity_analysis.md
│   ├── data_availability_statement.md
│   └── reviewer_anticipation.md
└── README.md
```

---

## Schedulers

| Name | File | Description |
|------|------|-------------|
| **BBO-DRL** | `bbo_drl.py` | Proposed method: DQN top-K pre-filter + BBO inner search |
| **PSO+DQN** | `pso_dqn.py` | Ablation hybrid: same DQN architecture, PSO replaces BBO inner search |
| **BBO-only** | `bbo_only.py` | BBO over full candidate set, no DQN pre-filter |
| **DQN-only** | `dqn_only.py` | DQN argmax, no BBO refinement |
| PSO | `pso.py` | Per-task particle swarm, stateless |
| ACO | `aco.py` | Pheromone-reinforcement ant colony |
| HS-HHO | `hs_hho.py` | Slime Mould + Harris Hawks hybrid (strongest published comparator) |
| Local-Only | `local_only.py` | All tasks on originating wearable |
| Cloud-Only | `cloud_only.py` | All tasks forwarded to cloud server |

The DQN is implemented in pure NumPy (no PyTorch / TensorFlow dependency).

---

## Dependencies

Python 3.10 or later required.

```bash
pip install -r code/requirements.txt
```

Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `wfdb`, `openpyxl`, `tqdm`, `pyyaml`.
Required for XAI and Privacy Guard: `scikit-learn`, `shap`.

---

## Data Setup

Four publicly available datasets are required. Download and place them under `data/`:

```
data/
  MIT-BIH/
      100.dat  100.hea  ...  (48 records)
      PhysioNet: https://physionet.org/content/mitdb/1.0.0/
      Quick download: python -c "import wfdb; wfdb.dl_database('mitdb', 'data/MIT-BIH/')"

  Mendeley-IoMT/
      patients_data_with_alerts.xlsx
      DOI: https://doi.org/10.17632/7dpr3yvptn.1

  CICIoMT2024/
      <CSV traffic trace files>
      https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

  MedSec-25/
      <CSV files>
      https://www.kaggle.com/datasets/abdullah001234/medsec-25-iomt-cybersecurity-dataset
```

Then ingest all four datasets into the unified internal format:

```bash
python code/run_data_ingestion.py
```

This creates `data/processed/` with normalised task streams and labelled network flows.

---

## Quick Sanity Check

Runs all nine schedulers on 500 synthetic tasks without any dataset download:

```bash
python code/run_test.py
```

Expected output confirms that all schedulers complete without error and that BBO-DRL leads on privacy risk while PSO/HS-HHO lead on latency (exact numbers vary by platform and seed).

---

## Running the Full Q1 Experiment Pipeline

The main orchestrator is `code/run_q1_pipeline.py`. It runs all experiment steps in sequence, each of which can be skipped independently.

```bash
# Full pipeline — all steps, 30 runs, 5 task scales (N = 100, 500, 1000, 2000, 5000)
# Warning: this takes several hours on a standard laptop
python code/run_q1_pipeline.py --n_runs 30

# Run individual steps
python code/run_q1_pipeline.py --n_runs 30 --skip-stats --skip-weight --skip-privacy \
    --skip-mitbih --skip-figures --skip-highci --skip-overhead --skip-decomp --skip-routing
# (runs Step 1 — Monte Carlo only)

python code/run_q1_pipeline.py --n_runs 30 --skip-mc --skip-weight --skip-privacy \
    --skip-mitbih --skip-figures --skip-highci --skip-overhead --skip-decomp --skip-routing
# (runs Step 2 — Wilcoxon + Bonferroni tests only)
```

### Pipeline steps and their deliverables

| Flag | Step | Deliverable |
|------|------|-------------|
| `--skip-mc` | Monte Carlo: all 9 schedulers × 5 scales × 30 runs | `results/mc_full_summary.json` |
| `--skip-stats` | Wilcoxon rank-sum + Bonferroni (24 tests) | `results/table3_n1000_with_pvals.csv` |
| `--skip-weight` | CI weight ablation, mixed-CI workload | `results/table5_weight_ablation.csv` |
| `--skip-privacy` | Privacy Guard on MedSec-25 (sklearn AUC) | `results/privacy_guard_metrics.json` |
| `--skip-mitbih` | MIT-BIH real-trace evaluation (8,640 windows) | `results/table5_mitbih_trace.csv` |
| `--skip-figures` | All 10 publication figures | `latex/figures/fig*.pdf` |
| `--skip-highci` | All-high-CI weight ablation (ICU scenario) | `results/table6_highci_weights.csv` |
| `--skip-overhead` | Per-task scheduling wall-clock timing | `results/scheduling_overhead_summary.csv` |
| `--skip-decomp` | Latency decomposition tx/queue/compute | `results/latency_decomposition.csv` |
| `--skip-routing` | DQN-only routing distribution by quintile | `results/dqn_only_routing_summary.json` |

A framing note (`results/framing_note.txt`) comparing BBO-DRL vs PSO+DQN on privacy is generated automatically at the end of every run.

### Parallel processing

All multi-run steps use `multiprocessing.Pool` with a `spawn` context (Windows-safe). The worker count defaults to `cpu_count - 1`; override with `--workers N`.

---

## Regenerating Figures from Saved Results

If you have existing result files and only need to regenerate figures:

```bash
python code/run_q1_pipeline.py --n_runs 30 --skip-mc --skip-stats --skip-weight \
    --skip-privacy --skip-mitbih --skip-highci --skip-overhead --skip-decomp --skip-routing
```

Or directly:

```bash
python code/regen_figures.py
```

---

## Publication Figures

All figures are in `latex/figures/` as PDF + PNG pairs.

| File | Figure | Description |
|------|--------|-------------|
| `fig1_latency_vs_scale` | Fig. 1 | Mean latency vs. task scale (N = 100–5,000) |
| `fig2_energy_sla_vs_scale` | Fig. 2 | Energy per task and SLA violation rate vs. scale |
| `fig3_metric_bars` | Fig. 3 | Per-metric bar comparison at N = 1,000 |
| `fig4_pareto_energy_latency` | Fig. 4 | Pareto frontier: energy vs. latency |
| `fig5_pareto_latency_privacy` | Fig. 5 | Pareto frontier: latency vs. privacy risk |
| `fig6_epsilon_convergence` | Fig. 6 | ε-decay trajectory (30-run mean ± 1 SD) |
| `fig7_weight_ablation` | Fig. 7 | CI weight scheme ablation (mixed-CI panel; all-high-CI panel pending) |
| `fig8_privacy_guard_roc` | Fig. 8 | Privacy Guard ROC curve on MedSec-25 |
| `fig9_shap_summary` | Fig. 9 | Mean |SHAP| values for the CI module |
| `fig10_mitbih_trace` | Fig. 10 | MIT-BIH real-trace results |

---

## Completed Result Files

The following result files are committed and reproducible:

| File | Contents |
|------|----------|
| `results/mc_full_summary.json` | Per-scale per-algorithm statistics from 30-run MC |
| `results/mc_full_results.json` | Raw per-run outputs |
| `results/table3_n1000.csv` | Headline comparison table at N = 1,000 |
| `results/table3_n1000_with_pvals.csv` | Table III with Bonferroni-corrected p-values |
| `results/table5_weight_ablation.csv` | CI weight scheme ablation (mixed-CI) |
| `results/table5_mitbih_trace.csv` | MIT-BIH real-trace comparison |
| `results/privacy_guard_metrics.json` | AUC, TPR, FPR, F1 at τ = 0.55 |
| `results/shap_feature_importance.json` | SHAP mean |value| per physiological feature |
| `results/epsilon_trajectory.json` | ε-decay trajectory across 30 runs |
| `results/mitbih_trace_raw.json` | Raw per-run MIT-BIH results |

Pending (require new runs): `table6_highci_weights.csv`, `scheduling_overhead_summary.csv`, `latency_decomposition.csv`, `dqn_only_routing_summary.json`.

---

## Recommended Execution Order

1. `pip install -r code/requirements.txt`
2. Download datasets and run `python code/run_data_ingestion.py`
3. `python code/run_test.py` — verify all schedulers work
4. `python code/run_q1_pipeline.py --n_runs 30` — full pipeline (~4–6 hours)
5. Check `results/framing_note.txt` for PSO+DQN vs BBO-DRL assessment before finalising manuscript claims

---

## Citing This Work

**Manuscript (under review):**
```bibtex
@article{bbodrl2026,
  title   = {BBO-DRL: A Hybrid Bombardier Beetle Optimizer with Deep Reinforcement
             Learning for Adaptive Task Offloading in IoT-Edge-Cloud Healthcare Networks},
  author  = {Sinha, Anay},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

**Bombardier Beetle Optimizer:**
```
arXiv:2510.17005 (Oct 2025) — original algorithm and CEC 2017 benchmark results
```

**Provisional patent (system architecture):**
```
Bio-Inspired Adaptive Task Offloading System for Energy-Efficient IoT-Edge-Cloud
Healthcare Continuum. U.S. Provisional Patent Application, 2024.
```

---

## License

MIT License. See `LICENSE` for details.

The four datasets used in this work are each subject to their own access terms: PhysioNet Restricted Health Data License (MIT-BIH), CC BY 4.0 (Mendeley IoMT), CIC Terms of Use (CICIoMT2024), and Kaggle Community Data License (MedSec-25). Do not redistribute dataset files.
