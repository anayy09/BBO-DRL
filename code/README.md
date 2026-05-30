# DQN-ES Simulation Framework — Code Directory

Technical reference for the simulation codebase. Covers architecture, module descriptions, design decisions, reproducibility controls, and extension guides.

---

## Framework Architecture

```
IoMT Event Stream
       |
       v
+------------------+
|  data_ingestion/ |  <-- Parses MIT-BIH, Mendeley, CICIoMT2024, MedSec-25
|  event_generator |      Produces HealthcareTask stream (unified format)
+------------------+
       |
       v
+------------------+        +------------------+
|  simulation/     |        |  core/           |
|  topology.py     |------->|  task.py         |  HealthcareTask dataclass
|  environment.py  |        |  network.py      |  NetworkTopology, Node, Link
|  monte_carlo.py  |        |  cost_function.py|  F(x), weight functions
|  metrics.py      |        |  hardware_profil |  RPi4, ESP32, Fog, Cloud specs
+------------------+        +------------------+
       |
       |  calls select_node() per task
       v
+------------------+
|  algorithms/     |
|  bbo_drl.py      |  <-- PRIMARY: DQN + BBO hybrid (research contribution)
|  pso.py          |  <-- Baseline: Particle Swarm Optimization
|  aco.py          |  <-- Baseline: Ant Colony Optimization
|  hs_hho.py       |  <-- Baseline: Hybrid Slime Mold + Harris Hawks
|  local_only.py   |  <-- Lower bound: always process on wearable
|  cloud_only.py   |  <-- Upper latency bound: always offload to cloud
|  base_scheduler  |  <-- Abstract base: evaluate_node(), record_decision()
+------------------+
       |
       v
+------------------+
|  analysis/       |
|  xai_ci_module   |  <-- SHAP TreeExplainer on RF CI predictor (Mendeley)
+------------------+
       |
       v
  results/  +  latex/figures/   (JSON metrics, PDF/PNG figures)
```

---

## Module Descriptions

### `src/core/`

**`task.py`** — Defines `HealthcareTask`, the central data object passed through the entire pipeline. Fields: `task_id`, `device_id`, `timestamp`, `data_size_bits`, `cpu_cycles`, `max_delay_s`, `privacy_sensitivity`, `ci_score`, `attack_probability`, `source` (dataset tag). All schedulers receive a `HealthcareTask` and return a `node_id`.

**`network.py`** — `NetworkNode` (CPU frequency, current load, node type), `NetworkLink` (bandwidth, propagation delay), `NetworkTopology` (graph with `get_uplink_rate()` and `get_propagation_delay()` methods). Topology is constructed once and shared across all schedulers within a Monte Carlo run.

**`cost_function.py`** — Implements the full multi-objective cost function F(x) = ŵ_E·Ê + ŵ_L·L̂ + ŵ_P·R_P. Contains the CI-adaptive weight functions (exponential decay for energy, exponential growth for latency, power-law decay for privacy), the M/M/1 queuing-based latency model, the CMOS dynamic energy model (E = κ·C·f²), and the Shannon entropy privacy risk function. All weight parameters (ALPHA_E=3.0, BETA_L=4.0, GAMMA_P=2.0) are defined as module-level constants.

**`hardware_profiles.py`** — Physical device specifications as named dictionaries: `WEARABLE_ESP32` (240 MHz, 10 mW TX power, 2 mW idle), `EDGE_GATEWAY_RPI4` (1.5 GHz Cortex-A72), `FOG_NODE` (2.5 GHz x86-class), `CLOUD_SERVER` (3.2 GHz, effectively unlimited queue capacity). These profiles are used by `topology.py` to initialize node parameters.

### `src/algorithms/`

**`base_scheduler.py`** — Abstract base class `BaseScheduler`. Subclasses must implement `select_node(task: HealthcareTask) -> int`. The base class provides `evaluate_node()` (computes latency, energy, privacy risk for a candidate node using `cost_function.py`), `estimate_feasible_bounds()` (scans all candidate nodes to derive min-max normalization bounds), and `record_decision()` (updates the offload history for privacy entropy computation). All schedulers share identical `evaluate_node()` logic, ensuring fair comparison.

**`bbo_drl.py`** — The primary research contribution. Contains three classes: `ReplayBuffer` (circular deque with `push`/`sample`), `DQNNetwork` (pure NumPy two-hidden-layer network with Xavier initialization, ReLU activations, gradient clipping at norm=5.0), and `BBOOptimizer` (antenna-based continuous optimizer with quadratic antenna-length decay). `BBODRLScheduler` composes these into the full hybrid: DQN narrows to top-K candidates via ε-greedy selection, BBO refines over the K-node continuous subspace, experience is stored in replay, and the DQN policy is updated online via Bellman targets from the target network.

**`pso.py`** — Particle Swarm Optimization. 30 particles, 50 iterations, inertia weight 0.7, cognitive/social constants 1.5. Re-initialized from scratch per task (no inter-task memory). Candidate positions are real-valued and mapped to the nearest discrete node index.

**`aco.py`** — Ant Colony Optimization. 20 ants, 30 iterations, pheromone evaporation rate 0.3, α=1, β=2. Pheromone is reset between tasks to prevent persistent bias across a Monte Carlo run.

**`hs_hho.py`** — Hybrid Slime Mold + Harris Hawks heuristic. 30 agents, 50 iterations. Combines the slime mold's oscillatory exploration with Harris Hawks' exploitation dive phases. Tuned to match the HS-HHO configuration from the survey literature.

**`local_only.py`** and **`cloud_only.py`** — Trivial schedulers that always return the wearable node ID and the cloud node ID, respectively. These serve as lower and upper performance bounds.

### `src/data_ingestion/`

**`parse_mitbih.py`** — Uses the `wfdb` library to read WFDB-format records from `data/MIT-BIH/`. Extracts arrhythmia event timestamps and annotation labels. Converts annotated episodes to high-CI task arrival bursts calibrated to 13.8% episode prevalence.

**`parse_mendeley.py`** — Reads `patients_data_with_alerts.xlsx` using `openpyxl`. Resolves column names via case-insensitive substring matching to handle encoding variants. Builds the CI target vector from alert labels (`High/Low/Abnormal/Normal`) and disease-boost scores. Outputs a DataFrame suitable for XAI training.

**`parse_ciciot.py`** — Reads CICIoMT2024 CSV files. Extracts per-flow attack probability features (attack label prevalence, flow entropy) used to populate the `attack_probability` field of `HealthcareTask`.

**`parse_medsec.py`** — Reads MedSec-25 CSV files. Maps attack categories to binary threat flags and derives per-device attack probability estimates consistent with CICIoMT2024 format.

**`event_generator.py`** — `generate_synthetic_tasks()` and `generate_event_stream()` produce task streams without requiring any dataset. Uses configurable distributions for payload size (log-normal, mean 50 KB), CPU cycles (exponential, mean 10M cycles), CI score (beta distribution, α=0.5, β=2.0 to model predominantly low-CI baseline with rare high-CI spikes), and adversarial threat probability (uniform 0–0.3). Used by `run_test.py` and as a fallback when dataset files are absent.

### `src/simulation/`

**`topology.py`** — `build_healthcare_topology(n_wearables, n_fog_nodes, seed)` constructs the four-tier simulation network. Node placement: wearables within 50 m of edge gateway (origin), fog nodes at 1–5 km, cloud at 50 km (WAN abstraction). All inter-node link parameters are derived from the hardware profiles in `core/hardware_profiles.py`.

**`environment.py`** — `OffloadingEnvironment` wraps a topology and a scheduler. The `run(tasks)` method iterates over the task list, calls `scheduler.select_node(task)` for each, records the returned node ID, computes realized latency/energy/privacy via `base_scheduler.evaluate_node()`, and returns a list of per-task result dictionaries.

**`metrics.py`** — `compute_metrics()` aggregates a run's result list into a `SimulationMetrics` dataclass (mean latency, mean energy, SLA violation rate, mean privacy risk). `aggregate_mc_runs()` computes mean ± std across 30 runs. `compare_algorithms()` builds the comparison table used in manuscript Tables III and IV.

**`monte_carlo.py`** — Top-level experiment orchestrator. Iterates over 6 task scales × 6 algorithms × 30 runs, collects metrics, and saves JSON output. Supports `--n_runs`, `--output`, `--fast` flags. Can be run as a module: `python -m src.simulation.monte_carlo`.

### `src/analysis/`

**`xai_ci_module.py`** — Trains a 100-tree Random Forest regressor on the Mendeley IoMT dataset to predict CI scores from five vital-sign features (Heart Rate, SpO2, Systolic BP, Diastolic BP, Body Temperature). Applies SHAP `TreeExplainer` if the `shap` package is installed; falls back to scikit-learn permutation importance otherwise. Generates three IEEE-format figures: SHAP beeswarm plot, mean |SHAP| bar chart (color-coded by feature group), and CI score distribution histogram with cumulative CDF overlay.

---

## Key Design Decisions

**Why NumPy DQN instead of PyTorch?** The target deployment environment is an edge gateway (e.g., Raspberry Pi 4) where installing PyTorch with CUDA is impractical and CPU-only PyTorch adds ~200 MB to the dependency footprint. The DQN is small enough (5,583 parameters) that hand-written NumPy backpropagation with gradient clipping is both correct and sufficient. Replacing it with a PyTorch equivalent is straightforward — the `DQNNetwork.forward()` and `DQNNetwork.update()` interfaces are drop-in replaceable.

**Why M/M/1 queuing?** The M/M/1 sojourn time formula W = 1/(μ − λ) provides a closed-form queuing delay estimate directly usable in the real-time cost function without simulation-within-simulation. M/G/1 or G/G/1 models are more accurate for bursty IoMT traffic but require numerical inversion at each scheduling call. The M/M/1 approximation is consistent across all six algorithms, preserving relative comparisons. A configuration flag (`use_mmg1=True`) enables the Pollaczek-Khinchine M/G/1 formula as a sensitivity-analysis alternative.

**Why SHAP over LIME?** The CI predictor is a Random Forest — a tree ensemble for which `TreeExplainer` provides exact Shapley values in polynomial time via the TreeSHAP algorithm (Lundberg et al., 2020). LIME approximates locally around a query point using a linear surrogate and is less stable for high-variance features (e.g., SpO2 spikes). SHAP's consistency guarantee (changing a feature's contribution always changes its SHAP value in the same direction) is important for clinical interpretability claims.

---

## Reproducibility

All randomness in the simulation is controlled by integer seeds:

| Component | Seed parameter | Default |
|---|---|---|
| NumPy global seed | `np.random.seed(seed)` called in `environment.py` | 42 |
| DQN weight initialization | `DQNNetwork(seed=seed)` | 42 |
| BBO population initialization | `BBOOptimizer(seed=seed)` | 42 |
| Monte Carlo run seeds | `SEED_BASE + run_index` | 42, 43, 44, ... |
| Train/test split in XAI | `train_test_split(random_state=42)` | 42 |
| Random Forest training | `RandomForestRegressor(random_state=42)` | 42 |
| Synthetic task generation | `generate_synthetic_tasks(seed=seed)` | 42 |

To reproduce a specific Monte Carlo run exactly, pass `seed=SEED_BASE + run_index` to both the task generator and the scheduler constructor. The Monte Carlo runner does this automatically.

To change the global seed for a sensitivity analysis:
```python
from src.simulation.monte_carlo import run_all_experiments
run_all_experiments(seed_base=123, n_runs=30)
```

---

## Running Order for a Clean Experiment

```bash
# Step 1: Install dependencies
pip install -r requirements.txt
pip install shap scikit-learn  # for XAI module (optional but recommended)

# Step 2: Ingest datasets (requires data/ directory populated)
python run_data_ingestion.py

# Step 3: Run full Monte Carlo experiments
python -m src.simulation.monte_carlo --n_runs 30

# Step 4: Run XAI analysis (requires Mendeley dataset)
python src/analysis/xai_ci_module.py

# Step 5: View figures
# All PDFs and PNGs are written to latex/figures/
# JSON metrics are written to results/
```

All five steps must be run from the `code/` directory, or from the project root with `code/` on sys.path. The `run_test.py` script sets up sys.path automatically and can always be run from the project root.

---

## Testing: Expected Output of `run_test.py`

Run from the project root:
```bash
python code/run_test.py
```

The script exercises all six schedulers on 500 synthetic tasks (seed=42) and prints a comparison table. It should complete without errors in under 30 seconds. The approximate expected output is:

```
Algorithm      Avg Lat (ms)  Avg Energy (mJ)  Privacy Risk  SLA Viols
--------------------------------------------------------------------
LocalOnly             ~38           ~0.082         0.0000   ~127/500
CloudOnly            ~143           ~0.002         0.8800    ~83/500
PSO                   ~61           ~0.041         ~0.320    ~44/500
ACO                   ~58           ~0.040         ~0.310    ~41/500
HS-HHO                ~56           ~0.038         ~0.290    ~36/500
DQN-ES                ~39           ~0.030         ~0.210    ~18/500

All schedulers completed successfully.
```

Exact values depend on NumPy version and platform due to floating-point differences in matrix operations. The relative ranking (DQN-ES achieves lowest SLA violations and lowest average latency among the metaheuristic schedulers) should be stable across platforms.

---

## How to Extend: Adding a New Algorithm

1. Create `src/algorithms/my_algorithm.py`. Import and subclass `BaseScheduler`:
   ```python
   from src.algorithms.base_scheduler import BaseScheduler
   from src.core.task import HealthcareTask

   class MyAlgorithmScheduler(BaseScheduler):
       def select_node(self, task: HealthcareTask) -> int:
           lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
           # Your optimization loop here
           # Use self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
           #   which returns (cost, latency_s, energy_j, privacy_risk)
           # self._candidate_nodes is the list of valid node IDs
           best_node = self._candidate_nodes[0]  # replace with your selection
           self.record_decision(task.device_id, best_node)
           return best_node
   ```

2. Add the algorithm to the scheduler factory in `src/simulation/monte_carlo.py`:
   ```python
   from src.algorithms.my_algorithm import MyAlgorithmScheduler
   ALGORITHM_CLASSES['MyAlg'] = MyAlgorithmScheduler
   ```

3. Run `python code/run_test.py` to verify no import errors or runtime exceptions.

4. Run `python -m src.simulation.monte_carlo --n_runs 5` to include the new algorithm in a quick comparison run.

---

## How to Extend: Adding a New Dataset Parser

1. Create `src/data_ingestion/parse_mydataset.py`. Implement a function with signature:
   ```python
   def load_tasks(data_dir: str) -> list[HealthcareTask]:
       """Load dataset and return a list of HealthcareTask objects."""
       ...
   ```
   Populate all `HealthcareTask` fields. For `ci_score`, derive from available labels or use `0.1` as a default baseline. For `attack_probability`, use `0.0` if the dataset has no adversarial content.

2. Register the parser in `run_data_ingestion.py`:
   ```python
   from src.data_ingestion.parse_mydataset import load_tasks as load_mydataset
   tasks_my = load_mydataset(str(DATA_DIR))
   all_tasks.extend(tasks_my)
   ```

3. Document the expected file path in `README.md` under the Data Setup section.
