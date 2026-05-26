"""
run_mg1_sensitivity.py — M/G/1 queueing sensitivity analysis (parallelised).

Replaces the M/M/1 queue delay (c_v=1) with M/G/1 for c_v in {1.5, 2.5}
using the Pollaczek–Khinchine formula:
    E[W_G] = E[W_MM1] * (1 + c_v^2) / 2

Runs all algorithms at N=1000 for 30 replicates under each c_v setting.
All (cv, algorithm) pairs are executed concurrently via ProcessPoolExecutor,
giving roughly cpu_count-fold speedup over the serial version.

Saves results/mg1_sensitivity.json and results/table_mg1.csv.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

from src.config import GLOBAL_SEED, N_FOG_NODES, N_RUNS, N_WEARABLES, get_full_algorithm_registry
from src.simulation.topology import build_healthcare_topology

N_TASKS  = 1000
CV_VALUES = [1.0, 1.5, 2.5]
N_RUNS_CV = N_RUNS   # 30
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# ---------------------------------------------------------------------------
# Top-level worker — must be importable (no closures) for multiprocessing
# ---------------------------------------------------------------------------

def _run_cv_alg_worker(args: tuple):
    """Run N_RUNS_CV replicates for one (cv, alg_name) pair.

    Builds its own topology and imports so each subprocess is self-contained.
    Returns (cv, alg_name, list[dict]).
    """
    cv, alg_name, n_tasks, n_runs, seed_base, base_path = args

    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    import random as _r
    import numpy as _np
    import warnings as _w

    from src.config import N_FOG_NODES as _NF, N_WEARABLES as _NW, get_full_algorithm_registry
    from src.core.task import HealthcareTask
    from src.data_ingestion.event_generator import generate_synthetic_tasks
    from src.simulation.environment import OffloadingEnvironment
    from src.simulation.topology import build_healthcare_topology

    def _ht(t, topo):
        wids = [nid for nid, n in topo.nodes.items() if n.node_type == 'wearable']
        return HealthcareTask(
            task_id=t.task_id, device_id=wids[t.device_id % len(wids)],
            timestamp=t.timestamp, data_size_bits=t.data_size_bits,
            cpu_cycles=t.cpu_cycles, max_delay_s=t.max_delay_s,
            privacy_sensitivity=t.privacy_sensitivity, ci_score=t.ci_score,
            attack_probability=t.attack_probability, source=t.source,
        )

    def _accepts_seed(cls):
        import inspect
        try:
            return 'seed' in inspect.signature(cls.__init__).parameters
        except Exception:
            return False

    topo = build_healthcare_topology(n_wearables=_NW, n_fog_nodes=_NF, seed=seed_base)
    sched_cls = get_full_algorithm_registry()[alg_name]

    q_frac = 0.25
    scale  = (1.0 + cv ** 2) / 2.0  # cv=1→1.0, cv=1.5→1.625, cv=2.5→3.625

    runs = []
    for run_id in range(n_runs):
        try:
            seed = seed_base + run_id * 1000 + n_tasks
            _r.seed(seed)
            _np.random.seed(seed)

            sim_tasks = generate_synthetic_tasks(n_tasks, ci_distribution='mixed', seed=seed)
            tasks = [_ht(t, topo) for t in sim_tasks]

            sched = sched_cls(topo, seed=seed) if _accepts_seed(sched_cls) else sched_cls(topo)
            env   = OffloadingEnvironment(topo, sched, n_tasks=n_tasks, seed=seed)
            res   = env.run(tasks)

            if not res:
                continue

            lats = []
            for r in res:
                lat = r['latency_ms']
                if r.get('node_type', '') != 'local':
                    lat = lat * (1 - q_frac) + lat * q_frac * scale
                lats.append(lat)

            dl   = [t.max_delay_s * 1000.0 for t in tasks]
            sla  = 100.0 * sum(1 for l, d in zip(lats, dl) if l > d) / len(res)

            runs.append({
                'avg_latency_ms':    float(_np.mean(lats)),
                'avg_energy_mj':     float(_np.mean([r['energy_mj']   for r in res])),
                'avg_privacy_risk':  float(_np.mean([r['privacy_risk'] for r in res])),
                'sla_violation_pct': sla,
            })
        except Exception as exc:
            _w.warn(f'[M/G/1] {alg_name} cv={cv} run={run_id}: {exc}')

    return cv, alg_name, runs


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _agg(runs: list[dict], key: str) -> dict:
    vals = np.array([r[key] for r in runs], dtype=float)
    return {'mean': float(vals.mean()), 'std': float(vals.std()),
            'min': float(vals.min()),   'max': float(vals.max()),
            'n': int(len(vals))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    registry    = get_full_algorithm_registry()
    algs_to_test = [k for k in registry if k not in ('Local-Only', 'Cloud-Only')]

    print(f'[M/G/1] {len(CV_VALUES)} cv values × {len(algs_to_test)} algorithms × '
          f'{N_RUNS_CV} runs = {len(CV_VALUES)*len(algs_to_test)*N_RUNS_CV} total simulations')
    print(f'[M/G/1] Workers: {N_WORKERS}  (parallelising over cv×algorithm pairs)')
    print(f'[M/G/1] Algorithms: {algs_to_test}', flush=True)

    job_args = [
        (cv, alg, N_TASKS, N_RUNS_CV, GLOBAL_SEED, BASE)
        for cv  in CV_VALUES
        for alg in algs_to_test
    ]

    raw: dict[float, dict[str, list]] = {cv: {} for cv in CV_VALUES}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_cv_alg_worker, a): a for a in job_args}
        for fut in as_completed(futures):
            try:
                cv, alg_name, runs = fut.result()
            except Exception as exc:
                _, alg_name, *_ = futures[fut]
                print(f'  [ERROR] {alg_name}: {exc}', flush=True)
                continue
            raw[cv][alg_name] = runs
            if runs:
                m = _agg(runs, 'avg_latency_ms')
                p = _agg(runs, 'avg_privacy_risk')
                s = _agg(runs, 'sla_violation_pct')
                print(f'  [cv={cv:.1f}] {alg_name:<10}  '
                      f'lat={m["mean"]:.2f}±{m["std"]:.2f}  '
                      f'priv={p["mean"]:.4f}  sla={s["mean"]:.2f}%  '
                      f'({len(runs)} runs)', flush=True)
            else:
                print(f'  [cv={cv:.1f}] {alg_name:<10}  NO RESULTS', flush=True)

    # Aggregate
    results: dict[float, dict] = {cv: {} for cv in CV_VALUES}
    for cv in CV_VALUES:
        for alg in algs_to_test:
            runs = raw[cv].get(alg, [])
            if not runs:
                results[cv][alg] = {}
                continue
            results[cv][alg] = {
                'avg_latency_ms':    _agg(runs, 'avg_latency_ms'),
                'avg_energy_mj':     _agg(runs, 'avg_energy_mj'),
                'avg_privacy_risk':  _agg(runs, 'avg_privacy_risk'),
                'sla_violation_pct': _agg(runs, 'sla_violation_pct'),
            }

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'mg1_sensitivity.json'
    with open(out_path, 'w') as fh:
        json.dump({str(cv): results[cv] for cv in CV_VALUES}, fh, indent=2)
    print(f'\n[SAVE] {out_path}')

    # Save CSV
    csv_path = RESULTS_DIR / 'table_mg1.csv'
    header = (['Algorithm']
              + [f'cv={cv} lat±std' for cv in CV_VALUES]
              + [f'cv={cv} priv'         for cv in CV_VALUES]
              + [f'cv={cv} SLA%'         for cv in CV_VALUES])
    rows = [header]
    for alg in algs_to_test:
        row = [alg]
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            lat = d.get('avg_latency_ms', {})
            row.append(f'{lat.get("mean",0):.2f}±{lat.get("std",0):.2f}')
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            row.append(f'{d.get("avg_privacy_risk",{}).get("mean",0):.4f}')
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            row.append(f'{d.get("sla_violation_pct",{}).get("mean",0):.2f}')
        rows.append(row)

    with open(csv_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f'[SAVE] {csv_path}')

    # Ordering check
    print('\n[ORDERING CHECK] Privacy risk ordering under M/G/1:')
    for cv in CV_VALUES:
        privs = {alg: results[cv].get(alg, {}).get('avg_privacy_risk', {}).get('mean', 999)
                 for alg in algs_to_test}
        ordered = sorted(privs, key=privs.get)
        print(f'  cv={cv}: {" < ".join(f"{a}={privs[a]:.4f}" for a in ordered)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
