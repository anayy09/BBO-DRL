"""
weight_ablation.py — Fix 6: CI weight function ablation.

Compares four CI-to-weight schemes (flat, step, linear, proposed
non-linear) for BBO-DRL at the primary scale, 30 Monte Carlo trials each.

The weight scheme is switched globally via core.cost_function.set_weight_mode
so that every per-task cost evaluation in the scheduler, environment, and
reward shaping uses the same scheme for the run.

Outputs:
  results/table5_weight_ablation.csv
  results/weight_ablation_raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.algorithms.bbo_drl import BBODRLScheduler
from src.config import (
    GLOBAL_SEED,
    N_FOG_NODES,
    N_RUNS,
    N_WEARABLES,
    PRIMARY_SCALE,
)
from src.core.cost_function import get_weight_mode, set_weight_mode
from src.core.task import HealthcareTask
from src.data_ingestion.event_generator import generate_synthetic_tasks
from src.simulation.environment import OffloadingEnvironment
from src.simulation.topology import build_healthcare_topology

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


WEIGHT_MODES = ['flat', 'step', 'linear', 'nonlinear']


def _to_healthcare(t, topo):
    wids = [nid for nid, n in topo.nodes.items() if n.node_type == 'wearable']
    dev = wids[t.device_id % len(wids)]
    return HealthcareTask(
        task_id=t.task_id, device_id=dev, timestamp=t.timestamp,
        data_size_bits=t.data_size_bits, cpu_cycles=t.cpu_cycles,
        max_delay_s=t.max_delay_s, privacy_sensitivity=t.privacy_sensitivity,
        ci_score=t.ci_score, attack_probability=t.attack_probability,
        source=t.source,
    )


def _run_once(n_tasks, run_id, topo):
    import random as _r
    seed = GLOBAL_SEED + run_id * 1000 + n_tasks
    _r.seed(seed)
    np.random.seed(seed)
    raws = generate_synthetic_tasks(n_tasks, 'mixed', seed=seed)
    tasks = [_to_healthcare(t, topo) for t in raws]
    sched = BBODRLScheduler(topo, seed=seed)
    env = OffloadingEnvironment(topo, sched, n_tasks=n_tasks, seed=seed)
    res = env.run(tasks)
    if not res:
        return None
    return {
        'avg_latency_ms':    mean(r['latency_ms']    for r in res),
        'avg_energy_mj':     mean(r['energy_mj']     for r in res),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in res),
        'sla_violation_pct': 100.0 * sum(r['sla_violated']
                                         for r in res) / len(res),
    }


def run_ablation(n_tasks: int, n_runs: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = build_healthcare_topology(
        n_wearables=N_WEARABLES, n_fog_nodes=N_FOG_NODES, seed=GLOBAL_SEED,
    )

    raw: dict = defaultdict(list)
    for mode in WEIGHT_MODES:
        print(f'[WEIGHT-AB] Mode={mode}  ({n_runs} runs, n_tasks={n_tasks})')
        set_weight_mode(mode)
        it = range(n_runs)
        if _TQDM:
            it = tqdm(it, desc=f'  {mode:<10s}', leave=False, ncols=70)
        for run_id in it:
            m = _run_once(n_tasks, run_id, topo)
            if m is not None:
                raw[mode].append(m)
    # Reset to default
    set_weight_mode('nonlinear')

    # Aggregate
    summary = {}
    csv_rows = []
    for mode in WEIGHT_MODES:
        runs = raw[mode]
        row = {'weight_mode': mode}
        agg = {}
        for k in ['avg_latency_ms', 'avg_energy_mj',
                  'avg_privacy_risk', 'sla_violation_pct']:
            vs = np.array([r[k] for r in runs], dtype=float)
            agg[k] = {'mean': float(vs.mean()), 'std': float(vs.std()),
                      'samples': vs.tolist()}
            row[f'{k}_mean'] = float(vs.mean())
            row[f'{k}_std']  = float(vs.std())
        summary[mode] = agg
        csv_rows.append(row)

    with open(out_dir / 'table5_weight_ablation.csv', 'w', newline='',
              encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(csv_rows)
    with open(out_dir / 'weight_ablation_raw.json', 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    print('\n[WEIGHT-AB] Summary:')
    hdr = (f"  {'Mode':<10s} {'Lat(ms)':>10s} {'Eng(mJ)':>10s} "
           f"{'Priv':>8s} {'SLA%':>8s}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for mode in WEIGHT_MODES:
        s = summary[mode]
        print(f'  {mode:<10s} '
              f'{s["avg_latency_ms"]["mean"]:>10.2f} '
              f'{s["avg_energy_mj"]["mean"]:>10.4f} '
              f'{s["avg_privacy_risk"]["mean"]:>8.4f} '
              f'{s["sla_violation_pct"]["mean"]:>8.2f}')
    print(f'\n[WEIGHT-AB] Saved table5_weight_ablation.csv and '
          f'weight_ablation_raw.json -> {out_dir}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs', type=int, default=N_RUNS)
    p.add_argument('--n_tasks', type=int, default=PRIMARY_SCALE)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    out_dir = (Path(args.output) if args.output
               else project_root / 'results')
    run_ablation(args.n_tasks, args.n_runs, out_dir)


if __name__ == '__main__':
    main()
