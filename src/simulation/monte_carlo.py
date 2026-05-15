"""
Monte Carlo simulation runner.

Executes the full comparative experiment:
  - 6 task scales: [100, 500, 1000, 2000, 5000, 10000]
  - 30 Monte Carlo runs per (algorithm × scale) configuration
  - 6 algorithms: BBO-DRL, PSO, ACO, HS-HHO, LocalOnly, CloudOnly
  - Results: mean ± std per metric, saved to JSON

Usage:
    python -m src.simulation.monte_carlo
    python -m src.simulation.monte_carlo --output results/mc_run1 --n_runs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.algorithms.aco import ACOScheduler
from src.algorithms.bbo_drl import BBODRLScheduler
from src.algorithms.cloud_only import CloudOnlyScheduler
from src.algorithms.hs_hho import HSHHOScheduler
from src.algorithms.local_only import LocalOnlyScheduler
from src.algorithms.pso import PSOScheduler
from src.core.task import HealthcareTask
from src.data_ingestion.event_generator import generate_event_stream
from src.simulation.environment import OffloadingEnvironment
from src.simulation.metrics import (
    SimulationMetrics,
    aggregate_mc_runs,
    compare_algorithms,
    compute_metrics,
    metrics_to_dict,
)
from src.simulation.topology import build_healthcare_topology

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
TASK_SCALES = [100, 500, 1000, 2000, 5000, 10_000]
N_RUNS = 30
N_DEVICES = 10
N_FOG_NODES = 3
SEED_BASE = 42


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

def build_algorithms(topology, seed: int = 42) -> Dict[str, object]:
    """
    Instantiate all six scheduling algorithms for a given topology.

    Returns
    -------
    dict mapping algorithm_name → scheduler instance
    """
    return {
        'BBO-DRL':   BBODRLScheduler(topology, seed=seed),
        'PSO':       PSOScheduler(topology, seed=seed),
        'ACO':       ACOScheduler(topology, seed=seed),
        'HS-HHO':    HSHHOScheduler(topology, seed=seed),
        'LocalOnly':  LocalOnlyScheduler(topology),
        'CloudOnly':  CloudOnlyScheduler(topology),
    }


# ---------------------------------------------------------------------------
# Single run for all algorithms
# ---------------------------------------------------------------------------

def run_all_algorithms(
    n_tasks: int,
    run_id: int,
    topology,
    seed: int,
) -> Dict[str, SimulationMetrics]:
    """
    Execute one Monte Carlo run for all algorithms.

    Parameters
    ----------
    n_tasks  : int — number of tasks in this run
    run_id   : int — run index (used for seed offset)
    topology : NetworkTopology (freshly built for this run)
    seed     : int — base random seed

    Returns
    -------
    dict mapping algorithm_name → SimulationMetrics
    """
    # Generate task stream for this run
    task_stream = generate_event_stream(
        n_tasks=n_tasks,
        n_devices=N_DEVICES,
        seed=seed + run_id,
    )

    # Convert SimulationTask → HealthcareTask
    tasks = _to_healthcare_tasks(task_stream)

    # Compute simulation duration (max - min timestamp)
    timestamps = [t.timestamp for t in tasks]
    sim_duration_s = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1.0

    run_results: Dict[str, SimulationMetrics] = {}

    algorithms = build_algorithms(topology, seed=seed + run_id)

    for algo_name, scheduler in algorithms.items():
        wall_start = time.time()

        # Fresh environment for each algorithm (same tasks, fresh state)
        env = OffloadingEnvironment(
            topology=topology,
            scheduler=scheduler,
            n_tasks=n_tasks,
            seed=seed + run_id,
        )
        env.reset()

        # Deep-copy tasks so scheduler decisions don't bleed across algorithms
        tasks_copy = _copy_tasks(tasks)

        results = env.run(tasks_copy)

        metrics = compute_metrics(
            results=results,
            algorithm_name=algo_name,
            sim_duration_s=sim_duration_s,
            wall_clock_start=wall_start,
        )
        run_results[algo_name] = metrics

    return run_results


# ---------------------------------------------------------------------------
# Monte Carlo main loop
# ---------------------------------------------------------------------------

def run_monte_carlo(
    output_dir: str = 'results',
    task_scales: Optional[List[int]] = None,
    n_runs: int = N_RUNS,
    verbose: bool = True,
) -> dict:
    """
    Full Monte Carlo experiment across all task scales and algorithms.

    Results structure:
      {
        'task_scale_N': {
          'algorithm_name': {
            'runs': [metrics_dict, ...],
            'summary': {metric: {'mean': ..., 'std': ...}, ...}
          }
        }
      }

    Parameters
    ----------
    output_dir  : str  — directory to write JSON results
    task_scales : list — override default TASK_SCALES
    n_runs      : int  — number of MC iterations per configuration
    verbose     : bool — print progress

    Returns
    -------
    Full results dict (also saved to JSON).
    """
    if task_scales is None:
        task_scales = TASK_SCALES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_experiment_results = {}
    experiment_start = time.time()

    for n_tasks in task_scales:
        scale_key = f'n_tasks_{n_tasks}'
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  TASK SCALE: {n_tasks}  |  {n_runs} Monte Carlo runs")
            print(f"{'=' * 70}")

        # Build topology once per scale (same physical network)
        topology = build_healthcare_topology(
            n_wearables=N_DEVICES,
            n_fog_nodes=N_FOG_NODES,
            seed=SEED_BASE,
        )

        # Per-algorithm accumulator
        # algo_name → list of SimulationMetrics
        algo_mc_results: Dict[str, List[SimulationMetrics]] = {
            'BBO-DRL':   [],
            'PSO':       [],
            'ACO':       [],
            'HS-HHO':    [],
            'LocalOnly':  [],
            'CloudOnly':  [],
        }

        for run_id in range(n_runs):
            run_seed = SEED_BASE + run_id * 1000

            if verbose and (run_id % 5 == 0 or run_id == n_runs - 1):
                elapsed = time.time() - experiment_start
                print(
                    f"  Run {run_id + 1:3d}/{n_runs}  "
                    f"[n_tasks={n_tasks}]  "
                    f"elapsed={elapsed:.1f}s"
                )

            run_metrics = run_all_algorithms(
                n_tasks=n_tasks,
                run_id=run_id,
                topology=topology,
                seed=run_seed,
            )

            for algo_name, metrics in run_metrics.items():
                algo_mc_results[algo_name].append(metrics)

        # Aggregate MC results per algorithm
        scale_summary = {}
        best_metrics: Dict[str, SimulationMetrics] = {}

        for algo_name, runs_list in algo_mc_results.items():
            summary = aggregate_mc_runs(runs_list)
            scale_summary[algo_name] = {
                'runs':    [metrics_to_dict(m) for m in runs_list],
                'summary': summary,
            }
            # Store the final (last) run's metrics for the comparison table
            if runs_list:
                best_metrics[algo_name] = runs_list[-1]

        all_experiment_results[scale_key] = scale_summary

        # Print comparison table for this scale
        if verbose and best_metrics:
            print(f"\n  [Scale={n_tasks}] Final-run comparison:")
            compare_algorithms(best_metrics, verbose=True)

        # Save intermediate results (allows recovery on crash)
        _save_results(all_experiment_results, out_path / 'mc_results_partial.json')

    # Save final results
    final_path = out_path / 'mc_results_final.json'
    _save_results(all_experiment_results, final_path)

    total_time = time.time() - experiment_start
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Monte Carlo complete.  Total wall time: {total_time:.1f}s")
        print(f"  Results saved to: {final_path}")
        print(f"{'=' * 70}")

    return all_experiment_results


# ---------------------------------------------------------------------------
# Helper: save results to JSON
# ---------------------------------------------------------------------------

def _save_results(data: dict, path: Path) -> None:
    """Serialise results dict to JSON, handling numpy scalars."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, default=_convert)


# ---------------------------------------------------------------------------
# Helper: SimulationTask → HealthcareTask
# ---------------------------------------------------------------------------

def _to_healthcare_tasks(sim_tasks) -> List[HealthcareTask]:
    """Convert event_generator.SimulationTask objects to HealthcareTask."""
    out = []
    for st in sim_tasks:
        ht = HealthcareTask(
            task_id=st.task_id,
            device_id=st.device_id,
            timestamp=st.timestamp,
            data_size_bits=st.data_size_bits,
            cpu_cycles=st.cpu_cycles,
            max_delay_s=st.max_delay_s,
            privacy_sensitivity=st.privacy_sensitivity,
            ci_score=st.ci_score,
            attack_probability=st.attack_probability,
            source=st.source,
        )
        out.append(ht)
    return out


def _copy_tasks(tasks: List[HealthcareTask]) -> List[HealthcareTask]:
    """Shallow copy of task list, resetting mutable outcome fields."""
    out = []
    for t in tasks:
        ht = HealthcareTask(
            task_id=t.task_id,
            device_id=t.device_id,
            timestamp=t.timestamp,
            data_size_bits=t.data_size_bits,
            cpu_cycles=t.cpu_cycles,
            max_delay_s=t.max_delay_s,
            privacy_sensitivity=t.privacy_sensitivity,
            ci_score=t.ci_score,
            attack_probability=t.attack_probability,
            source=t.source,
            assigned_node=None,
            actual_latency_s=None,
            actual_energy_j=None,
            sla_violated=False,
        )
        out.append(ht)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo simulation for Bio-Inspired Task Offloading.'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for JSON results (default: results/)'
    )
    parser.add_argument(
        '--n_runs', type=int, default=N_RUNS,
        help=f'Monte Carlo iterations per configuration (default: {N_RUNS})'
    )
    parser.add_argument(
        '--scales', type=int, nargs='+', default=None,
        help='Task scales to test (default: 100 500 1000 2000 5000 10000)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_monte_carlo(
        output_dir=args.output,
        task_scales=args.scales,
        n_runs=args.n_runs,
        verbose=not args.quiet,
    )
