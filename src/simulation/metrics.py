"""
Simulation metrics aggregation and reporting.

Provides:
  - SimulationMetrics dataclass for structured per-algorithm results
  - compute_metrics(): aggregate per-task dicts → SimulationMetrics
  - compare_algorithms(): formatted comparison table (console + return string)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationMetrics:
    """Aggregated performance summary for one algorithm run."""
    algorithm_name: str
    n_tasks: int

    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float

    # Energy
    avg_energy_mj: float
    total_energy_mwh: float          # milliwatt-hours = Σ energy_mJ / 3600
    std_energy_mj: float

    # Privacy & cost
    avg_privacy_risk: float
    avg_cost: float

    # SLA
    sla_violation_rate: float        # fraction of tasks violating deadline
    n_sla_violations: int

    # Throughput & timing
    throughput_tasks_per_s: float    # n_tasks / sim_duration_s
    convergence_time_ms: float       # wall-clock time to run experiment

    # Offloading distribution
    pct_local: float                 # % tasks run locally on wearable
    pct_edge: float                  # % tasks offloaded to edge
    pct_fog: float                   # % tasks offloaded to fog
    pct_cloud: float                 # % tasks offloaded to cloud


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    results: List[dict],
    algorithm_name: str,
    sim_duration_s: float,
    wall_clock_start: Optional[float] = None,
) -> SimulationMetrics:
    """
    Aggregate a list of per-task result dicts into a SimulationMetrics object.

    Parameters
    ----------
    results         : list of dicts returned by OffloadingEnvironment.run()
    algorithm_name  : str — label for this algorithm
    sim_duration_s  : float — simulated time span (max_timestamp - min_timestamp)
    wall_clock_start: float — time.time() at experiment start (for convergence_time_ms)

    Returns
    -------
    SimulationMetrics
    """
    if not results:
        return _empty_metrics(algorithm_name)

    n = len(results)

    latencies  = np.array([r['latency_ms']    for r in results], dtype=float)
    energies   = np.array([r['energy_mj']     for r in results], dtype=float)
    priv_risks = np.array([r['privacy_risk']  for r in results], dtype=float)
    costs      = np.array([r['cost']          for r in results], dtype=float)
    sla_viols  = np.array([r['sla_violated']  for r in results], dtype=bool)

    node_types = [r.get('node_type', 'unknown') for r in results]

    # Energy in mWh: sum of mJ / 3600
    total_energy_mwh = float(energies.sum()) / 3600.0

    # Throughput
    throughput = n / max(sim_duration_s, 1e-6)

    # Convergence time (wall-clock)
    if wall_clock_start is not None:
        convergence_ms = (time.time() - wall_clock_start) * 1000.0
    else:
        convergence_ms = 0.0

    # Offloading distribution
    type_counts = {'wearable': 0, 'edge': 0, 'fog': 0, 'cloud': 0}
    for nt in node_types:
        if nt in type_counts:
            type_counts[nt] += 1

    return SimulationMetrics(
        algorithm_name=algorithm_name,
        n_tasks=n,

        avg_latency_ms=float(latencies.mean()),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        max_latency_ms=float(latencies.max()),
        std_latency_ms=float(latencies.std()),

        avg_energy_mj=float(energies.mean()),
        total_energy_mwh=total_energy_mwh,
        std_energy_mj=float(energies.std()),

        avg_privacy_risk=float(priv_risks.mean()),
        avg_cost=float(costs.mean()),

        sla_violation_rate=float(sla_viols.mean()),
        n_sla_violations=int(sla_viols.sum()),

        throughput_tasks_per_s=throughput,
        convergence_time_ms=convergence_ms,

        pct_local=100.0 * type_counts['wearable'] / n,
        pct_edge=100.0 * type_counts['edge'] / n,
        pct_fog=100.0 * type_counts['fog'] / n,
        pct_cloud=100.0 * type_counts['cloud'] / n,
    )


def _empty_metrics(algorithm_name: str) -> SimulationMetrics:
    """Return a zero-filled metrics object (for empty result sets)."""
    return SimulationMetrics(
        algorithm_name=algorithm_name,
        n_tasks=0,
        avg_latency_ms=0.0, p50_latency_ms=0.0, p95_latency_ms=0.0,
        p99_latency_ms=0.0, max_latency_ms=0.0, std_latency_ms=0.0,
        avg_energy_mj=0.0, total_energy_mwh=0.0, std_energy_mj=0.0,
        avg_privacy_risk=0.0, avg_cost=0.0,
        sla_violation_rate=0.0, n_sla_violations=0,
        throughput_tasks_per_s=0.0, convergence_time_ms=0.0,
        pct_local=0.0, pct_edge=0.0, pct_fog=0.0, pct_cloud=0.0,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_algorithms(
    all_results: Dict[str, SimulationMetrics],
    verbose: bool = True,
) -> str:
    """
    Print and return a formatted comparison table of algorithm metrics.

    Parameters
    ----------
    all_results : dict mapping algorithm_name → SimulationMetrics
    verbose     : bool — print to stdout if True

    Returns
    -------
    table_str : str — formatted ASCII table
    """
    if not all_results:
        return "No results to compare."

    # Column headers
    cols = [
        ('Algorithm',         'algorithm_name',       '<20s'),
        ('Tasks',             'n_tasks',               '>6d'),
        ('Avg Lat (ms)',      'avg_latency_ms',        '>12.2f'),
        ('P95 Lat (ms)',      'p95_latency_ms',        '>12.2f'),
        ('Avg E (mJ)',        'avg_energy_mj',         '>11.4f'),
        ('Total E (mWh)',     'total_energy_mwh',      '>13.4f'),
        ('Priv Risk',         'avg_privacy_risk',      '>10.4f'),
        ('SLA Viol %',        'sla_violation_rate',    '>10.2%'),
        ('Throughput',        'throughput_tasks_per_s','>11.1f'),
        ('Avg Cost',          'avg_cost',              '>9.4f'),
    ]

    # Build header row
    header = '  '.join(f"{c[0]:{c[2][1:-2]}s}" if c[2][-1] == 's'
                        else f"{c[0]:>{c[2][1:-1]}}" for c in cols)
    # Simpler approach:
    sep = '-' * 130
    lines = [sep, '  ALGORITHM PERFORMANCE COMPARISON', sep]

    # Header
    hdr_parts = []
    for label, _, fmt in cols:
        width = int(''.join(filter(str.isdigit, fmt.split('.')[0])))
        hdr_parts.append(f"{label:>{width}}")
    lines.append('  '.join(hdr_parts))
    lines.append(sep)

    for algo_name, m in sorted(all_results.items()):
        row_parts = []
        for label, attr, fmt in cols:
            val = getattr(m, attr)
            try:
                formatted = format(val, fmt.lstrip('<>^'))
            except (ValueError, TypeError):
                formatted = str(val)
            width = int(''.join(filter(str.isdigit, fmt.split('.')[0])))
            row_parts.append(f"{formatted:>{width}}")
        lines.append('  '.join(row_parts))

    lines.append(sep)

    # Highlight best per metric
    metric_attrs = [
        ('avg_latency_ms',       'lower'),
        ('avg_energy_mj',        'lower'),
        ('avg_privacy_risk',     'lower'),
        ('sla_violation_rate',   'lower'),
        ('throughput_tasks_per_s', 'higher'),
    ]
    lines.append('  BEST PER METRIC:')
    for attr, direction in metric_attrs:
        vals = {name: getattr(m, attr) for name, m in all_results.items()}
        best_name = (min if direction == 'lower' else max)(vals, key=lambda k: vals[k])
        best_val = vals[best_name]
        lines.append(f"    {attr:<30s}: {best_name} ({best_val:.4f})")
    lines.append(sep)

    table_str = '\n'.join(lines)
    if verbose:
        print(table_str)
    return table_str


# ---------------------------------------------------------------------------
# Monte Carlo aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_mc_runs(
    runs: List[SimulationMetrics],
) -> Dict[str, dict]:
    """
    Aggregate a list of Monte Carlo SimulationMetrics into mean ± std dicts.

    Parameters
    ----------
    runs : list of SimulationMetrics from repeated runs

    Returns
    -------
    dict mapping metric_name → {'mean': float, 'std': float}
    """
    if not runs:
        return {}

    numeric_attrs = [
        'avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
        'max_latency_ms', 'std_latency_ms',
        'avg_energy_mj', 'total_energy_mwh', 'std_energy_mj',
        'avg_privacy_risk', 'avg_cost',
        'sla_violation_rate', 'n_sla_violations',
        'throughput_tasks_per_s', 'convergence_time_ms',
        'pct_local', 'pct_edge', 'pct_fog', 'pct_cloud',
    ]

    result = {}
    for attr in numeric_attrs:
        vals = np.array([getattr(r, attr) for r in runs], dtype=float)
        result[attr] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std()),
            'min':  float(vals.min()),
            'max':  float(vals.max()),
        }
    return result


def metrics_to_dict(m: SimulationMetrics) -> dict:
    """Convert a SimulationMetrics object to a plain dictionary."""
    return {
        'algorithm_name':        m.algorithm_name,
        'n_tasks':               m.n_tasks,
        'avg_latency_ms':        m.avg_latency_ms,
        'p50_latency_ms':        m.p50_latency_ms,
        'p95_latency_ms':        m.p95_latency_ms,
        'p99_latency_ms':        m.p99_latency_ms,
        'max_latency_ms':        m.max_latency_ms,
        'std_latency_ms':        m.std_latency_ms,
        'avg_energy_mj':         m.avg_energy_mj,
        'total_energy_mwh':      m.total_energy_mwh,
        'std_energy_mj':         m.std_energy_mj,
        'avg_privacy_risk':      m.avg_privacy_risk,
        'avg_cost':              m.avg_cost,
        'sla_violation_rate':    m.sla_violation_rate,
        'n_sla_violations':      m.n_sla_violations,
        'throughput_tasks_per_s': m.throughput_tasks_per_s,
        'convergence_time_ms':   m.convergence_time_ms,
        'pct_local':             m.pct_local,
        'pct_edge':              m.pct_edge,
        'pct_fog':               m.pct_fog,
        'pct_cloud':             m.pct_cloud,
    }
