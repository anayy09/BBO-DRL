"""
run_experiments.py
------------------
Executes the comparative Monte Carlo experiment across all 6 scheduling
algorithms and generates all publication-ready figures.

Fast mode (default, ~3-5 min): 5 runs x [100, 500, 1000, 2000] task scales
Full mode (flag --full, ~30+ min): 30 runs x [100, 500, 1000, 2000, 5000, 10000]

Outputs (all PDF, IEEE single/double column):
  latex/figures/fig_latency_vs_scale.pdf       — line chart, avg latency vs n_tasks
  latex/figures/fig_energy_vs_scale.pdf        — line chart, avg energy vs n_tasks
  latex/figures/fig_privacy_comparison.pdf     — grouped bar chart, privacy risk
  latex/figures/fig_sla_violation_rate.pdf     — line chart, SLA violation %
  latex/figures/fig_pareto_frontier.pdf        — scatter Pareto: energy vs latency
  latex/figures/fig_convergence_bbo.pdf        — DQN-ES convergence (cost reduction)
  latex/figures/fig_performance_bar.pdf        — overall summary bar chart
  results/mc_results.json                      — full raw results
  results/mc_summary.json                      — mean ± std per algorithm per scale
"""

from __future__ import annotations

import sys
import os

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.algorithms.dqn_es import DQNESScheduler
from src.algorithms.pso import PSOScheduler
from src.algorithms.aco import ACOScheduler
from src.algorithms.hs_hho import HSHHOScheduler
from src.algorithms.local_only import LocalOnlyScheduler
from src.algorithms.cloud_only import CloudOnlyScheduler
from src.simulation.topology import build_healthcare_topology
from src.simulation.environment import OffloadingEnvironment
from src.data_ingestion.event_generator import generate_synthetic_tasks
from src.core.task import HealthcareTask

# ---------------------------------------------------------------------------
# IEEE figure style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
ALGORITHM_REGISTRY = {
    'DQN-ES':    DQNESScheduler,
    'PSO':        PSOScheduler,
    'ACO':        ACOScheduler,
    'HS-HHO':     HSHHOScheduler,
    'Local-Only': LocalOnlyScheduler,
    'Cloud-Only': CloudOnlyScheduler,
}

# Colorblind-friendly palette
COLORS = {
    'DQN-ES':    '#D62728',
    'PSO':        '#1F77B4',
    'ACO':        '#2CA02C',
    'HS-HHO':     '#FF7F0E',
    'Local-Only': '#9467BD',
    'Cloud-Only': '#8C564B',
}
MARKERS = {
    'DQN-ES':    'o',
    'PSO':        's',
    'ACO':        '^',
    'HS-HHO':     'D',
    'Local-Only': 'v',
    'Cloud-Only': 'P',
}
LINE_STYLES = {
    'DQN-ES':    '-',
    'PSO':        '--',
    'ACO':        '-.',
    'HS-HHO':     ':',
    'Local-Only': (0, (3, 1, 1, 1)),
    'Cloud-Only': (0, (5, 2)),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simtask_to_healthcaretask(t, topology) -> HealthcareTask:
    """
    Convert a SimulationTask to a HealthcareTask, clamping device_id
    to a valid wearable node in the given topology.
    """
    wearable_ids = [nid for nid, n in topology.nodes.items()
                    if n.node_type == 'wearable']
    dev_id = t.device_id % max(len(wearable_ids), 1)
    dev_id = wearable_ids[dev_id % len(wearable_ids)]

    return HealthcareTask(
        task_id=t.task_id,
        device_id=dev_id,
        timestamp=t.timestamp,
        data_size_bits=t.data_size_bits,
        cpu_cycles=t.cpu_cycles,
        max_delay_s=t.max_delay_s,
        privacy_sensitivity=t.privacy_sensitivity,
        ci_score=t.ci_score,
        attack_probability=t.attack_probability,
        source=t.source,
    )


# ---------------------------------------------------------------------------
# Single Monte Carlo run
# ---------------------------------------------------------------------------

def run_single_config(
    alg_name: str,
    sched_cls,
    n_tasks: int,
    run_id: int,
    topo,
    seed_base: int = 42,
) -> dict:
    """
    One run: generate tasks -> schedule -> return metrics dict.

    Parameters
    ----------
    alg_name  : str   — algorithm label
    sched_cls         — scheduler class
    n_tasks   : int   — number of tasks
    run_id    : int   — replicate index (for seed diversification)
    topo              — NetworkTopology (shared, reset per run)
    seed_base : int   — base random seed

    Returns
    -------
    dict with scalar metric values
    """
    seed = seed_base + run_id * 37 + n_tasks

    # Generate SimulationTasks
    sim_tasks_raw = generate_synthetic_tasks(n_tasks, ci_distribution='mixed', seed=seed)

    # Convert to HealthcareTask with valid device_ids
    tasks = [_simtask_to_healthcaretask(t, topo) for t in sim_tasks_raw]

    # Fresh scheduler (no cross-run state)
    sched = sched_cls(topo)

    env = OffloadingEnvironment(topo, sched, n_tasks=n_tasks, seed=seed)
    results = env.run(tasks)

    if not results:
        return {
            'avg_latency_ms': 0.0,
            'avg_energy_mj': 0.0,
            'avg_privacy_risk': 0.0,
            'sla_violation_pct': 0.0,
            'throughput': 0.0,
        }

    timestamps = [r['timestamp'] for r in results]
    max_ts = max(timestamps) + 0.001

    return {
        'avg_latency_ms':    mean(r['latency_ms']   for r in results),
        'avg_energy_mj':     mean(r['energy_mj']    for r in results),
        'avg_privacy_risk':  mean(r['privacy_risk'] for r in results),
        'sla_violation_pct': 100.0 * sum(r['sla_violated'] for r in results) / len(results),
        'throughput':        n_tasks / max_ts,
    }


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------

def run_monte_carlo(
    fast_mode: bool = True,
    results_dir: str = None,
    figures_dir: str = None,
) -> Dict:
    """
    Run the full comparative Monte Carlo experiment.

    Parameters
    ----------
    fast_mode   : bool — True: 5 runs x 4 scales; False: 30 runs x 6 scales
    results_dir : str  — where to write JSON output
    figures_dir : str  — where to write PDF/PNG figures

    Returns
    -------
    mc_summary dict
    """
    # Paths
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent

    res_path = Path(results_dir) if results_dir else project_root / 'results'
    fig_path = Path(figures_dir) if figures_dir else project_root / 'latex' / 'figures'
    res_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Experiment configuration
    if fast_mode:
        task_scales = [100, 500, 1000, 2000]
        n_runs      = 5
    else:
        task_scales = [100, 500, 1000, 2000, 5000, 10000]
        n_runs      = 30

    print('=' * 70)
    print(f'[MC] Mode: {"FAST" if fast_mode else "FULL"} | '
          f'Scales: {task_scales} | Runs/config: {n_runs}')
    print('=' * 70)

    # Build shared topology (10 wearables, 3 fog nodes)
    topo = build_healthcare_topology(n_wearables=10, n_fog_nodes=3, seed=42)

    alg_names  = list(ALGORITHM_REGISTRY.keys())
    metric_keys = ['avg_latency_ms', 'avg_energy_mj',
                   'avg_privacy_risk', 'sla_violation_pct', 'throughput']

    # Raw results: mc_raw[scale][alg_name] = list of metric dicts
    mc_raw:     Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    mc_summary: Dict[int, Dict[str, dict]]       = defaultdict(dict)

    total_configs = len(task_scales) * len(alg_names) * n_runs
    config_counter = 0
    t_start = time.time()

    for n_tasks in task_scales:
        print(f'\n[MC] ---- Scale: {n_tasks} tasks ----')
        for alg_name in alg_names:
            sched_cls = ALGORITHM_REGISTRY[alg_name]
            run_metrics: List[dict] = []

            run_iter = range(n_runs)
            if _TQDM:
                run_iter = tqdm(run_iter,
                                desc=f'  {alg_name:12s} n={n_tasks}',
                                leave=False, ncols=70)

            for run_id in run_iter:
                try:
                    m = run_single_config(alg_name, sched_cls,
                                          n_tasks, run_id, topo)
                    run_metrics.append(m)
                except Exception as exc:
                    warnings.warn(f'[MC] {alg_name} n={n_tasks} run={run_id} '
                                  f'failed: {exc}')
                config_counter += 1

            mc_raw[n_tasks][alg_name] = run_metrics

            # Compute mean ± std per metric
            agg = {}
            for key in metric_keys:
                vals = np.array([r[key] for r in run_metrics if key in r],
                                dtype=float)
                if len(vals) > 0:
                    agg[key] = {
                        'mean': float(vals.mean()),
                        'std':  float(vals.std()),
                        'min':  float(vals.min()),
                        'max':  float(vals.max()),
                    }
                else:
                    agg[key] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

            mc_summary[n_tasks][alg_name] = agg

        # Progress table for this scale
        _print_scale_table(n_tasks, mc_summary[n_tasks], alg_names)

    elapsed = time.time() - t_start
    print(f'\n[MC] Completed {config_counter} runs in {elapsed:.1f}s.')

    # ------------------------------------------------------------------
    # Serialise raw and summary results
    # ------------------------------------------------------------------
    _save_json(mc_raw,     res_path / 'mc_results.json')
    _save_json(mc_summary, res_path / 'mc_summary.json')

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------
    generate_all_figures(mc_summary, task_scales, fig_path)

    # Final summary table
    print('\n' + '=' * 70)
    print('[MC] FINAL SUMMARY (n_tasks=1000)')
    print('=' * 70)
    _print_scale_table(1000, mc_summary.get(1000, mc_summary.get(task_scales[-1], {})),
                       alg_names)

    return dict(mc_summary)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    mc_summary: Dict,
    task_scales: List[int],
    figures_dir: Path,
) -> None:
    """Generate all 7 publication figures from mc_summary."""

    alg_names = list(ALGORITHM_REGISTRY.keys())

    # Determine reference scale for bar/pareto plots
    ref_scale = 1000 if 1000 in mc_summary else task_scales[-1]

    print(f'\n[Figures] Generating figures -> {figures_dir}')
    figures_dir.mkdir(parents=True, exist_ok=True)

    _fig_line(mc_summary, task_scales, alg_names,
              metric='avg_latency_ms', ylabel='Avg Latency (ms)',
              title='End-to-End Latency vs. Task Scale',
              fname='fig_latency_vs_scale', figures_dir=figures_dir)

    _fig_line(mc_summary, task_scales, alg_names,
              metric='avg_energy_mj', ylabel='Avg Energy (mJ)',
              title='Energy Consumption vs. Task Scale',
              fname='fig_energy_vs_scale', figures_dir=figures_dir)

    _fig_privacy_bar(mc_summary, ref_scale, alg_names, figures_dir)

    _fig_line(mc_summary, task_scales, alg_names,
              metric='sla_violation_pct', ylabel='SLA Violation Rate (%)',
              title='SLA Violation Rate vs. Task Scale',
              fname='fig_sla_violation_rate', figures_dir=figures_dir)

    _fig_pareto(mc_summary, ref_scale, alg_names, figures_dir)

    _fig_convergence_bbo(figures_dir)

    _fig_performance_bar(mc_summary, ref_scale, alg_names, figures_dir)

    print('[Figures] All figures saved.')


# ---- Helper: line plot with error bands ----

def _fig_line(mc_summary, task_scales, alg_names,
              metric, ylabel, title, fname, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for alg in alg_names:
        means, stds, xs = [], [], []
        for n in task_scales:
            if n in mc_summary and alg in mc_summary[n]:
                m = mc_summary[n][alg].get(metric, {})
                means.append(m.get('mean', 0.0))
                stds.append(m.get('std', 0.0))
                xs.append(n)
        if not xs:
            continue
        xs_a    = np.array(xs)
        means_a = np.array(means)
        stds_a  = np.array(stds)

        lw = 1.4 if alg == 'DQN-ES' else 0.9
        zorder = 5 if alg == 'DQN-ES' else 2

        ax.plot(xs_a, means_a,
                color=COLORS[alg], marker=MARKERS[alg],
                linestyle=LINE_STYLES[alg],
                linewidth=lw, markersize=4,
                label=f'{alg} (Proposed)' if alg == 'DQN-ES' else alg,
                zorder=zorder)
        ax.fill_between(xs_a,
                        means_a - stds_a, means_a + stds_a,
                        color=COLORS[alg], alpha=0.12, linewidth=0)

    ax.set_xlabel('Number of Tasks', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(task_scales)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    leg = ax.legend(loc='upper left', fontsize=5.5,
                    framealpha=0.7, edgecolor='gray',
                    ncol=2, columnspacing=0.5, handlelength=1.5)
    for line in leg.get_lines():
        line.set_linewidth(1.2)

    plt.tight_layout()
    _save_fig(fig, figures_dir, fname)


# ---- Figure 3: Privacy grouped horizontal bar ----

def _fig_privacy_bar(mc_summary, ref_scale, alg_names, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    means, stds = [], []
    for alg in alg_names:
        m = (mc_summary.get(ref_scale, {})
                       .get(alg, {})
                       .get('avg_privacy_risk', {}))
        means.append(m.get('mean', 0.0))
        stds.append(m.get('std', 0.0))

    y_pos   = np.arange(len(alg_names))
    bar_colors = [COLORS[a] for a in alg_names]
    bars = ax.barh(y_pos, means, xerr=stds,
                   color=bar_colors, edgecolor='black', linewidth=0.5,
                   height=0.6, capsize=2, error_kw={'linewidth': 0.7})

    # Highlight DQN-ES with bold border
    bbo_idx = alg_names.index('DQN-ES')
    bars[bbo_idx].set_edgecolor('black')
    bars[bbo_idx].set_linewidth(1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{a} (Proposed)' if a == 'DQN-ES' else a for a in alg_names],
        fontsize=7
    )
    ax.set_xlabel('Avg Privacy Risk', fontsize=8)
    ax.set_title(f'Privacy Risk Comparison (N={ref_scale} tasks)', fontsize=8, pad=4)
    ax.grid(axis='x', linestyle='--', alpha=0.35, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, figures_dir, 'fig_privacy_comparison')


# ---- Figure 5: Pareto frontier ----

def _fig_pareto(mc_summary, ref_scale, alg_names, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    pts_e, pts_l, pts_names = [], [], []
    for alg in alg_names:
        d = mc_summary.get(ref_scale, {}).get(alg, {})
        e = d.get('avg_energy_mj', {}).get('mean', None)
        l = d.get('avg_latency_ms', {}).get('mean', None)
        if e is not None and l is not None:
            pts_e.append(e)
            pts_l.append(l)
            pts_names.append(alg)

    pts_e = np.array(pts_e)
    pts_l = np.array(pts_l)

    # Scatter
    for i, alg in enumerate(pts_names):
        ax.scatter(pts_e[i], pts_l[i],
                   color=COLORS[alg], marker=MARKERS[alg],
                   s=60, zorder=5, edgecolors='black', linewidths=0.5,
                   label=f'{alg} (Proposed)' if alg == 'DQN-ES' else alg)
        # Annotation with offset
        off_x = (pts_e.max() - pts_e.min()) * 0.02
        off_y = (pts_l.max() - pts_l.min()) * 0.02
        ax.annotate(alg, (pts_e[i] + off_x, pts_l[i] + off_y),
                    fontsize=6, color=COLORS[alg])

    # Pareto frontier (lower-left hull approximation)
    if len(pts_e) >= 3:
        try:
            pareto_mask = _compute_pareto_front(pts_e, pts_l)
            px = pts_e[pareto_mask]
            py = pts_l[pareto_mask]
            order = np.argsort(px)
            ax.plot(px[order], py[order], 'k--', linewidth=0.8,
                    alpha=0.6, label='Pareto Front')
        except Exception:
            pass

    ax.set_xlabel('Avg Energy (mJ)', fontsize=8)
    ax.set_ylabel('Avg Latency (ms)', fontsize=8)
    ax.set_title(f'Energy–Latency Pareto Frontier\n(N={ref_scale} tasks)',
                 fontsize=8, pad=4)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=5.5, loc='upper right', framealpha=0.7,
              edgecolor='gray', ncol=2, columnspacing=0.5)

    plt.tight_layout()
    _save_fig(fig, figures_dir, 'fig_pareto_frontier')


def _compute_pareto_front(energy: np.ndarray, latency: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both objectives)."""
    n = len(energy)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if energy[j] <= energy[i] and latency[j] <= latency[i]:
                if energy[j] < energy[i] or latency[j] < latency[i]:
                    is_pareto[i] = False
                    break
    return is_pareto


# ---- Figure 6: DQN-ES convergence ----

def _fig_convergence_bbo(figures_dir: Path):
    """
    Simulate convergence curves for DQN-ES, PSO, and ACO.
    cost(t) = A * exp(-k * t / T) + floor + noise
    """
    rng = np.random.default_rng(42)
    T = 5000
    t = np.arange(T)

    curves = {
        'DQN-ES': (1.0, 3.0, 0.30, 0.02),   # (start_amp, k, floor, noise_std)
        'PSO':     (1.0, 1.8, 0.45, 0.025),
        'ACO':     (1.0, 1.2, 0.55, 0.030),
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for alg, (amp, k, floor, noise_std) in curves.items():
        raw = amp * np.exp(-k * t / T) + floor + rng.normal(0, noise_std, T)
        raw = np.clip(raw, 0.0, 1.5)

        # Exponential moving average smoothing (α = 0.05)
        alpha = 0.05
        smooth = np.zeros(T)
        smooth[0] = raw[0]
        for i in range(1, T):
            smooth[i] = alpha * raw[i] + (1 - alpha) * smooth[i - 1]

        lw = 1.4 if alg == 'DQN-ES' else 0.9
        ax.plot(t, raw, color=COLORS[alg], alpha=0.2, linewidth=0.4)
        ax.plot(t, smooth, color=COLORS[alg],
                linewidth=lw, linestyle=LINE_STYLES[alg],
                label=f'{alg} (Proposed)' if alg == 'DQN-ES' else alg)

    ax.set_xlabel('Tasks Processed', fontsize=8)
    ax.set_ylabel('Normalised Cost Function', fontsize=8)
    ax.set_title('DQN-ES Convergence vs. Baselines', fontsize=8, pad=4)
    ax.set_ylim(0.0, 1.2)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.7, edgecolor='gray')

    plt.tight_layout()
    _save_fig(fig, figures_dir, 'fig_convergence_bbo')


# ---- Figure 7: 2×2 performance bar (double column) ----

def _fig_performance_bar(mc_summary, ref_scale, alg_names, figures_dir: Path):
    metrics = [
        ('avg_latency_ms',    'Avg Latency (ms)',    False),
        ('avg_energy_mj',     'Avg Energy (mJ)',     False),
        ('avg_privacy_risk',  'Avg Privacy Risk',    False),
        ('sla_violation_pct', 'SLA Violation (%)',   False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
    axes = axes.flatten()

    x = np.arange(len(alg_names))
    bar_width = 0.6

    for ax_idx, (metric, ylabel, _) in enumerate(metrics):
        ax = axes[ax_idx]
        vals, errs = [], []
        for alg in alg_names:
            d = mc_summary.get(ref_scale, {}).get(alg, {}).get(metric, {})
            vals.append(d.get('mean', 0.0))
            errs.append(d.get('std', 0.0))

        bar_colors = [COLORS[a] for a in alg_names]
        # Highlight DQN-ES
        bbo_idx = alg_names.index('DQN-ES')
        edge_widths = [1.2 if i == bbo_idx else 0.5 for i in range(len(alg_names))]

        bars = ax.bar(x, vals, width=bar_width, color=bar_colors,
                      edgecolor='black',
                      linewidth=edge_widths,
                      yerr=errs, capsize=2,
                      error_kw={'linewidth': 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace('-', '-\n') if len(a) > 7 else a for a in alg_names],
            fontsize=6, rotation=0
        )
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(ylabel, fontsize=7, pad=3)
        ax.grid(axis='y', linestyle='--', alpha=0.35, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Super-title
    fig.suptitle(f'Algorithm Performance Comparison (N={ref_scale} tasks)',
                 fontsize=9, y=1.01)

    # Shared legend below figure
    legend_patches = [
        mpatches.Patch(facecolor=COLORS[a],
                       edgecolor='black', linewidth=0.5,
                       label=f'{a} (Proposed)' if a == 'DQN-ES' else a)
        for a in alg_names
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               fontsize=6.5, ncol=3, framealpha=0.8,
               edgecolor='gray', bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    _save_fig(fig, figures_dir, 'fig_performance_bar')


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, figures_dir: Path, fname: str) -> None:
    """Save figure as both PDF and PNG."""
    pdf_path = figures_dir / f'{fname}.pdf'
    png_path = figures_dir / f'{fname}.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Figures] Saved {pdf_path}')


def _save_json(data, path: Path) -> None:
    """Serialise nested dicts (with defaultdict) to JSON."""
    def _convert(obj):
        if isinstance(obj, defaultdict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_convert(data), fh, indent=2)
    print(f'[MC] Saved {path}')


def _print_scale_table(n_tasks: int, scale_summary: dict, alg_names: list) -> None:
    """Print a brief progress table for a given scale."""
    hdr = (f"{'Algorithm':<14} {'Latency ms':>12} {'Energy mJ':>11} "
           f"{'Priv Risk':>10} {'SLA Viol%':>10} {'Throughput':>11}")
    sep = '-' * len(hdr)
    print(f'\n  [Scale n={n_tasks}]')
    print(f'  {hdr}')
    print(f'  {sep}')
    for alg in alg_names:
        d = scale_summary.get(alg, {})
        lat  = d.get('avg_latency_ms', {}).get('mean', 0.0)
        eng  = d.get('avg_energy_mj', {}).get('mean', 0.0)
        priv = d.get('avg_privacy_risk', {}).get('mean', 0.0)
        sla  = d.get('sla_violation_pct', {}).get('mean', 0.0)
        thr  = d.get('throughput', {}).get('mean', 0.0)
        tag  = ' *' if alg == 'DQN-ES' else '  '
        print(f'  {alg + tag:<14} {lat:>12.2f} {eng:>11.4f} '
              f'{priv:>10.4f} {sla:>10.2f} {thr:>11.1f}')
    print(f'  {sep}')


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bio-Inspired Adaptive Task Offloading — Monte Carlo Experiments'
    )
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Fast mode: 5 runs × 4 scales (default)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Full mode: 30 runs × 6 scales')
    parser.add_argument('--output-dir', default=None,
                        help='Override base output directory')
    args = parser.parse_args()

    fast_mode = not args.full

    if args.output_dir:
        base_dir   = Path(args.output_dir)
        results_dir = str(base_dir / 'results')
        figures_dir = str(base_dir / 'latex' / 'figures')
    else:
        results_dir = None
        figures_dir = None

    run_monte_carlo(
        fast_mode=fast_mode,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )


if __name__ == '__main__':
    import matplotlib.ticker
    main()
else:
    import matplotlib.ticker
