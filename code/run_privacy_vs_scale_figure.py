"""
run_privacy_vs_scale_figure.py — Generate Fig: Privacy risk vs task scale.

Reads results/mc_full_summary.json (must already exist) and writes
latex/figures/fig_privacy_vs_scale.pdf matching the style of Fig. 3
(latency vs N): line plot with ±1 SD shaded bands.

Run after run_psodqn_only.py so PSO+DQN trace is included.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'
FIGURES_DIR  = PROJECT_ROOT / 'latex' / 'figures'

ALGO_STYLE = {
    'DQN-ES':     {'color': '#1f77b4', 'lw': 2.5, 'ls': '-',  'marker': 'o', 'zorder': 5},
    'PSO+DQN':    {'color': '#ff7f0e', 'lw': 2.0, 'ls': '--', 'marker': 's', 'zorder': 4},
    'PSO':        {'color': '#2ca02c', 'lw': 1.5, 'ls': '-',  'marker': '^', 'zorder': 3},
    'HS-HHO':     {'color': '#d62728', 'lw': 1.5, 'ls': '-',  'marker': 'v', 'zorder': 3},
    'ES-only':    {'color': '#9467bd', 'lw': 1.5, 'ls': ':',  'marker': 'D', 'zorder': 3},
    'ACO':        {'color': '#8c564b', 'lw': 1.5, 'ls': '-',  'marker': 'x', 'zorder': 3},
    'DQN-only':   {'color': '#e377c2', 'lw': 1.5, 'ls': '--', 'marker': 'P', 'zorder': 3},
    'Cloud-Only': {'color': '#7f7f7f', 'lw': 1.0, 'ls': '-.',  'marker': None, 'zorder': 2},
    'Local-Only': {'color': '#bcbd22', 'lw': 1.0, 'ls': '-.',  'marker': None, 'zorder': 2},
}

PLOT_ORDER = ['DQN-ES', 'PSO+DQN', 'PSO', 'HS-HHO', 'ES-only', 'ACO', 'DQN-only',
              'Cloud-Only', 'Local-Only']


def main():
    mc_path = RESULTS_DIR / 'mc_full_summary.json'
    if not mc_path.exists():
        print(f'[ERROR] {mc_path} not found. Run the full MC pipeline first.')
        sys.exit(1)

    with open(mc_path) as fh:
        summary = json.load(fh)

    scales_str = sorted(summary.keys(), key=int)
    scales = [int(s) for s in scales_str]

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in PLOT_ORDER:
        means, stds = [], []
        found = False
        for s in scales_str:
            if algo in summary[s]:
                found = True
                m = summary[s][algo]['avg_privacy_risk']['mean']
                sd = summary[s][algo]['avg_privacy_risk']['std']
                means.append(m)
                stds.append(sd)
            else:
                means.append(None)
                stds.append(None)

        if not found:
            continue

        style = ALGO_STYLE.get(algo, {'color': 'k', 'lw': 1.5, 'ls': '-',
                                      'marker': None, 'zorder': 2})
        xs = [s for s, m in zip(scales, means) if m is not None]
        ys = [m for m in means if m is not None]
        sd_plot = [sd for sd in stds if sd is not None]

        ax.plot(xs, ys, label=algo, color=style['color'],
                linewidth=style['lw'], linestyle=style['ls'],
                marker=style['marker'], markersize=5, zorder=style['zorder'])
        ax.fill_between(xs,
                         [y - s for y, s in zip(ys, sd_plot)],
                         [y + s for y, s in zip(ys, sd_plot)],
                         color=style['color'], alpha=0.12, zorder=style['zorder'] - 1)

    ax.set_xlabel('Task scale $N$', fontsize=11)
    ax.set_ylabel('Mean privacy risk $R_P$', fontsize=11)
    ax.set_title('Privacy risk vs.\ task scale (30-run mean, $\\pm1$ SD)', fontsize=11)
    ax.set_xscale('log')
    ax.set_xticks(scales)
    ax.set_xticklabels([str(s) for s in scales])
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.grid(True, which='major', linestyle=':', alpha=0.4)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / 'fig_privacy_vs_scale.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] Saved {out_path}')

    # Print key values for the paper
    print('\nKey values for Section 5.5 text:')
    for algo in ['DQN-ES', 'PSO+DQN', 'PSO', 'HS-HHO']:
        row = []
        for s in scales_str:
            if algo in summary[s]:
                m = summary[s][algo]['avg_privacy_risk']['mean']
                sd = summary[s][algo]['avg_privacy_risk']['std']
                row.append(f'N={s}: {m:.4f}±{sd:.4f}')
        if row:
            print(f'  {algo}: {", ".join(row)}')


if __name__ == '__main__':
    main()
