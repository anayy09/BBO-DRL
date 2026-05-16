"""
run_q1_pipeline.py — Q1 elevation orchestrator.

Calls every step required by docs/Fix.md, in order:

  Step 1  full Monte Carlo (Fix 1, 2, 3, 5, 7)
  Step 2  Bonferroni-corrected pairwise tests (Fix 4)
  Step 3  CI weight ablation (Fix 6)
  Step 4  Privacy Guard validation on MedSec-25 (Fix 8)
  Step 5  MIT-BIH real-trace evaluation (Fix 10)
  Step 6  Figures (Fix 5, 7, 9)

Each step writes its outputs to results/ and latex/figures/.
The user can skip steps via flags, e.g. --skip-mitbih.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))   # code/
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'
FIGURES_DIR  = PROJECT_ROOT / 'latex' / 'figures'
DATA_DIR     = PROJECT_ROOT / 'data'


def _header(msg):
    print('\n' + '=' * 72)
    print(f'  {msg}')
    print('=' * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs',   type=int, default=30,
                   help='Monte Carlo runs per cell (default 30)')
    p.add_argument('--scales',   type=int, nargs='+', default=None,
                   help='Task scales (default 100 500 1000 2000 5000)')
    p.add_argument('--skip-mc',           action='store_true')
    p.add_argument('--skip-stats',        action='store_true')
    p.add_argument('--skip-weight',       action='store_true')
    p.add_argument('--skip-privacy',      action='store_true')
    p.add_argument('--skip-mitbih',       action='store_true')
    p.add_argument('--skip-figures',      action='store_true')
    args = p.parse_args()

    t0 = time.time()

    if not args.skip_mc:
        _header('Step 1/6 — Monte Carlo (Fix 1, 2, 3, 5, 7)')
        from src.analysis.run_full_experiments import run_full
        from src.config import TASK_SCALES
        run_full(
            task_scales=args.scales or TASK_SCALES,
            n_runs=args.n_runs,
            results_dir=RESULTS_DIR,
        )

    if not args.skip_stats:
        _header('Step 2/6 — Pairwise Wilcoxon + Bonferroni (Fix 4)')
        from src.analysis.statistical_tests import run_pairwise_tests
        run_pairwise_tests(RESULTS_DIR / 'mc_full_summary.json', RESULTS_DIR)

    if not args.skip_weight:
        _header('Step 3/6 — CI weight ablation (Fix 6)')
        from src.analysis.weight_ablation import run_ablation
        from src.config import PRIMARY_SCALE
        run_ablation(PRIMARY_SCALE, args.n_runs, RESULTS_DIR)

    if not args.skip_privacy:
        _header('Step 4/6 — Privacy Guard on MedSec-25 (Fix 8)')
        from src.analysis.privacy_guard import run_validation
        run_validation(DATA_DIR, RESULTS_DIR, FIGURES_DIR)

    if not args.skip_mitbih:
        _header('Step 5/6 — MIT-BIH real-trace evaluation (Fix 10)')
        from src.analysis.mitbih_trace_eval import run_mitbih_trace
        run_mitbih_trace(DATA_DIR, RESULTS_DIR, n_runs=args.n_runs)

    if not args.skip_figures:
        _header('Step 6/6 — Publication figures (Fix 5, 7, 9)')
        from src.analysis.figures_q1 import (
            fig1_latency_vs_scale,
            fig2_metric_comparison,
            fig3_pareto_energy_latency,
            fig3b_pareto_latency_privacy,
            fig_epsilon_convergence,
            _load_summary,
        )
        from src.config import PRIMARY_SCALE
        summary = _load_summary(RESULTS_DIR / 'mc_full_summary.json')
        scales = sorted(summary.keys())
        fig1_latency_vs_scale(summary, FIGURES_DIR, scales=scales)
        fig2_metric_comparison(summary, FIGURES_DIR, ref_scale=PRIMARY_SCALE)
        fig3_pareto_energy_latency(summary, FIGURES_DIR,
                                   ref_scale=PRIMARY_SCALE)
        fig3b_pareto_latency_privacy(summary, FIGURES_DIR,
                                     ref_scale=PRIMARY_SCALE)
        fig_epsilon_convergence(RESULTS_DIR / 'epsilon_trajectory.json',
                                FIGURES_DIR)

    dt = time.time() - t0
    _header(f'Q1 pipeline complete in {dt/60:.1f} min')
    print('  Inspect:')
    print(f'   - {RESULTS_DIR}/mc_full_summary.json')
    print(f'   - {RESULTS_DIR}/table3_n1000_with_pvals.csv')
    print(f'   - {RESULTS_DIR}/table5_weight_ablation.csv')
    print(f'   - {RESULTS_DIR}/table5_mitbih_trace.csv')
    print(f'   - {RESULTS_DIR}/privacy_guard_metrics.json')
    print(f'   - {FIGURES_DIR}/fig*.pdf')


if __name__ == '__main__':
    main()
