"""
statistical_tests.py — Fix 4.

Pairwise Wilcoxon rank-sum (Mann-Whitney U) tests between DQN-ES and each
baseline on the four headline metrics, using the 30 per-run samples from
mc_full_summary.json (at PRIMARY_SCALE).  Applies Bonferroni correction
across the (n_baselines × n_metrics) family of comparisons.

Outputs:
  results/table3_stat_tests.csv         pairwise p, significant flag
  results/table3_n1000_with_pvals.csv   Table III formatted with p-values
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.config import (
    PRIMARY_SCALE,
    STAT_ALPHA,
    STAT_BASELINES,
    STAT_METRICS,
)


def _run_test_stats(payload):
    m, b, s1, s2 = payload
    if len(s1) == 0 or len(s2) == 0: return m, b, float('nan')
    try:
        from scipy import stats
        _, p_raw = stats.ranksums(s1, s2)
        return m, b, float(p_raw)
    except Exception:
        return m, b, float('nan')


def run_pairwise_tests(summary_path: Path, out_dir: Path,
                       scale: int = PRIMARY_SCALE, workers: int | None = None) -> dict:
    """
    Run Wilcoxon rank-sum tests for each (baseline, metric) pair against
    DQN-ES, then apply Bonferroni correction.

    Returns
    -------
    dict keyed by metric -> {baseline: {'p_raw', 'p_corrected', 'significant'}}
    """
    with open(summary_path, 'r', encoding='utf-8') as fh:
        summary = json.load(fh)

    scale_key = str(scale)
    if scale_key not in summary:
        raise KeyError(f'Scale {scale} not in summary file.')

    cell = summary[scale_key]
    if 'DQN-ES' not in cell:
        raise KeyError('DQN-ES not present in summary file.')

    n_comparisons = len(STAT_BASELINES) * len(STAT_METRICS)
    alpha_corr = STAT_ALPHA / n_comparisons

    results = {}
    csv_rows = []

    import concurrent.futures
    payloads = []
    for metric in STAT_METRICS:
        results[metric] = {}
        if 'DQN-ES' not in cell or metric not in cell['DQN-ES']:
            continue
        bbo_samples = np.array(cell['DQN-ES'][metric]['samples'], dtype=float)
        for base in STAT_BASELINES:
            if base not in cell or metric not in cell[base]:
                continue
            base_samples = np.array(cell[base][metric]['samples'], dtype=float)
            payloads.append((metric, base, bbo_samples, base_samples))
    
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for m, b, p_raw in executor.map(_run_test_stats, payloads):
            p_corr = min(p_raw * n_comparisons, 1.0) if not np.isnan(p_raw) else float('nan')
            sig = (not np.isnan(p_corr)) and (p_corr < STAT_ALPHA)
            results[m][b] = {
                'p_raw': float(p_raw),
                'p_corrected': float(p_corr),
                'significant': bool(sig),
            }
            csv_rows.append({
                'metric': m,
                'baseline': b,
                'p_raw': p_raw,
                'p_corrected': p_corr,
                'significant_after_bonferroni': sig,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'table3_stat_tests.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(csv_rows)
    print(f'[STAT] Saved {csv_path}')

    # ----- Pretty Table III with per-comparison p-values (Fix E) -----
    # Format: mean ± std (p=X.XXe-Y vs DQN-ES) per Fix E requirement
    fmt_path = out_dir / 'table3_n1000_with_pvals.csv'
    with open(fmt_path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow([
            'algorithm',
            *[f'{m}_mean±std (p_corr vs DQN-ES)' for m in STAT_METRICS],
        ])
        for alg in ['DQN-ES'] + STAT_BASELINES:
            if alg not in cell:
                continue
            row = [alg]
            for m in STAT_METRICS:
                d = cell[alg][m]
                mu, sd = d['mean'], d['std']
                if alg == 'DQN-ES':
                    row.append(f'{mu:.3f} ± {sd:.3f}')
                else:
                    pc = results[m].get(alg, {}).get('p_corrected',
                                                     float('nan'))
                    star = '*' if (not np.isnan(pc) and pc < STAT_ALPHA) else ''
                    if np.isnan(pc):
                        p_str = 'p=nan'
                    elif pc < 1e-3:
                        p_str = f'p={pc:.2e}'   # scientific notation for small p
                    else:
                        p_str = f'p={pc:.4f}'
                    row.append(f'{mu:.3f} ± {sd:.3f} ({p_str}{star})')
            wr.writerow(row)
        # Footnote — Fix E: state family size and correction method explicitly
        wr.writerow([])
        wr.writerow([
            f'Footnote: p_corrected = Wilcoxon rank-sum p × {n_comparisons} '
            f'(Bonferroni correction, family size = {len(STAT_BASELINES)} baselines '
            f'× {len(STAT_METRICS)} metrics = {n_comparisons} tests), '
            f'alpha={STAT_ALPHA}. '
            f'* indicates p_corr < {STAT_ALPHA}. '
            f'Baselines: {", ".join(STAT_BASELINES)}.',
        ])
    print(f'[STAT] Saved {fmt_path}')

    # Console digest
    print('\n[STAT] Bonferroni-corrected pairwise tests vs DQN-ES '
          f'(scale={scale}, alpha={STAT_ALPHA}, '
          f'family size={n_comparisons}):')
    for m in STAT_METRICS:
        print(f'  {m}')
        for base in STAT_BASELINES:
            r = results[m].get(base)
            if not r:
                continue
            tag = ' *' if r['significant'] else '  '
            print(f'    vs {base:<10s} p_corr={r["p_corrected"]:.4f}{tag}')

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=str, default=None,
                   help='Path to mc_full_summary.json')
    p.add_argument('--scale', type=int, default=PRIMARY_SCALE)
    p.add_argument('--output', type=str, default=None,
                   help='Output directory for CSVs')
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    summary_path = (Path(args.summary) if args.summary
                    else project_root / 'results' / 'mc_full_summary.json')
    out_dir = (Path(args.output) if args.output
               else project_root / 'results')
    run_pairwise_tests(summary_path, out_dir, scale=args.scale)


if __name__ == '__main__':
    main()
