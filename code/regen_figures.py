"""Load saved mc_summary.json and regenerate all experiment figures."""
import sys, os, json
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))  # code/
ROOT = os.path.dirname(BASE)                        # project root
sys.path.insert(0, BASE)

from src.analysis.run_experiments import generate_all_figures

results_path = Path(ROOT) / 'results' / 'mc_summary.json'
figures_dir  = Path(ROOT) / 'latex' / 'figures'

print(f'[RegFig] Loading: {results_path}')
with open(results_path, 'r', encoding='utf-8') as f:
    raw = json.load(f)

# JSON keys are strings; convert to int
mc_summary = {int(k): v for k, v in raw.items()}
task_scales = sorted(mc_summary.keys())
print(f'[RegFig] Scales loaded: {task_scales}')

generate_all_figures(mc_summary, task_scales, figures_dir)
print('[RegFig] Done.')
