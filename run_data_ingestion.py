"""Test real dataset ingestion pipeline."""
import sys, os
BASE = r"c:\Users\sinha\OneDrive - University of Florida\Papers\Bio-Inspired Adaptive Task Offloading System"
DATA_DIR = os.path.join(BASE, "data")
sys.path.insert(0, BASE)

from src.data_ingestion.parse_mendeley import load_mendeley_events
from src.data_ingestion.parse_ciciot import load_ciciot_events
from src.data_ingestion.parse_medsec import load_medsec_events
from src.data_ingestion.parse_mitbih import load_mitbih_events
from src.data_ingestion.event_generator import generate_event_stream

print("=" * 60)
print("DATASET INGESTION TEST")
print("=" * 60)

print("\n[1/4] Loading Mendeley IoMT dataset...")
mendeley = load_mendeley_events(DATA_DIR)
print(f"  -> {len(mendeley)} patient events loaded")
if mendeley:
    sample = mendeley[0]
    print(f"  -> Sample: CI={sample['ci_score']:.3f}, task={sample.get('task_type','?')}, D={sample.get('data_size_bits',0)//1000}Kbits")
    ci_vals = [e['ci_score'] for e in mendeley]
    high_ci = sum(1 for c in ci_vals if c > 0.7)
    print(f"  -> High CI (>0.7): {high_ci}/{len(mendeley)} ({100*high_ci/max(len(mendeley),1):.1f}%)")

print("\n[2/4] Loading MIT-BIH Arrhythmia dataset...")
mitbih = load_mitbih_events(DATA_DIR)
print(f"  -> {len(mitbih)} arrhythmia window events loaded")
if mitbih:
    sample = mitbih[0]
    print(f"  -> Sample: record={sample.get('record_id','?')}, CI={sample.get('ci_score',0):.3f}, arrhythmia_ratio={sample.get('arrhythmia_ratio',0):.3f}")

print("\n[3/4] Loading CICIoMT2024 dataset...")
ciciot = load_ciciot_events(DATA_DIR)
print(f"  -> {len(ciciot)} network threat events loaded")
if ciciot:
    sample = ciciot[0]
    print(f"  -> Sample: attack={sample.get('attack_type','?')}, severity={sample.get('threat_severity',0):.2f}, prob={sample.get('attack_probability',0):.2f}")

print("\n[4/4] Loading MedSec-25 dataset (chunked)...")
medsec = load_medsec_events(DATA_DIR)
print(f"  -> {len(medsec)} IoMT attack flow events loaded")
if medsec:
    labels = {}
    for e in medsec:
        labels[e.get('label','?')] = labels.get(e.get('label','?'), 0) + 1
    print(f"  -> Attack label distribution: {dict(list(labels.items())[:5])}")

print("\n[STREAM] Generating unified event stream (2000 tasks)...")
stream = generate_event_stream(
    n_tasks=2000, n_devices=20,
    mendeley_events=mendeley,
    mitbih_events=mitbih,
    ciciot_events=ciciot,
    seed=42
)
print(f"  -> {len(stream)} tasks in stream")
ci_scores = [t.ci_score for t in stream]
print(f"  -> CI: min={min(ci_scores):.3f}, mean={sum(ci_scores)/len(ci_scores):.3f}, max={max(ci_scores):.3f}")
sources = {}
for t in stream:
    sources[t.source] = sources.get(t.source, 0) + 1
print(f"  -> Sources: {sources}")

print("\nData ingestion pipeline: COMPLETE")
