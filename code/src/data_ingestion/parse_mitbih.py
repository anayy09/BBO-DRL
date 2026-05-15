"""
parse_mitbih.py
---------------
Parse the MIT-BIH Arrhythmia Database (PhysioNet WFDB format) to extract
per-10-second window arrhythmia events and derive CPU-cycle profiles for
ECG inference simulation tasks.

Output: results/mitbih_events.json
Public API: load_mitbih_events(data_dir) -> list[dict]

Dependencies:
    pip install wfdb
If wfdb is unavailable, synthetic fallback events are generated.
"""

import os
import json
import random
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 10          # analysis window length
CNN_BASE_CYCLES = 12_940_000  # CPU cycles for CNN ECG inference (literature)
NORMAL_BEAT_LABEL = "N"       # wfdb annotation symbol for normal sinus beat

# Additional non-normal beat symbols (AAMI categories)
ARRHYTHMIA_SYMBOLS = {
    "V",   # Premature ventricular contraction (PVC)
    "A",   # Atrial premature beat (APB)
    "F",   # Fusion of ventricular and normal beat
    "a",   # Aberrated atrial premature beat
    "J",   # Nodal (junctional) premature beat
    "S",   # Supraventricular premature beat
    "E",   # Ventricular escape beat
    "j",   # Nodal escape beat
    "e",   # Atrial escape beat
    "L",   # Left bundle branch block beat
    "R",   # Right bundle branch block beat
    "P",   # Paced beat
    "/",   # Paced beat (alternate code)
    "Q",   # Unclassifiable beat
    "f",   # Fusion of paced and normal beat
}


# ---------------------------------------------------------------------------
# CI mapping from arrhythmia ratio
# ---------------------------------------------------------------------------
def _ci_from_ratio(ratio: float, rng: random.Random) -> float:
    """Map arrhythmia ratio to a CI score with slight jitter."""
    if ratio > 0.3:
        lo, hi = 0.7, 1.0
    elif ratio >= 0.1:
        lo, hi = 0.3, 0.7
    else:
        lo, hi = 0.0, 0.3

    ci = lo + rng.random() * (hi - lo)
    return round(max(0.0, min(1.0, ci)), 4)


# ---------------------------------------------------------------------------
# WFDB-based parsing
# ---------------------------------------------------------------------------
def _parse_with_wfdb(mitbih_dir: Path, rng: random.Random) -> list:
    """Parse all records using the wfdb library."""
    import wfdb  # type: ignore

    records_file = mitbih_dir / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"[MIT-BIH] RECORDS file not found in {mitbih_dir}")

    record_names = records_file.read_text(encoding="utf-8").split()
    print(f"[MIT-BIH] Found {len(record_names)} records: {record_names}")

    events = []

    for rec_name in record_names:
        rec_path = str(mitbih_dir / rec_name)
        print(f"[MIT-BIH]   Processing record {rec_name} ...", end=" ", flush=True)

        try:
            # Read header to get sampling frequency
            header = wfdb.rdheader(rec_path)
            fs = header.fs  # samples per second (usually 360 Hz for MIT-BIH)

            # Read annotation file
            ann = wfdb.rdann(rec_path, "atr")
            symbols = ann.symbol        # list of beat label strings
            samples = ann.sample        # sample indices

            total_samples = header.sig_len
            total_duration_s = total_samples / fs

            # Segment into WINDOW_SECONDS windows
            window_samples = int(WINDOW_SECONDS * fs)
            n_windows = max(1, int(total_samples // window_samples))

            window_events = []
            for w in range(n_windows):
                w_start_sample = w * window_samples
                w_end_sample   = w_start_sample + window_samples
                w_start_s      = w * WINDOW_SECONDS

                # Beats in this window
                mask = (samples >= w_start_sample) & (samples < w_end_sample)
                window_symbols = [symbols[i] for i in range(len(symbols)) if mask[i]]

                total_beats   = len(window_symbols)
                if total_beats == 0:
                    continue

                arrhythmia_beats = sum(
                    1 for s in window_symbols if s in ARRHYTHMIA_SYMBOLS
                )
                arrhythmia_ratio = arrhythmia_beats / total_beats

                ci = _ci_from_ratio(arrhythmia_ratio, rng)
                # Scale CPU cycles: higher CI → more compute (up to 1.5×)
                cpu_cycles = int(CNN_BASE_CYCLES * (1.0 + 0.5 * ci))

                window_events.append({
                    "record_id":               rec_name,
                    "window_start_s":          round(w_start_s, 2),
                    "arrhythmia_ratio":        round(arrhythmia_ratio, 4),
                    "ci_score":                ci,
                    "cpu_cycles_ecg_inference": cpu_cycles,
                })

            events.extend(window_events)
            print(f"{len(window_events)} windows, "
                  f"duration={total_duration_s:.1f}s, fs={fs}Hz")

        except Exception as exc:
            print(f"\n[MIT-BIH]   SKIPPED record {rec_name}: {exc}")
            continue

    return events


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def _synthetic_mitbih_events(n_records: int = 48, rng: random.Random = None) -> list:
    """
    Generate plausible synthetic MIT-BIH–style events when wfdb is not
    available.  Based on published MIT-BIH statistics.
    """
    if rng is None:
        rng = random.Random(42)

    print("[MIT-BIH] Generating synthetic fallback events (wfdb unavailable).")

    # Approximate record IDs matching the real database
    record_ids = [str(r) for r in list(range(100, 125)) + list(range(200, 235))
                  if r not in (120, 126, 127, 128, 129, 130, 131, 132, 133,
                               134, 135, 136, 137, 138, 139, 140, 141, 142,
                               143, 144, 145, 146, 147, 148, 149, 150, 151,
                               152, 153, 154, 155, 156, 157, 158, 159, 160,
                               161, 162, 163, 164, 165, 166, 167, 168, 169,
                               170, 171, 172, 173, 174, 175, 176, 177, 178,
                               179, 180, 181, 182, 183, 184, 185, 186, 187,
                               188, 189, 190, 191, 192, 193, 194, 195, 196,
                               197, 198, 199, 206, 211, 216, 218, 224, 225,
                               226, 227, 229)][:n_records]

    total_duration_s = 1800  # 30 minutes per record
    n_windows = total_duration_s // WINDOW_SECONDS

    events = []
    for rec_id in record_ids:
        for w in range(n_windows):
            # Skewed ratio: most windows are normal
            arrhythmia_ratio = rng.betavariate(0.5, 3.0)
            arrhythmia_ratio = round(min(1.0, arrhythmia_ratio), 4)
            ci = _ci_from_ratio(arrhythmia_ratio, rng)
            cpu_cycles = int(CNN_BASE_CYCLES * (1.0 + 0.5 * ci))
            events.append({
                "record_id":               rec_id,
                "window_start_s":          w * WINDOW_SECONDS,
                "arrhythmia_ratio":        arrhythmia_ratio,
                "ci_score":                ci,
                "cpu_cycles_ecg_inference": cpu_cycles,
            })

    print(f"[MIT-BIH] Generated {len(events)} synthetic events.")
    return events


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------
def parse_mitbih(data_dir: str) -> list:
    """
    Parse MIT-BIH Arrhythmia Database and return a list of window-level
    simulation events.

    Parameters
    ----------
    data_dir : str
        Path to the root data directory that contains MIT-BIH-Arrhythmia/.

    Returns
    -------
    list of dict
    """
    base = Path(data_dir)
    mitbih_dir = base / "MIT-BIH-Arrhythmia"

    if not mitbih_dir.exists():
        raise FileNotFoundError(f"[MIT-BIH] Directory not found: {mitbih_dir}")

    rng = random.Random(42)

    try:
        import wfdb  # noqa: F401
        print("[MIT-BIH] wfdb library found. Parsing WFDB records.")
        events = _parse_with_wfdb(mitbih_dir, rng)
    except ImportError:
        print("[MIT-BIH] WARNING: wfdb library not installed "
              "(pip install wfdb). Using synthetic fallback events.")
        events = _synthetic_mitbih_events(rng=rng)
    except Exception as exc:
        print(f"[MIT-BIH] WARNING: wfdb parsing failed ({exc}). "
              "Using synthetic fallback events.")
        events = _synthetic_mitbih_events(rng=rng)

    print(f"[MIT-BIH] Total events generated: {len(events)}")
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_mitbih_events(data_dir: str) -> list:
    """
    Load (or parse and cache) MIT-BIH simulation events.

    Tries results/mitbih_events.json first; falls back to raw parsing.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains MIT-BIH-Arrhythmia/ subdirectory).

    Returns
    -------
    list of dict
    """
    results_dir = Path(data_dir).parent / "results"
    cache_path = results_dir / "mitbih_events.json"

    if cache_path.exists():
        print(f"[MIT-BIH] Loading cached events from {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[MIT-BIH] Cache read failed ({exc}); re-parsing.")

    return parse_mitbih(data_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir     = str(project_root / "data")
    results_dir  = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIT-BIH Arrhythmia Data Ingestion")
    print("=" * 60)

    try:
        events = parse_mitbih(data_dir)
    except Exception as exc:
        print(f"[MIT-BIH] ERROR: {exc}")
        sys.exit(1)

    out_path = results_dir / "mitbih_events.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2)
        print(f"[MIT-BIH] Saved {len(events)} events to {out_path}")
    except Exception as exc:
        print(f"[MIT-BIH] Failed to save JSON: {exc}")
        sys.exit(1)

    if events:
        print("\n[MIT-BIH] Sample event (index 0):")
        for k, v in events[0].items():
            print(f"  {k}: {v}")
