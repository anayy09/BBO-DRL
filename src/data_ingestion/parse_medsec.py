"""
parse_medsec.py
---------------
Parse MedSec-25.csv (248 MB, 554 534 bidirectional network flows) in
memory-efficient chunks to extract IoMT attack flow events for privacy
risk validation.

MedSec-25 columns (detected from header):
    Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp,
    Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts,
    TotLen Bwd Pkts, ... (80+ features) ..., Label

Output: results/medsec_events.json  (≤ 10 000 stratified rows)
Public API: load_medsec_events(data_dir) -> list[dict]
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Attack-label → severity mapping (MedSec-25 campaign labels)
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    # Benign
    "benign":              0.0,
    # Reconnaissance
    "recon":               0.7,
    "reconnaissance":      0.7,
    "portscan":            0.65,
    "ping":                0.55,
    "scan":                0.65,
    # DoS / DDoS
    "dos":                 0.85,
    "ddos":                0.9,
    "flood":               0.85,
    "syn":                 0.8,
    "icmp":                0.8,
    "udp":                 0.75,
    "slowloris":           0.8,
    # Exploitation / Lateral movement
    "exploit":             0.92,
    "lateral":             0.95,
    "pivoting":            0.9,
    "mitm":                0.88,
    "man-in-the-middle":   0.88,
    # MQTT / IoT-specific
    "mqtt":                0.85,
    "spoofing":            0.85,
    "injection":           0.9,
    "malformed":           0.75,
    # Exfiltration
    "exfiltration":        0.93,
    "data theft":          0.93,
    # Ransomware / Malware
    "ransomware":          0.95,
    "malware":             0.9,
    # ARP
    "arp":                 0.8,
}
_DEFAULT_SEVERITY = 0.6


def _severity_from_label(label: str) -> float:
    """Map a MedSec-25 label to a threat severity score."""
    lower = str(label).lower()
    if "benign" in lower or lower in {"", "nan", "none", "-"}:
        return 0.0
    for key, sev in SEVERITY_MAP.items():
        if key in lower:
            return sev
    return _DEFAULT_SEVERITY


def _find_label_column(columns) -> str | None:
    """Return the column name most likely to hold the attack/flow label."""
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in ["label", "attack_type", "attack type", "class",
                  "category", "type", "tag", "attack"]:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _find_flow_columns(columns) -> dict:
    """
    Detect useful flow-level columns: bytes, duration, flow_id.
    Returns a mapping: canonical_name -> actual_column_name.
    """
    lower_map = {c.lower().strip(): c for c in columns}
    mapping = {}

    # Flow bytes (forward + backward total)
    for cand in ["totlen fwd pkts", "totlen_fwd_pkts", "fwd pkt len max",
                  "tot size", "tot_size", "flow byts/s", "flow bytes/s"]:
        if cand in lower_map:
            mapping["flow_bytes"] = lower_map[cand]
            break

    # Flow duration
    for cand in ["flow duration", "flow_duration", "duration", "dur"]:
        if cand in lower_map:
            mapping["flow_duration"] = lower_map[cand]
            break

    # Flow ID
    for cand in ["flow id", "flow_id", "flowid", "id"]:
        if cand in lower_map:
            mapping["flow_id_col"] = lower_map[cand]
            break

    return mapping


def _attack_probability(severity: float, flow_bytes, flow_duration) -> float:
    """
    Compute attack probability as severity × confidence.
    Confidence is derived from flow bytes and duration.
    """
    import math

    confidence = 0.5
    try:
        dur   = float(flow_duration) if flow_duration is not None else 0.0
        fbytes = float(flow_bytes)    if flow_bytes   is not None else 0.0
        # Duration confidence: longer flows → higher confidence (capped)
        dur_conf  = 1.0 / (1.0 + math.exp(-dur / 1e5))
        # Byte confidence: more bytes → higher confidence (log-normalised)
        byte_conf = min(1.0, math.log1p(abs(fbytes)) / 20.0)
        confidence = (dur_conf + byte_conf) / 2.0
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return round(max(0.0, min(1.0, severity * confidence)), 4)


# ---------------------------------------------------------------------------
# Stratified downsampling helper
# ---------------------------------------------------------------------------
def _stratified_sample(per_label: dict, max_total: int,
                        rng: random.Random) -> list:
    """
    Given a dict mapping label → list[dict], return a combined list of at
    most max_total rows sampled proportionally across labels.
    """
    total_available = sum(len(v) for v in per_label.values())
    if total_available <= max_total:
        result = []
        for rows in per_label.values():
            result.extend(rows)
        return result

    result = []
    labels = list(per_label.keys())
    for label in labels:
        rows = per_label[label]
        quota = max(1, int(max_total * len(rows) / total_available))
        sampled = rng.sample(rows, min(quota, len(rows)))
        result.extend(sampled)

    # Top up or trim to exactly max_total
    rng.shuffle(result)
    return result[:max_total]


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------
def parse_medsec(data_dir: str,
                 chunksize: int = 50_000,
                 max_output_rows: int = 10_000) -> list:
    """
    Parse MedSec-25.csv in chunks and return a stratified sample of flow
    events.

    Parameters
    ----------
    data_dir : str
        Root data directory that contains MedSec-25/MedSec-25.csv.
    chunksize : int
        Rows per pandas chunk (default 50 000).
    max_output_rows : int
        Maximum events in output list (default 10 000).

    Returns
    -------
    list of dict
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    base      = Path(data_dir)
    csv_path  = base / "MedSec-25" / "MedSec-25.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"[MedSec] File not found: {csv_path}")

    print(f"\n[MedSec] Parsing (chunked): {csv_path}")
    print(f"[MedSec] File size: {csv_path.stat().st_size / 1e6:.1f} MB")

    rng              = random.Random(42)
    label_col        = None
    flow_cols        = {}
    header_printed   = False
    per_label        = defaultdict(list)   # label → list of event dicts
    global_row_id    = 0
    chunks_processed = 0

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunksize,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False,
        )
    except Exception as exc:
        try:
            reader = pd.read_csv(
                csv_path,
                chunksize=chunksize,
                encoding="latin-1",
                on_bad_lines="skip",
                low_memory=False,
            )
        except Exception as exc2:
            raise RuntimeError(
                f"[MedSec] Cannot open CSV: {exc2}"
            ) from exc2

    for chunk in reader:
        chunks_processed += 1

        # First chunk: inspect columns
        if not header_printed:
            print(f"[MedSec] Columns ({len(chunk.columns)}):")
            print(f"  {list(chunk.columns)}\n")
            label_col = _find_label_column(chunk.columns)
            flow_cols = _find_flow_columns(chunk.columns)
            print(f"[MedSec] Label column  : {label_col}")
            print(f"[MedSec] Flow columns  : {flow_cols}")
            header_printed = True

        for _, row in chunk.iterrows():
            # Label / attack type
            label    = str(row[label_col]).strip() if label_col else "Unknown"
            severity = _severity_from_label(label)

            # Flow bytes & duration
            fb = row[flow_cols["flow_bytes"]]   if "flow_bytes"   in flow_cols else None
            fd = row[flow_cols["flow_duration"]] if "flow_duration" in flow_cols else None

            # Native flow ID
            if "flow_id_col" in flow_cols:
                try:
                    native_id = str(row[flow_cols["flow_id_col"]])
                except Exception:
                    native_id = str(global_row_id)
            else:
                native_id = str(global_row_id)

            attack_prob = _attack_probability(severity, fb, fd)

            event = {
                "flow_id":          global_row_id,
                "native_flow_id":   native_id,
                "label":            label,
                "severity":         severity,
                "attack_probability": attack_prob,
                "flow_bytes":       float(fb)  if fb  is not None else 0.0,
                "flow_duration":    float(fd)  if fd  is not None else 0.0,
            }

            per_label[label].append(event)
            global_row_id += 1

        if chunks_processed % 5 == 0:
            total_so_far = sum(len(v) for v in per_label.values())
            print(f"[MedSec]   Processed chunk {chunks_processed} "
                  f"— {total_so_far:,} rows read so far …")

    total_read = sum(len(v) for v in per_label.values())
    print(f"\n[MedSec] Total rows read : {total_read:,}")
    print(f"[MedSec] Unique labels   : {sorted(per_label.keys())}")

    # Stratified downsampling
    events = _stratified_sample(per_label, max_output_rows, rng)
    print(f"[MedSec] After stratified sampling: {len(events)} events "
          f"(target ≤ {max_output_rows})")
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_medsec_events(data_dir: str) -> list:
    """
    Load (or parse and cache) MedSec-25 simulation events.

    Tries results/medsec_events.json first; falls back to raw CSV parsing.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains MedSec-25/ subdirectory).

    Returns
    -------
    list of dict
    """
    results_dir = Path(data_dir).parent / "results"
    cache_path  = results_dir / "medsec_events.json"

    if cache_path.exists():
        print(f"[MedSec] Loading cached events from {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[MedSec] Cache read failed ({exc}); re-parsing.")

    return parse_medsec(data_dir)


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
    print("MedSec-25 Data Ingestion")
    print("=" * 60)

    try:
        events = parse_medsec(data_dir)
    except Exception as exc:
        print(f"[MedSec] ERROR: {exc}")
        sys.exit(1)

    out_path = results_dir / "medsec_events.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2, default=str)
        print(f"[MedSec] Saved {len(events)} events to {out_path}")
    except Exception as exc:
        print(f"[MedSec] Failed to save JSON: {exc}")
        sys.exit(1)

    if events:
        print("\n[MedSec] Sample event (index 0):")
        for k, v in events[0].items():
            print(f"  {k}: {v}")
