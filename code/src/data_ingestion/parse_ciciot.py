"""
parse_ciciot.py
---------------
Parse CICIoMT2024 dataset CSVs from the attacks/ sub-directories to generate
network threat events that modulate privacy risk scores in the simulation.

Only the WiFI_and_MQTT/attacks/csv/ split is in CSV format; Bluetooth
attacks are stored as PCAP and are therefore skipped with a note.

Output: results/ciciot_events.json
Public API: load_ciciot_events(data_dir) -> list[dict]
"""

import os
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Attack-type → threat severity mapping
# ---------------------------------------------------------------------------
# Keys are matched as case-insensitive substrings of the filename stem or
# a label column value.
ATTACK_SEVERITY_MAP = {
    "ddos":               0.9,
    "dos":                0.85,
    "recon":              0.7,
    "reconnaissance":     0.7,
    "mqtt_spoofing":      0.85,
    "mqtt-spoofing":      0.85,
    "mqtt_malformed":     0.75,
    "mqtt-malformed":     0.75,
    "arp_spoofing":       0.8,
    "arp-spoofing":       0.8,
    "lateral_movement":   0.95,
    "lateral-movement":   0.95,
    "benign":             0.0,
}

_UNKNOWN_SEVERITY = 0.6


def _severity_from_name(name: str) -> float:
    """Infer threat severity from a filename stem or label string."""
    lower = name.lower()
    for key, sev in ATTACK_SEVERITY_MAP.items():
        if key in lower:
            return sev
    return _UNKNOWN_SEVERITY


def _infer_protocol(csv_path: Path) -> str:
    """Determine protocol from directory structure."""
    parts = [p.lower() for p in csv_path.parts]
    if "bluetooth" in parts:
        return "Bluetooth"
    if "wifi" in parts or "wifi_and_mqtt" in parts or "wifi" in " ".join(parts):
        return "WiFi_MQTT"
    return "Unknown"


def _normalise_attack_label(label) -> str:
    """Return a canonical attack label string."""
    s = str(label).strip()
    # Replace common separators for uniform matching
    return s.replace("_", " ").replace("-", " ").title()


def _identify_label_column(columns):
    """Return the name of the column most likely to contain attack labels."""
    lower_cols = {c.lower(): c for c in columns}
    for candidate in ["label", "attack_type", "attack type", "class",
                       "category", "type", "attack", "tag"]:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def _identify_numeric_columns(df):
    """Return a list of numeric column names useful as flow features."""
    import pandas as pd
    return df.select_dtypes(include=["number"]).columns.tolist()


def _confidence_from_flow(row, numeric_cols: list) -> float:
    """
    Compute a simple [0,1] confidence proxy from flow duration and byte ratio.
    Uses:
      - Duration (higher duration → more certain long-lived attack flow)
      - Byte-rate columns (extreme values suggest clear anomalies)
    Falls back to 0.5 if no useful columns are found.
    """
    import pandas as pd
    import math

    duration_col = None
    bytes_col    = None

    nc_lower = {c.lower(): c for c in numeric_cols}
    for cand in ["duration", "flow duration", "dur"]:
        if cand in nc_lower:
            duration_col = nc_lower[cand]
            break
    for cand in ["rate", "tot size", "tot_size", "bytes", "totlen fwd pkts",
                  "flow byts/s", "flow bytes/s"]:
        if cand in nc_lower:
            bytes_col = nc_lower[cand]
            break

    conf = 0.5
    try:
        if duration_col:
            dur = float(row[duration_col])
            # Sigmoid-normalise: long flows (>1000 units) get confidence → 1
            conf = 1.0 / (1.0 + math.exp(-dur / 100))
        if bytes_col:
            bval = float(row[bytes_col])
            byte_conf = min(1.0, abs(bval) / 1e6)
            conf = (conf + byte_conf) / 2.0
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return round(max(0.0, min(1.0, conf)), 4)


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------
def parse_ciciot(data_dir: str, max_rows_per_file: int = 5_000) -> list:
    """
    Walk CICIoMT2024 attacks/ subdirectories and parse CSV files.

    Parameters
    ----------
    data_dir : str
        Root data directory that contains CICIoMT2024/.
    max_rows_per_file : int
        Maximum rows to sample from each CSV (default 5 000) to cap memory.

    Returns
    -------
    list of dict
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    base = Path(data_dir)
    ciciot_root = base / "CICIoMT2024"

    if not ciciot_root.exists():
        raise FileNotFoundError(f"[CICIoT] Directory not found: {ciciot_root}")

    # Find all CSV files inside attacks/ sub-trees
    csv_files = sorted([
        p for p in ciciot_root.rglob("*.csv")
        if "attacks" in [part.lower() for part in p.parts]
    ])

    if not csv_files:
        print("[CICIoT] WARNING: No CSV files found under attacks/ subdirectories.")
        return []

    print(f"[CICIoT] Found {len(csv_files)} CSV file(s) under attacks/:")
    for f in csv_files:
        print(f"  {f.relative_to(ciciot_root)}")

    rng = random.Random(42)
    events = []
    global_flow_id = 0

    for csv_path in csv_files:
        protocol   = _infer_protocol(csv_path)
        # Derive attack type from filename (e.g. "ARP_Spoofing_test.pcap.csv")
        stem       = csv_path.stem.replace(".pcap", "")          # strip .pcap suffix
        # Remove train/test suffix for cleaner label
        for suffix in ["_train", "_test", "_Train", "_Test"]:
            stem = stem.replace(suffix, "")
        file_attack = stem                                        # e.g. "ARP_Spoofing"
        file_severity = _severity_from_name(file_attack)

        print(f"\n[CICIoT] Parsing {csv_path.name}  "
              f"(protocol={protocol}, file_severity={file_severity})", flush=True)

        try:
            # Read a sample to inspect columns
            df_head = pd.read_csv(csv_path, nrows=2, encoding="utf-8",
                                  on_bad_lines="skip")
        except UnicodeDecodeError:
            try:
                df_head = pd.read_csv(csv_path, nrows=2, encoding="latin-1",
                                      on_bad_lines="skip")
            except Exception as exc:
                print(f"[CICIoT]   SKIPPED (encoding error): {exc}")
                continue
        except Exception as exc:
            print(f"[CICIoT]   SKIPPED (read error): {exc}")
            continue

        print(f"[CICIoT]   Columns: {list(df_head.columns)}")

        label_col   = _identify_label_column(df_head.columns)
        print(f"[CICIoT]   Label column detected: {label_col}")

        # Now read the full file (up to max_rows_per_file)
        try:
            encoding = "utf-8"
            try:
                df = pd.read_csv(csv_path, encoding=encoding,
                                 on_bad_lines="skip")
            except UnicodeDecodeError:
                encoding = "latin-1"
                df = pd.read_csv(csv_path, encoding=encoding,
                                 on_bad_lines="skip")

            if len(df) > max_rows_per_file:
                df = df.sample(n=max_rows_per_file, random_state=42)

        except Exception as exc:
            print(f"[CICIoT]   SKIPPED (full-read error): {exc}")
            continue

        numeric_cols = _identify_numeric_columns(df)

        for _, row in df.iterrows():
            # Attack type & severity
            if label_col and label_col in df.columns:
                raw_label    = str(row[label_col])
                attack_type  = _normalise_attack_label(raw_label)
                severity     = _severity_from_name(raw_label)
            else:
                attack_type  = _normalise_attack_label(file_attack)
                severity     = file_severity

            # Confidence proxy from flow features
            confidence       = _confidence_from_flow(row, numeric_cols)
            attack_prob      = round(severity * confidence, 4)

            # Relative timestamp: use a numeric column if available, else sequential
            ts_col = None
            for cand in ["timestamp", "time", "iat", "duration", "flow duration"]:
                nc_lower = {c.lower(): c for c in numeric_cols}
                if cand in nc_lower:
                    ts_col = nc_lower[cand]
                    break
            if ts_col:
                try:
                    ts_relative = float(row[ts_col])
                except (TypeError, ValueError):
                    ts_relative = float(global_flow_id)
            else:
                ts_relative = float(global_flow_id)

            events.append({
                "flow_id":           global_flow_id,
                "protocol":          protocol,
                "attack_type":       attack_type,
                "threat_severity":   severity,
                "attack_probability": attack_prob,
                "timestamp_relative": ts_relative,
            })
            global_flow_id += 1

        print(f"[CICIoT]   Added {len(df)} flow events from {csv_path.name}")

    print(f"\n[CICIoT] Total events generated: {len(events)}")
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_ciciot_events(data_dir: str) -> list:
    """
    Load (or parse and cache) CICIoMT2024 threat events.

    Tries results/ciciot_events.json first; falls back to raw CSV parsing.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains CICIoMT2024/ subdirectory).

    Returns
    -------
    list of dict
    """
    results_dir = Path(data_dir).parent / "results"
    cache_path  = results_dir / "ciciot_events.json"

    if cache_path.exists():
        print(f"[CICIoT] Loading cached events from {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[CICIoT] Cache read failed ({exc}); re-parsing.")

    return parse_ciciot(data_dir)


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
    print("CICIoMT2024 Data Ingestion")
    print("=" * 60)

    try:
        events = parse_ciciot(data_dir)
    except Exception as exc:
        print(f"[CICIoT] ERROR: {exc}")
        sys.exit(1)

    out_path = results_dir / "ciciot_events.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2)
        print(f"[CICIoT] Saved {len(events)} events to {out_path}")
    except Exception as exc:
        print(f"[CICIoT] Failed to save JSON: {exc}")
        sys.exit(1)

    if events:
        print("\n[CICIoT] Sample event (index 0):")
        for k, v in events[0].items():
            print(f"  {k}: {v}")
