#!/usr/bin/env python3
"""Generate experiment plots from backend experiment plot-data endpoint.

Usage:
  python backend/plot_experiment.py \
    --experiment-id exp-20260310-120001-abc12345 \
    --base-url http://localhost:8082/api/v1 \
    --output-dir backend/experiment_plots
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import requests

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print("matplotlib import failed:", exc)
    print("Please install: pip install matplotlib")
    sys.exit(1)


def fetch_plot_data(base_url: str, experiment_id: str) -> Dict:
    url = f"{base_url.rstrip('/')}/experiments/{experiment_id}/plot-data"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"Backend returned error: {json.dumps(payload, ensure_ascii=False)}")
    return payload["data"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_latency_plot(data: Dict, output_dir: str) -> str:
    plt.figure(figsize=(10, 6))
    for series in data.get("latency_by_operation", []):
        name = series.get("name", "unknown")
        points = series.get("points", [])
        x = [p.get("round", 0) for p in points]
        y = [p.get("value", 0.0) for p in points]
        if x and y:
            plt.plot(x, y, marker="o", linewidth=2, label=name)
    plt.title("Latency by Operation (ms)")
    plt.xlabel("Round")
    plt.ylabel("Avg Duration (ms)")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(output_dir, "latency_by_operation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_output_cipher_plot(data: Dict, output_dir: str) -> str:
    points = data.get("output_cipher_by_round", [])
    x = [p.get("round", 0) for p in points]
    y = [p.get("value", 0.0) for p in points]

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.title("Output Cipher Count by Round")
    plt.xlabel("Round")
    plt.ylabel("Cipher Count")
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(output_dir, "output_cipher_by_round.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_throughput_plot(data: Dict, output_dir: str) -> str:
    points = data.get("throughput_by_round", [])
    x = [p.get("round", 0) for p in points]
    y = [p.get("value", 0.0) for p in points]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="s", linewidth=2)
    plt.title("Weight Throughput by Round")
    plt.xlabel("Round")
    plt.ylabel("Weights / sec")
    plt.grid(alpha=0.3)
    out = os.path.join(output_dir, "throughput_by_round.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_event_count_plot(data: Dict, output_dir: str) -> str:
    counts = data.get("event_count_by_operation", {})
    labels = sorted(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("Event Count by Operation")
    plt.xlabel("Operation")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(output_dir, "event_count_by_operation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment plots from MKHE backend")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    parser.add_argument("--base-url", default="http://localhost:8082/api/v1", help="Backend API base URL")
    parser.add_argument("--output-dir", default="backend/experiment_plots", help="Directory to store images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    data = fetch_plot_data(args.base_url, args.experiment_id)

    outputs: List[Tuple[str, str]] = []
    outputs.append(("latency", save_latency_plot(data, args.output_dir)))
    outputs.append(("output_cipher", save_output_cipher_plot(data, args.output_dir)))
    outputs.append(("throughput", save_throughput_plot(data, args.output_dir)))
    outputs.append(("event_count", save_event_count_plot(data, args.output_dir)))

    print("Generated plot files:")
    for name, path in outputs:
        print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
