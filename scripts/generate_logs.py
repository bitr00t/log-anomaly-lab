"""Generate a labelled synthetic log dataset.

Usage:
    python scripts/generate_logs.py [n] [anomaly_rate] [seed]
    python scripts/generate_logs.py 2000 0.03 42
"""

import sys

from loganomaly.config import settings
from loganomaly.models import save_jsonl
from loganomaly.simulation.generator import generate_logs


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    records = generate_logs(n=n, anomaly_rate=rate, seed=seed)
    save_jsonl(records, settings.logs_file)
    n_anomalies = sum(r.is_anomaly for r in records)
    print(f"Wrote {len(records)} records ({n_anomalies} anomalies) to {settings.logs_file}")


if __name__ == "__main__":
    main()
