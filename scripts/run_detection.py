"""Run the full pipeline: embed -> index -> detect -> evaluate -> report.

Usage:
    python scripts/run_detection.py
"""

from loganomaly.config import settings
from loganomaly.pipeline import AnomalyPipeline, build_store


def main() -> None:
    pipeline = AnomalyPipeline(build_store())

    n = pipeline.ingest()
    print(f"Embedded and indexed {n} log records into collection '{settings.collection}'.")

    frame = pipeline.detect()
    # k defaults to the number of injected anomalies -> realistic review budget.
    k = int(frame["is_anomaly"].sum()) or 50
    benchmark = pipeline.evaluate(frame, k=k)
    pipeline.write_reports(frame, benchmark)

    print(f"\nBenchmark (precision@{k} = review budget of {k} alerts):\n")
    print(benchmark.to_markdown(index=False))
    print(f"\nReports written to {settings.results_dir}/")


if __name__ == "__main__":
    main()
