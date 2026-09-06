"""Show one record and its nearest neighbours from Qdrant (ad-hoc triage).

Usage:
    python scripts/inspect_record.py log-000123
"""

import sys

from loganomaly.config import settings
from loganomaly.models import load_jsonl
from loganomaly.pipeline import build_store


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/inspect_record.py <record_id>")
    record_id = sys.argv[1]

    records = load_jsonl(settings.logs_file)
    match = next((r for r in records if r.record_id == record_id), None)
    if match is None:
        sys.exit(f"record_id {record_id} not found in {settings.logs_file}")

    # Embed just this one record and query the existing collection.
    from loganomaly.embedding.embedder import Embedder

    vector = Embedder(settings.embedding_model).embed([match.text])[0]
    print(f"{match.record_id} | {match.text}\n")
    print("Nearest neighbours:")
    for nb in build_store().neighbours(settings.collection, vector, limit=6)[1:]:
        print(f"  ({nb['score']:.3f}) {nb['record_id']} | {nb['level']} {nb['service']}: {nb['message'][:100]}")


if __name__ == "__main__":
    main()
