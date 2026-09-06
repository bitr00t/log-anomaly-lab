# Log Anomaly Lab

**Embedding-based anomaly detection on log streams — with a vector database
as part of the model and a labelled benchmark instead of eyeballing.**

Logs are text, and text embeddings turn "this line looks weird" into
geometry: healthy microservice traffic is structurally repetitive and forms
dense regions in embedding space, while stack traces, injection probes and
rogue services land in sparse regions. This project embeds log lines,
indexes them in Qdrant, runs three classic unsupervised detectors over the
vectors, and — because the synthetic dataset is labelled — measures each
detector with proper metrics (ROC-AUC, PR-AUC, precision@k, per-type
recall).

## What this project demonstrates

- **Classic ML on modern representations** — Isolation Forest and Local
  Outlier Factor (scikit-learn) operating on sentence-transformer
  embeddings instead of hand-crafted features.
- **The vector DB as a detector, not just storage** — the third detector
  (`knn_qdrant`) computes its anomaly score directly from Qdrant k-NN
  searches: mean cosine distance to the 10 nearest neighbours.
- **Honest evaluation under class imbalance** — PR-AUC and precision@k as
  headline metrics, plus per-anomaly-type recall that exposes detector
  blind spots (a detector can look great overall and still miss every
  SQL-injection probe).
- **Triage, not just alerts** — DBSCAN clusters the top alerts into
  incident groups and every alert ships with its nearest neighbours for
  context; that is what makes a detector *usable* on-call.
- **Fully local & reproducible** — no API keys anywhere; seeded data
  generator; unit tests that run without torch or Qdrant.

## Architecture

```
 scripts/generate_logs.py
        │  seeded synthetic logs (~3% labelled anomalies:
        │  stack_trace | oom | sqli_probe | auth_bruteforce | rogue_service)
        ▼
 data/logs.jsonl ──► Embedder (sentence-transformers, CPU)
                            │  L2-normalised vectors
                            ▼
                     Qdrant (cosine)
                      │            │
        ┌─────────────┘            └──────────────┐
        ▼                                         ▼
 sklearn detectors                        knn_qdrant detector
 (Isolation Forest, LOF)                  (mean distance to 10-NN,
  on the embedding matrix                  computed via Qdrant search)
        │                                         │
        └──────────────────┬──────────────────────┘
                           ▼
                 Evaluation vs. ground truth
                 ROC-AUC · PR-AUC · precision@k · recall per anomaly type
                           │
                           ▼
        results/detector_benchmark.md   (metric table)
        results/scores.csv              (per-record scores)
        results/triage_report.md        (top alerts + neighbours
                                         + DBSCAN alert clusters)
```

## Quickstart

Requirements: Python 3.10+ and Docker. **No API keys** — everything runs
locally on CPU.

```bash
# 1. Install
pip install -r requirements.txt && pip install -e .

# 2. Unit tests (run without Qdrant or torch)
pytest -v

# 3. Start Qdrant
docker compose up -d          # dashboard: http://localhost:6333/dashboard

# 4. Generate the labelled dataset (2000 lines, ~3% anomalies, seed 42)
python scripts/generate_logs.py 2000 0.03 42

# 5. Run the full pipeline: embed -> index -> detect -> evaluate -> report
python scripts/run_detection.py

# 6. Ad-hoc triage: inspect any record and its nearest neighbours
python scripts/inspect_record.py log-000123
```

`run_detection.py` prints the benchmark table and writes three artifacts:

| File | Content |
|---|---|
| `results/detector_benchmark.md` | one row per detector: ROC-AUC, PR-AUC, precision@k, recall@k per anomaly type |
| `results/scores.csv` | every record with the normalised score from every detector — for error analysis |
| `results/triage_report.md` | top alerts with nearest-neighbour context + DBSCAN clusters over the alerts |

*(Run the pipeline to produce real numbers for your machine — comparing the
three detectors, and finding which anomaly type each one misses, is the
point of the project.)*

## The three detectors

| Detector | Idea | Catches | Cost |
|---|---|---|---|
| `isolation_forest` | anomalies are isolated by fewer random splits | global outliers | ~linear, very fast |
| `lof` | compares local density with neighbours' density | local outliers inside mixed-density data | O(n·k) neighbour queries (sklearn, in-memory) |
| `knn_qdrant` | mean cosine distance to the 10 nearest neighbours, **computed by the vector DB** | sparse-region points; trivially scales to streaming (query per new log line) | one ANN search per record |

All three are unsupervised — labels are used **only** for evaluation.

## Why precision@k and per-type recall?

With ~3% anomalies, accuracy is meaningless and ROC-AUC is optimistic.
The operational question is: *"if an engineer reviews the top k alerts,
how many are real — and which incident types never make it into the top
k?"* That is exactly `precision@k` and `recall@k per anomaly type`. The
benchmark sets k to the number of injected anomalies, i.e. a realistic
review budget.

## Using your own logs

1. Convert your logs to the JSONL schema of `data/logs.jsonl`
   (`record_id`, `timestamp`, `level`, `service`, `message`; set
   `is_anomaly=false` everywhere if you have no labels).
2. Run `python scripts/run_detection.py`. Without labels the evaluation
   table is not meaningful, but `scores.csv` and `triage_report.md` are —
   they are the actual product of the pipeline.

## Project structure

```
log-anomaly-lab/
├── src/loganomaly/
│   ├── config.py                 # env-based settings
│   ├── models.py                 # LogRecord + JSONL I/O
│   ├── pipeline.py               # embed -> index -> detect -> evaluate -> report
│   ├── simulation/generator.py   # seeded synthetic logs + 5 anomaly injectors
│   ├── embedding/embedder.py     # sentence-transformers wrapper
│   ├── storage/vector_store.py   # Qdrant: index, kNN scores, neighbour lookup
│   ├── detection/detectors.py    # IsolationForest, LOF, DBSCAN triage clustering
│   └── evaluation/metrics.py     # ROC-AUC, PR-AUC, precision@k, per-type recall
├── scripts/                      # generate_logs / run_detection / inspect_record
├── tests/                        # unit tests (no torch, no Qdrant)
├── docker-compose.yml            # local Qdrant
└── docs/DESIGN_DECISIONS.md      # rationale for the non-obvious choices
```

## Design decisions

See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) — including why
the label never touches the detectors, why scores are min-max normalised,
and why DBSCAN is used for triage rather than detection.

## Limitations & possible extensions

- **Synthetic data** — deliberately simple and labelled; swapping in a
  public dataset (e.g. parsed HDFS/BGL logs) is the natural next step and
  only touches the loader.
- **Streaming mode** — `knn_qdrant` already works per-record; wrapping it
  in a small consumer (score each incoming line against the existing
  index) turns the batch pipeline into a near-real-time detector.
- **Drift handling** — deployments change what "normal" looks like;
  re-indexing windows or time-decayed collections would address this.
- **Ensembling** — rank-averaging the three detectors is a ~10-line
  experiment with the existing score frame.

## License

MIT
