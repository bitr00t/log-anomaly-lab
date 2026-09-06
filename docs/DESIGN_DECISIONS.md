# Design Decisions

The "why" behind the non-obvious choices — the questions this project
should be able to answer in an interview.

## Labels exist, but detectors never see them

The generator produces ground-truth labels, which turns anomaly detection
from a demo into a measurable benchmark. The labels flow *around* the
detectors, never *into* them: `Detector.fit_score(X)` receives only the
embedding matrix. This mirrors reality (production logs are unlabelled)
while still allowing quantitative evaluation on the synthetic set.

## The vector DB is part of the model, not just storage

`knn_qdrant` computes its anomaly score from Qdrant search results (mean
cosine distance to the 10 nearest neighbours). Besides demonstrating a
second role for the vector database, this detector has a property the
sklearn ones lack: it scores a *single new record* in one ANN query
against the existing index, which makes the batch pipeline trivially
extendable to streaming.

## Embeddings are L2-normalised

One normalisation, two payoffs: cosine similarity in Qdrant becomes a pure
dot product (correct and fast), and Euclidean distances used implicitly by
Isolation Forest / LOF are bounded and comparable across records.

## `level` and `service` are prepended to the embedded text

`f"{level} {service}: {message}"` — both fields carry real signal. A
never-seen service name (the `rogue_service` anomaly) is suspicious even
when its message text looks harmless; embedding it makes that signal
available to every detector without feature engineering.

## Min-max score normalisation, and why it is safe

Raw scores from Isolation Forest (path lengths), LOF (outlier factors) and
k-NN (distances) live on incomparable scales. Min-max scaling to [0, 1] is
a strictly monotone transformation, so ranking metrics (ROC-AUC, PR-AUC,
precision@k) are provably unchanged — normalisation exists purely for
readable reports and cross-detector score tables.

## PR-AUC and precision@k as headline metrics

At ~3% positives, ROC-AUC can look excellent while the top of the alert
list is full of false positives, because ROC-AUC rewards ranking among the
overwhelming majority of easy negatives. PR-AUC and precision@k evaluate
only what matters operationally: the top of the list. Per-type recall@k
additionally exposes blind spots — a detector can post a high PR-AUC while
consistently missing one anomaly category.

## DBSCAN clusters alerts, it does not detect

Clustering could be misused as a detector ("noise = anomaly"), but its
noise label conflates rarity with anomalousness and is very sensitive to
`eps`. Here DBSCAN serves triage instead: the top alerts are clustered so
an engineer reviews *incident groups* ("one OOM loop, one brute-force
wave") rather than hundreds of individual lines. DBSCAN fits triage well
because the number of incident types is unknown a priori and an explicit
noise bucket is a feature, not a bug.

## Deterministic everything

Seeded data generator, fixed `random_state` in Isolation Forest, ordered
records. Two runs on the same machine produce identical benchmark tables —
a precondition for comparing any future change (new embedder, new
detector, different k) against a stable baseline.

## Tests run without torch or Qdrant

The pure logic (generator invariants, detector ranking on planted numeric
outliers, metric math, cluster summaries) is tested on synthetic numpy
data. The embedding model and the vector store are integration concerns,
kept behind lazy imports — the test suite stays fast and CI stays free.
