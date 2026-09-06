"""Synthetic log stream generator.

Real production logs are rarely shareable, so the project ships a seeded
generator that produces a realistic mix of normal traffic plus labelled
anomalies. Because every record carries its ground-truth label, detectors
can be evaluated quantitatively (ROC-AUC, precision@k) instead of eyeballed.

Normal traffic: templated INFO/DEBUG/WARN lines with randomised fields
(latencies, user ids, endpoints) — high volume, structurally repetitive,
exactly like healthy microservice logs.

Anomalies (each a distinct *type*, so per-type recall can be analysed):
  * stack_trace    – unhandled exception with a Java-ish traceback
  * oom            – out-of-memory kills and heap exhaustion
  * sqli_probe     – SQL-injection-looking request payloads
  * auth_bruteforce– bursts of failed logins from one IP
  * rogue_service  – log lines from a service name that never appears
                     in normal traffic (e.g. crypto-miner behaviour)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from loganomaly.models import LogRecord

# --------------------------------------------------------------------------- #
# Normal traffic templates
# --------------------------------------------------------------------------- #
_SERVICES = ["api-gateway", "user-service", "order-service", "payment-service", "cache"]
_ENDPOINTS = ["/api/v1/users", "/api/v1/orders", "/api/v1/products", "/health", "/api/v1/cart"]

_NORMAL_TEMPLATES = [
    ("INFO", "{svc}", "GET {endpoint} completed in {ms}ms status=200"),
    ("INFO", "{svc}", "POST {endpoint} completed in {ms}ms status=201"),
    ("INFO", "user-service", "User {uid} logged in successfully from session {sid}"),
    ("INFO", "order-service", "Order {oid} created for user {uid}, total={amount} EUR"),
    ("INFO", "payment-service", "Payment {pid} authorized for order {oid}"),
    ("DEBUG", "cache", "Cache hit for key user:{uid}:profile ttl={ttl}s"),
    ("DEBUG", "cache", "Cache miss for key order:{oid}, fetching from database"),
    ("INFO", "api-gateway", "Health check passed, upstream latency {ms}ms"),
    ("WARN", "{svc}", "Slow query detected: {ms}ms for {endpoint}, above 500ms threshold"),
    ("INFO", "user-service", "Password changed for user {uid}"),
]

# --------------------------------------------------------------------------- #
# Anomaly injectors — each returns (level, service, message)
# --------------------------------------------------------------------------- #

def _stack_trace(rng: random.Random) -> tuple[str, str, str]:
    svc = rng.choice(_SERVICES)
    return (
        "ERROR",
        svc,
        f"Unhandled exception in request handler: NullPointerException at "
        f"com.shop.{svc.replace('-', '.')}.Handler.process(Handler.java:{rng.randint(40, 400)}) "
        f"caused by: java.lang.NullPointerException: order was null "
        f"at OrderMapper.toDto(OrderMapper.java:{rng.randint(10, 99)})",
    )


def _oom(rng: random.Random) -> tuple[str, str, str]:
    return (
        "ERROR",
        rng.choice(_SERVICES),
        f"java.lang.OutOfMemoryError: Java heap space, killed container after "
        f"exceeding memory limit {rng.choice([512, 1024, 2048])}Mi, restart count={rng.randint(2, 9)}",
    )


def _sqli_probe(rng: random.Random) -> tuple[str, str, str]:
    payload = rng.choice(
        ["' OR '1'='1' --", "'; DROP TABLE users; --", "1 UNION SELECT username, password FROM users"]
    )
    return (
        "WARN",
        "api-gateway",
        f"Suspicious request blocked: GET /api/v1/users?id={payload} from ip=185.220.{rng.randint(0,255)}.{rng.randint(1,254)}",
    )


def _auth_bruteforce(rng: random.Random) -> tuple[str, str, str]:
    ip = f"91.240.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return (
        "WARN",
        "user-service",
        f"Failed login attempt {rng.randint(10, 60)} for account admin from ip={ip} within 60s window",
    )


def _rogue_service(rng: random.Random) -> tuple[str, str, str]:
    return (
        "INFO",
        "xmr-worker",  # a service name that never occurs in normal traffic
        f"Connected to pool stratum+tcp://pool.example:{rng.randint(3000, 9000)}, "
        f"hashrate {rng.randint(100, 900)} H/s, shares accepted {rng.randint(1, 50)}",
    )


ANOMALY_INJECTORS = {
    "stack_trace": _stack_trace,
    "oom": _oom,
    "sqli_probe": _sqli_probe,
    "auth_bruteforce": _auth_bruteforce,
    "rogue_service": _rogue_service,
}


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

def _render_normal(rng: random.Random) -> tuple[str, str, str]:
    level, svc, template = rng.choice(_NORMAL_TEMPLATES)
    svc = svc.format(svc=rng.choice(_SERVICES))
    message = template.format(
        svc=svc,
        endpoint=rng.choice(_ENDPOINTS),
        ms=rng.randint(3, 900),
        uid=rng.randint(1000, 9999),
        sid=rng.getrandbits(32),
        oid=rng.randint(50_000, 99_999),
        pid=rng.randint(70_000, 99_999),
        amount=round(rng.uniform(5, 500), 2),
        ttl=rng.choice([60, 300, 900]),
    )
    return level, svc, message


def generate_logs(n: int = 2000, anomaly_rate: float = 0.03, seed: int = 42) -> list[LogRecord]:
    """Generate `n` records with roughly `anomaly_rate` labelled anomalies.

    Deterministic for a given seed, so benchmark runs are reproducible and
    detector comparisons are apples-to-apples.
    """
    if not 0.0 < anomaly_rate < 0.5:
        raise ValueError("anomaly_rate must be in (0, 0.5) — anomalies are rare by definition")
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, 0, 0, 0)
    injector_names = list(ANOMALY_INJECTORS)

    records: list[LogRecord] = []
    for i in range(n):
        is_anomaly = rng.random() < anomaly_rate
        if is_anomaly:
            anomaly_type = rng.choice(injector_names)
            level, svc, message = ANOMALY_INJECTORS[anomaly_type](rng)
        else:
            anomaly_type = ""
            level, svc, message = _render_normal(rng)

        records.append(
            LogRecord(
                record_id=f"log-{i:06d}",
                timestamp=(start + timedelta(seconds=i * 3)).isoformat(),
                level=level,
                service=svc,
                message=message,
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
            )
        )
    return records
