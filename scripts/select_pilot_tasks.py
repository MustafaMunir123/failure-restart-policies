#!/usr/bin/env python3
"""Phase 4 preview — deterministic 50-task pilot selection from the canonical test list.

Run on the host after bootstrap (needs the pinned repo):
    python scripts/select_pilot_tasks.py \
        --testlist ../thinkingbox-data/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml \
        --seed 20260823 --out configs/pilot_tasklist.yaml

Deterministic given --seed; rerunning must reproduce the identical list.
Replacement rule: if a drawn task fails smoke-level checks, take the NEXT seeded draw
and log the replacement in artifacts/task_replacements.md. Never hand-pick.
"""
import argparse
import hashlib
import random
from collections import defaultdict

DOMAINS = [
    "retail", "booking", "insurance", "neobank", "consulting",
]
PER_DOMAIN = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testlist", required=True)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # TODO(phase-4): parse the real YAML schema (verify keys on pinned commit).
    import yaml  # noqa: PLC0415 — host-side dependency
    with open(args.testlist) as f:
        entries = yaml.safe_load(f)

    by_domain: dict[str, list] = defaultdict(list)
    for e in entries:  # expects fields: name, domain — adjust to actual schema
        by_domain[str(e["domain"]).lower()].append(e["name"])

    rng = random.Random(hashlib.sha256(str(args.seed).encode()).hexdigest())
    selection = {}
    for domain in DOMAINS:
        pool = sorted(by_domain.get(domain, []))
        if len(pool) < PER_DOMAIN:
            raise SystemExit(f"domain {domain}: only {len(pool)} tasks available")
        selection[domain] = rng.sample(pool, PER_DOMAIN)

    lines = [f"# Pilot tasklist — seed={args.seed} generated deterministically."]
    for domain, names in selection.items():
        lines.append(f"{domain}:")
        lines += [f"  - {n}" for n in names]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({sum(len(v) for v in selection.values())} tasks)")


if __name__ == "__main__":
    main()
