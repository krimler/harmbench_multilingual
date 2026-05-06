"""Stratified sample of HarmBench behaviors by FunctionalCategory."""
import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "HarmBench" / "data" / "behavior_datasets" / "harmbench_behaviors_text_all.csv"
OUT = ROOT / "data" / "sampled_behaviors.csv"

SAMPLE_N = 100
SEED = 42


def stratified_sample(rows, n, key, seed):
    rng = random.Random(seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r[key], []).append(r)
    total = len(rows)
    target = {c: n * len(v) / total for c, v in by_cat.items()}
    initial = {c: int(math.floor(t)) for c, t in target.items()}
    deficit = n - sum(initial.values())
    fracs = sorted(((target[c] - initial[c], c) for c in by_cat), reverse=True)
    for i in range(deficit):
        initial[fracs[i % len(fracs)][1]] += 1
    overflow = 0
    for c, k in list(initial.items()):
        if k > len(by_cat[c]):
            overflow += k - len(by_cat[c])
            initial[c] = len(by_cat[c])
    if overflow:
        candidates = sorted(
            (c for c in by_cat if initial[c] < len(by_cat[c])),
            key=lambda c: -(target[c] - initial[c]),
        )
        i = 0
        while overflow and candidates:
            c = candidates[i % len(candidates)]
            if initial[c] < len(by_cat[c]):
                initial[c] += 1
                overflow -= 1
            i += 1
            if all(initial[c] == len(by_cat[c]) for c in candidates):
                break
    out = []
    for c, pool in by_cat.items():
        rng_local = random.Random(seed + hash(c) % 10_000)
        rng_local.shuffle(pool)
        out.extend(pool[: initial[c]])
    rng.shuffle(out)
    return out


def main():
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))
    sampled = stratified_sample(rows, SAMPLE_N, "FunctionalCategory", SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sampled)
    from collections import Counter

    print(f"wrote {OUT} (n={len(sampled)})")
    print("FunctionalCategory:", Counter(r["FunctionalCategory"] for r in sampled))
    print("SemanticCategory:", Counter(r["SemanticCategory"] for r in sampled))


if __name__ == "__main__":
    main()
