"""

Metrics (paper §3.2):
    Hit@3, MRR, MAP@50, NDCG@50

Stages reported (matching the paper's Table 2 row order):
    vector, vector_reranked, bm25, bm25_reranked, hyde, hyde_reranked,
    rrf_k60, final_k60, rrf_k0, final_k0

Usage:
    python compute_metrics.py --input bioasq_1M_corpus_eng_hybrid_detailed.json \\
                              --output bioasq_metrics.json
    python compute_metrics.py --input combined_detailed.json --by_source
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

STAGES = [
    ("vector",            "vector_top50"),
    ("vector_reranked",   "vector_reranked_top50"),
    ("bm25",              "bm25_top50"),
    ("bm25_reranked",     "bm25_reranked_top50"),
    ("hyde",              "hyde_top50"),
    ("hyde_reranked",     "hyde_reranked_top50"),
    ("rrf_k60",           "rrf_k60_top50"),
    ("final_k60",         "final_k60_top50"),
    ("rrf_k0",            "rrf_k0_top50"),
    ("final_k0",          "final_k0_top50"),
]


def doc_ids(ranking):
    return [str(item["doc"]) for item in ranking]


def hit_at_3(ranking, gold):
    return int(any(d in gold for d in doc_ids(ranking)[:3]))


def reciprocal_rank(ranking, gold):
    for i, d in enumerate(doc_ids(ranking), start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranking, gold, k=50):
    docs = doc_ids(ranking)[:k]
    if not gold:
        return 0.0
    hits = 0
    precisions = []
    for i, d in enumerate(docs, start=1):
        if d in gold:
            hits += 1
            precisions.append(hits / i)
    R = min(len(gold), k)
    return sum(precisions) / R if R else 0.0


def ndcg_at_k(ranking, gold, k=50):
    docs = doc_ids(ranking)[:k]
    dcg = sum((1.0) / math.log2(i + 1) for i, d in enumerate(docs, start=1) if d in gold)
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(results):
    """Return {stage: {metric: float}} averaged over results."""
    out = {name: {"hit@3": 0.0, "mrr": 0.0, "map@50": 0.0, "ndcg@50": 0.0}
           for name, _ in STAGES}
    n = len(results)
    if n == 0:
        return out
    for r in results:
        gold = set(str(g) for g in r["gold_docs"])
        for name, key in STAGES:
            ranking = r.get(key) or []
            out[name]["hit@3"]   += hit_at_3(ranking, gold)
            out[name]["mrr"]     += reciprocal_rank(ranking, gold)
            out[name]["map@50"]  += average_precision_at_k(ranking, gold, 50)
            out[name]["ndcg@50"] += ndcg_at_k(ranking, gold, 50)
    for name in out:
        for m in out[name]:
            out[name][m] = round(out[name][m] / n, 4)
    return out


def print_table(label, metrics):
    print(f"\n=== {label} (n={metrics['_n']}) ===")
    print(f"{'Stage':<22}{'Hit@3':>8}{'MRR':>8}{'MAP@50':>9}{'NDCG@50':>10}")
    for name, _ in STAGES:
        m = metrics[name]
        print(f"{name:<22}{m['hit@3']:>8.3f}{m['mrr']:>8.3f}{m['map@50']:>9.3f}{m['ndcg@50']:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="*_detailed.json from evaluate.py")
    ap.add_argument("--output", default=None, help="JSON output path")
    ap.add_argument("--by_source", action="store_true", help="also report per-dataset (pubmedqa/bioasq)")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    results = data["results"]

    overall = aggregate(results)
    overall["_n"] = len(results)
    print_table("OVERALL", overall)

    report = {"overall": overall}

    if args.by_source:
        by_src = defaultdict(list)
        for r in results:
            by_src[r.get("source", "unknown")].append(r)
        for src, items in by_src.items():
            m = aggregate(items)
            m["_n"] = len(items)
            print_table(src, m)
            report[src] = m

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
