"""Cross-layer and temporal routing predictability from moe-trace JSONL files.

Track-1's analyze_locality.py measured temporal locality (reuse, LRU curves)
on Qwen3-30B-A3B. This analyzer answers the YALIS-probe question instead:
can the expert set be predicted BEFORE the selecting hidden state exists?
That is the only property that makes async expert prefetch worth building.

Predictors scored (recall = fraction of the actual expert set named ahead of
time, over-fetch = predicted-set size / actual-set size):

  temporal-1      E(L, t-1)                       -> E(L, t)
  cross-layer     E(L-1, t)                       -> E(L, t)
  fate-union      E(L-1, t) ∪ E(L, t-1)           -> E(L, t)   (the FATE fork's
                  exact heuristic; on Qwen3 traces it must reproduce the
                  fork's measured ~54% real-text hit rate, which validates
                  this metric against an independent implementation)
  fate-union-2    E(L-1, t) ∪ E(L, t-1) ∪ E(L, t-2) -> E(L, t)
  oracle-lastk    E(L, t-k..t-1) union, k=2,4,8   -> E(L, t)   (upper bound of
                  any id-history scheme)

Inputs are JSONL files with one record per (layer, token column):
    {"pos": int, "layer": int, "experts": [int, ...]}
Model-agnostic; runs on Qwen3 (48L) and gpt-oss (24L) traces alike.
Pooled metrics are request-weighted across layers and files. Nothing is
extrapolated beyond the recorded traces.

Progress banners print per trace file. Ctrl+C writes partial results to the
output JSON with status="interrupted" and exits with code 130.

Usage:
    .venv/Scripts/python.exe experiments/gpt_oss/analyze_predictability.py \
        --traces results/gpt-oss/traces/trace-*.jsonl --out results/gpt-oss/predictability.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "moe_trace"))
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from analyze_locality import load_trace  # noqa: E402
from common import ts  # noqa: E402


def sets_by_layer_pos(per_layer) -> dict[int, dict[int, frozenset]]:
    return {layer: {pos: frozenset(experts) for pos, experts in records}
            for layer, records in per_layer.items()}


def score_trace(per_layer) -> dict:
    """Request-weighted recall and over-fetch per predictor for one trace."""
    m = sets_by_layer_pos(per_layer)
    layers = sorted(m)
    # accumulators: predictor -> [named_correct, actual_total, predicted_total]
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for layer in layers:
        by_pos = m[layer]
        prev_layer = m.get(layer - 1)
        for pos, actual in by_pos.items():
            if not actual:
                continue

            def record(name: str, predicted: frozenset | None) -> None:
                if predicted is None:
                    return
                a = acc[name]
                a[0] += len(actual & predicted)
                a[1] += len(actual)
                a[2] += len(predicted)

            t1 = by_pos.get(pos - 1)
            t2 = by_pos.get(pos - 2)
            record("temporal-1", t1)

            if prev_layer is not None:
                cross = prev_layer.get(pos)
                record("cross-layer", cross)
                if cross is not None or t1 is not None:
                    record("fate-union", frozenset((cross or frozenset()) | (t1 or frozenset())))
                if cross is not None or t1 is not None or t2 is not None:
                    record("fate-union-2",
                           frozenset((cross or frozenset()) | (t1 or frozenset()) | (t2 or frozenset())))

            for k in (2, 4, 8):
                recent = [by_pos.get(pos - d) for d in range(1, k + 1)]
                recent = [r for r in recent if r is not None]
                if recent:
                    record(f"oracle-last{k}", frozenset().union(*recent))

    return acc


def summarize(acc_all: dict[str, list[int]]) -> dict:
    out = {}
    for name, (correct, actual_total, predicted_total) in sorted(acc_all.items()):
        out[name] = {
            "recall": correct / actual_total if actual_total else None,
            "over_fetch": predicted_total / actual_total if actual_total else None,
            "actual_requests": actual_total,
        }
    return out


def write_payload(out: Path, paths: list[str], per_file: dict,
                  acc_all: dict[str, list[int]], status: str) -> dict:
    payload = {
        "status": status,
        "traces": [Path(p).name for p in paths],
        "predictors": {
            "temporal-1": "E(L,t-1) -> E(L,t)",
            "cross-layer": "E(L-1,t) -> E(L,t)",
            "fate-union": "E(L-1,t) | E(L,t-1) -> E(L,t) (FATE fork heuristic)",
            "fate-union-2": "E(L-1,t) | E(L,t-1) | E(L,t-2) -> E(L,t)",
            "oracle-lastk": "union of last k same-layer sets -> E(L,t), k=2,4,8",
        },
        "per_file": per_file,
        "pooled": summarize(acc_all),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True,
                        help="trace JSONL files or glob patterns")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths: list[str] = []
    for item in args.traces:
        paths.extend(sorted(glob.glob(item)))
    if not paths:
        raise SystemExit(f"no trace files matched: {args.traces}")

    print(f"[{ts()}] predictability analysis: {len(paths)} trace file(s), "
          f"5 predictor families", flush=True)
    print(f"[{ts()}] output -> {args.out} (Ctrl+C writes partial results)",
          flush=True)

    acc_all: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_file = {}
    processed: list[str] = []
    try:
        for i, path in enumerate(paths, 1):
            stem = Path(path).stem
            print(f"[{ts()}] [file {i}/{len(paths)}] {stem}: loading and "
                  f"scoring", flush=True)
            per_layer = load_trace(Path(path))
            acc = score_trace(per_layer)
            for name, vals in acc.items():
                for j in range(3):
                    acc_all[name][j] += vals[j]
            tokens = sum(len(v) for v in per_layer.values()) // max(len(per_layer), 1)
            per_file[stem] = {"layers": len(per_layer), "tokens_per_layer": tokens}
            processed.append(path)
            union = acc.get("fate-union", [0, 0, 0])
            recall = union[0] / union[1] if union[1] else None
            line = (f"{len(per_layer)} layers, {tokens} tokens/layer, "
                    f"fate-union recall {recall:.3f}" if recall is not None else
                    f"{len(per_layer)} layers, {tokens} tokens/layer")
            print(f"[{ts()}] [file {i}/{len(paths)}] {stem}: {line}",
                  flush=True)
    except KeyboardInterrupt:
        write_payload(args.out, processed, per_file, acc_all,
                      status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(processed)}/"
              f"{len(paths)} files) -> {args.out}", flush=True)
        sys.exit(130)

    payload = write_payload(args.out, paths, per_file, acc_all,
                            status="complete")

    print("\npooled predictors (request-weighted):", flush=True)
    print(f"  {'predictor':<14} {'recall':>7} {'over-fetch':>10}", flush=True)
    for name, s in payload["pooled"].items():
        print(f"  {name:<14} {s['recall']:7.3f} {s['over_fetch']:10.2f}",
              flush=True)
    print(f"\n[{ts()}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
