"""Next-experiments #4 quality gate: perplexity of mixed-precision variants.

Three-way same-corpus comparison on the perplexity half of the Track-3 corpus
(results/track3-low-bit/corpus-ppl-half.txt) — the imatrix used for the
IQ2_XXS layers was built from the OTHER half (corpus-imatrix-half.txt), so
the gate never measures text the quantizer tuned for. All models run at the
Track-1 measured optimum (-ngl 48 --n-cpu-moe 33).

Models: original Q4_K_M anchor, the plain requantize control (isolates
requantization loss), and the mixed-precision variant(s). Missing files are
recorded as skipped rows, never fabricated.

Child stdout/stderr stream live by default (--quiet suppresses). Ctrl+C
terminates the running child, writes partial results to the output JSON with
status="interrupted", and exits with code 130.

Output: results/moe-locality/mixed-precision-ppl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "system"))
from common import run_live, ts  # noqa: E402

PERP = ROOT / "tools/llama.cpp-b10355/llama-perplexity.exe"
OUT = ROOT / "results/moe-locality/mixed-precision-ppl.json"
CORPUS = ROOT / "results/track3-low-bit/corpus-ppl-half.txt"

MODELS = [
    ("original-q4km", "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"),
    ("control-requant-q4km", "models/Qwen3-30B-A3B-GGUF/qwen3-30b-a3b-q4km-requant-control.gguf"),
    ("variant-mid24-iq2xxs", "models/Qwen3-30B-A3B-GGUF/qwen3-30b-a3b-mid24-iq2xxs.gguf"),
    ("variant-mid24-q3k", "models/Qwen3-30B-A3B-GGUF/qwen3-30b-a3b-mid24-q3k.gguf"),
]

PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
PLACEMENT = ["-ngl", "48", "--n-cpu-moe", "33"]


def write_payload(records: list[dict], status: str) -> None:
    payload = {
        "status": status,
        "tool": "llama-perplexity b10355 (unmodified)",
        "corpus": "results/track3-low-bit/corpus-ppl-half.txt (imatrix-disjoint half; "
                  "relative comparison only)",
        "placement": " ".join(PLACEMENT),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live child output (banners still print)")
    args = ap.parse_args()

    records: list[dict] = []
    print(f"[{ts()}] quality gate: {len(MODELS)} models, corpus {CORPUS.name}, "
          f"placement {' '.join(PLACEMENT)}", flush=True)
    print(f"[{ts()}] output -> {OUT} (Ctrl+C writes partial results)",
          flush=True)
    try:
        for i, (model_id, rel_path) in enumerate(MODELS, 1):
            model = ROOT / rel_path
            label = f"[{i}/{len(MODELS)}] {model_id}"
            if not model.exists():
                records.append({"model": model_id, "status": "skipped",
                                "reason": "file not present"})
                print(f"[{ts()}] {label}: SKIPPED (file not present)",
                      flush=True)
                continue
            cmd = [str(PERP), "-m", str(model), "-f", str(CORPUS), *PLACEMENT]
            print(f"[{ts()}] {label}: measuring", flush=True)
            res = run_live(cmd, label, quiet=args.quiet)
            record = {"model": model_id, "rc": res["rc"],
                      "wall_s": res["wall_s"]}
            match = PPL_RE.search(res["stderr"] + res["stdout"])
            if match:
                record["ppl"] = float(match.group(1))
                record["ppl_stderr"] = float(match.group(2))
                print(f"[{ts()}] {label}: PPL = {match.group(1)} "
                      f"+/- {match.group(2)}", flush=True)
            else:
                record["status"] = "error"
                record["stderr_tail"] = res["stderr"][-400:]
                print(f"[{ts()}] {label}: FAILED rc={res['rc']}: "
                      f"{res['stderr'][-200:]!r}", flush=True)
            records.append(record)
    except KeyboardInterrupt:
        write_payload(records, status="interrupted")
        print(f"\n[{ts()}] interrupted — partial results ({len(records)} "
              f"rows) -> {OUT}", flush=True)
        sys.exit(130)

    write_payload(records, status="complete")
    print(f"\n[{ts()}] wrote {OUT}", flush=True)
    for r in records:
        if "ppl" in r:
            print(f"  {r['model']}: PPL {r['ppl']} +/- {r['ppl_stderr']}",
                  flush=True)


if __name__ == "__main__":
    main()
