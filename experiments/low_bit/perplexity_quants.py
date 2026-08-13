"""Track 3: perplexity quality pass over a fixed local corpus.

Quality is measured as llama-perplexity over a fixed corpus assembled from
project documentation (consistent across models; a relative comparison, not
an absolute benchmark). Runs each candidate plus the Q4_K_M dense anchor and
the MoE anchor at their best known offload configs.

Output: results/track3-low-bit/perplexity.json
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERP = ROOT / "tools/llama.cpp-b10355/llama-perplexity.exe"
OUT = ROOT / "results/track3-low-bit/perplexity.json"
CORPUS = ROOT / "results/track3-low-bit/corpus.txt"

MODELS = [
    # (id, path, extra args)
    ("qwen3-32b-q4km", "models/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf", ["-ngl", "22"]),
    ("qwen3-32b-q3km", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-Q3_K_M.gguf", ["-ngl", "22"]),
    ("qwen3-32b-iq2xxs", "models/Qwen3-32B-GGUF/Qwen_Qwen3-32B-IQ2_XXS.gguf", ["-ngl", "40"]),
    ("bitnet-2b-i2s", "models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf", ["-ngl", "99"]),
    ("qwen3-30b-a3b-q4km", "models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
     ["-ngl", "48", "--n-cpu-moe", "33"]),
]

PPL_RE = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def build_corpus() -> None:
    """Concatenate project docs into a fixed corpus (relative comparisons only)."""
    parts = []
    for name in ["README.md", "initial_prompt.md", "docs/moe-track-plan.md",
                 "docs/system-characterization.md", "docs/model-selection.md"]:
        path = ROOT / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text("\n\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_corpus()
    records = []
    for model_id, rel_path, extra in MODELS:
        model = ROOT / rel_path
        if not model.exists():
            records.append({"model": model_id, "status": "missing"})
            print(f"[skip] {model_id}: not found", flush=True)
            continue
        cmd = [str(PERP), "-m", str(model), "-f", str(CORPUS), *extra]
        print(f"[ppl] {model_id}", flush=True)
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=3600)
        record = {"model": model_id, "rc": proc.returncode,
                  "wall_s": round(time.perf_counter() - start, 1)}
        match = PPL_RE.search(proc.stderr + proc.stdout)
        if match:
            record["ppl"] = float(match.group(1))
            record["ppl_stderr"] = float(match.group(2))
            print(f"  PPL = {match.group(1)} +/- {match.group(2)}", flush=True)
        else:
            record["status"] = "error"
            record["stderr_tail"] = proc.stderr[-400:]
            print(f"  FAILED rc={proc.returncode}: {proc.stderr[-200:]!r}", flush=True)
        records.append(record)

    payload = {
        "tool": "llama-perplexity b10355 (unmodified)",
        "corpus": "results/track3-low-bit/corpus.txt (project docs; relative comparison only)",
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
