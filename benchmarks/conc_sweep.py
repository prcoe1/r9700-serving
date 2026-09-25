#!/usr/bin/env python3
"""Concurrency sweep sized to the live server: levels cover 1..max_num_seqs
(powers of two plus the max, see profile_config), so a conc-4 profile tests
[1, 2, 4] and a conc-8 profile [1, 2, 4, 8]. Depths default to the running
server's `--max-model-len` ladder (live `GET /v1/models`, env fallback —
same sizing as depth_sweep.py).

Usage:
    python3 benchmarks/conc_sweep.py [model] [--depth D ...]
        [--levels C ...] [--pp N] [--tg N] [--runs N] [--dry-run]

One llama-benchy invocation over the depth x concurrency product
(`--pp 2048 --tg 32 --runs 1` by default, thinking off); prints an md
summary and appends a record to observability/conc_history.jsonl. Requires
a reachable vLLM server (default http://localhost:8180, override with
VLLM_BASE_URL).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_config as pc  # noqa: E402
import sweep_common as sc  # noqa: E402

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8180")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--depth", type=int, nargs="*", default=None)
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--pp", type=int, default=2048)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = pc.load_profile()
    model = a.model or pc.served_name(cfg)
    max_len, src = pc.resolve_max_len(cfg, base_url=BASE)
    max_seqs = pc.max_num_seqs(cfg)
    levels = a.levels if a.levels is not None else pc.conc_ladder(max_seqs)
    if a.depth is not None:
        depths = a.depth
    else:
        if max_len is None:
            raise SystemExit("live /v1/models unreachable and "
                             "VLLM_MAX_MODEL_LEN unset, and no --depth given")
        depths = pc.depth_ladder(max_len, pp=a.pp, tg=a.tg)
    tokenizer = (cfg.get("VLLM_TOKENIZER")
                 or ("Qwen/Qwen3-8B" if "qwen" in model.lower() else "gpt2"))
    print(f"conc sweep: model={model} maxlen={max_len} (source={src}) "
          f"maxseqs={max_seqs} levels={levels} depths={depths} "
          f"pp={a.pp} tg={a.tg} runs={a.runs}", flush=True)

    cmd = sc.benchy_cmd(BASE, model, tokenizer, a.pp, a.tg, depths,
                        levels=levels, runs=a.runs)
    rc, out, elapsed = sc.run(cmd, dry_run=a.dry_run)
    if a.dry_run:
        return
    record = {"ts": time.time(), "elapsed": elapsed, "model": model,
              "returncode": rc, "depths": depths, "concurrency": levels,
              "raw": out}
    try:
        result = sc.parse_result(out)
        record.update({k: v for k, v in result.items() if k != "benchmarks"})
        sc.print_conc_table(result)
    except Exception as e:
        print(f"[result parse failed: {e}]", flush=True)
    sc.append_history("conc", record)
    if rc != 0:
        raise SystemExit(f"llama-benchy exited {rc}")


if __name__ == "__main__":
    main()
