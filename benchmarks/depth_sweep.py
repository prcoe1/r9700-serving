#!/usr/bin/env python3
"""Depth sweep sized to the live profile: rungs are powers of two up to the
largest depth that fits under VLLM_MAX_MODEL_LEN (see profile_config), so a
131072 profile sweeps [0..125952] instead of the old hardcoded 0-200K ladder
whose top rung exceeded the window.

Usage:
    python3 benchmarks/depth_sweep.py [model] [--pp N] [--tg N] [--runs N]
        [--depth D ...] [--dry-run]

One llama-benchy invocation (`--pp 2048 --tg 32 --runs 2 --no-cache` by
default, thinking off); prints an md summary and appends a record to
observability/depth_history.jsonl. Requires a reachable vLLM server
(default http://localhost:8180, override with VLLM_BASE_URL).
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
    ap.add_argument("--pp", type=int, default=2048)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--depth", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = pc.load_profile()
    model = a.model or pc.served_name(cfg)
    max_len = pc.max_model_len(cfg)
    if a.depth is not None:
        depths = a.depth
    else:
        if max_len is None:
            raise SystemExit("VLLM_MAX_MODEL_LEN unset and no --depth given")
        depths = pc.depth_ladder(max_len, pp=a.pp, tg=a.tg)
    tokenizer = (cfg.get("VLLM_TOKENIZER")
                 or ("Qwen/Qwen3-8B" if "qwen" in model.lower() else "gpt2"))
    print(f"depth sweep: model={model} maxlen={max_len} depths={depths} "
          f"pp={a.pp} tg={a.tg} runs={a.runs}", flush=True)

    cmd = sc.benchy_cmd(BASE, model, tokenizer, a.pp, a.tg, depths,
                        runs=a.runs)
    rc, out, elapsed = sc.run(cmd, dry_run=a.dry_run)
    if a.dry_run:
        return
    record = {"ts": time.time(), "elapsed": elapsed, "model": model,
              "returncode": rc, "depths": depths, "raw": out}
    try:
        result = sc.parse_result(out)
        record.update({k: v for k, v in result.items() if k != "benchmarks"})
        sc.print_depth_table(result)
    except Exception as e:
        print(f"[result parse failed: {e}]", flush=True)
    sc.append_history("depth", record)
    if rc != 0:
        raise SystemExit(f"llama-benchy exited {rc}")


if __name__ == "__main__":
    main()
