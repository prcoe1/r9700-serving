#!/usr/bin/env python3
"""Shared llama-benchy runner for the CLI sweep wrappers (depth_sweep.py,
conc_sweep.py). Runs one benchy invocation, parses the `--format json`
result (same shape the dashboard normalizes), prints a compact md summary,
and appends a record to the observability history jsonl in the established
{ts, elapsed, model, returncode, depths[, concurrency], raw} shape.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHY = ["uvx", "llama-benchy@0.4.0"]


def benchy_cmd(base_url, model, tokenizer, pp, tg, depths,
               levels=None, runs=2, no_cache=True):
    cmd = BENCHY + [
        "--base-url", f"{base_url}/v1",
        "--model", model,
        "--tokenizer", tokenizer,
        "--pp", str(pp),
        "--tg", str(tg),
        "--depth", *[str(d) for d in depths],
    ]
    if levels:
        cmd += ["--concurrency", *[str(c) for c in levels]]
    cmd += ["--runs", str(runs)]
    if no_cache:
        cmd += ["--no-cache"]
    cmd += [
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--format", "json",
    ]
    return cmd


def run(cmd, dry_run=False):
    print(f"$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0, "", 0.0
    start = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - start
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    return p.returncode, p.stdout, elapsed


def parse_result(stdout):
    """Extract the benchy result dict (printed last on stdout).

    benchy also prints single-line JSON progress objects mid-run, so a
    first-`{`-to-last-`}` span is not valid JSON (`Extra data`). Scan for
    top-level objects instead and take the last one shaped like a benchy
    result (has "benchmarks")."""
    stripped = stdout.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    dec = json.JSONDecoder()
    idx, n = 0, len(stdout)
    fallback, last = None, None
    while True:
        nxt = stdout.find("{", idx)
        if nxt == -1:
            break
        try:
            obj, end = dec.raw_decode(stdout, nxt)
        except json.JSONDecodeError:
            idx = nxt + 1
            continue
        if isinstance(obj, dict):
            if "benchmarks" in obj:
                last = obj
            elif fallback is None:
                fallback = obj
        idx = end
    if last is not None:
        return last
    if fallback is not None:
        return fallback
    raise ValueError("no JSON object found in benchy output")


def rows(result):
    """Flatten result benchmarks to (depth, conc, pp, tg, ttft) tuples."""
    out = []
    for b in result.get("benchmarks") or []:
        out.append((
            b.get("depth", b.get("context_size", 0)),
            b.get("concurrency", 1),
            (b.get("pp_throughput") or {}).get("mean"),
            (b.get("tg_throughput") or {}).get("mean"),
            (b.get("e2e_ttft") or {}).get("mean"),
        ))
    return out


def fmt(v, unit=""):
    return f"{v:.2f}{unit}" if isinstance(v, (int, float)) else "-"


def print_depth_table(result):
    print("\n| depth | pp t/s | tg t/s | ttft s |", flush=True)
    print("|:------|-------:|-------:|-------:|", flush=True)
    for depth, _c, pp, tg, ttft in rows(result):
        print(f"| {depth} | {fmt(pp)} | {fmt(tg)} | {fmt(ttft)} |", flush=True)


def print_conc_table(result):
    print("\n| conc | depth | pp t/s | tg t/s | ttft s |", flush=True)
    print("|:-----|------:|-------:|-------:|-------:|", flush=True)
    for depth, conc, pp, tg, ttft in rows(result):
        print(f"| {conc} | {depth} | {fmt(pp)} | {fmt(tg)} | {fmt(ttft)} |",
              flush=True)


def append_history(name, record):
    path = REPO_ROOT / "observability" / f"{name}_history.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Logged to observability/{name}_history.jsonl", flush=True)
