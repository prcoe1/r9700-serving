#!/usr/bin/env python3
"""Concurrent inter-token-latency (ITL) probe: how much does a big prefill
stall a co-decoding request?

Request A (small prompt, streaming decode) runs first; once A has delivered
`FIRE_AT` token batches, request B (huge prompt of unique random text,
streaming) is fired in a background thread. The probe records A's
inter-arrival times in three windows (before B fires, during B's prefill,
after B completes), B's TTFT and total time, and samples server metrics
(KV usage, running/waiting) throughout.

MTP note: one streamed batch can carry >1 accepted token, so arrivals are
token *batches*; tokens/arrival is reported from usage.

Usage:
    python3 benchmarks/conc_itl_probe.py [model] [big_prompt_tokens] [label]

Defaults: model=qwen3.8-27b, big_prompt_tokens=131072. Requires a reachable
vLLM server (default http://localhost:8180, override with VLLM_BASE_URL).
Runs from the host or inside the container.
"""
import json
import os
import random
import sys
import threading
import time

import requests

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8180")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b"
BIG_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
LABEL = sys.argv[3] if len(sys.argv) > 3 else f"big={BIG_TOKENS}"

A_PROMPT_TOKENS = 16384   # request A's context (rough; ~4 chars/token)
A_MAX_TOKENS = 512
B_MAX_TOKENS = 16
FIRE_AT = 32              # fire B after A has delivered this many batches
METRICS_INTERVAL = 0.5

WORDS = ("the quick brown fox jumps over lazy dog package manager kernel "
         "scheduler attention tensor batch token stream decode prefill "
         "memory cache buffer pointer matrix vector gradient descent "
         "compiler runtime library function module import export return "
         "value state update copy merge split chunk block page table "
         "address index offset length size limit bound check assert "
         "assertion error handler catch throw raise signal wait lock "
         "mutex thread process fork join yield resume pause clock timer "
         "deadline timeout retry backoff queue buffer pipe socket bind "
         "listen accept connect send receive buffer flush drain poll "
         "select epoll event loop callback promise future async await "
         "coroutine stack heap garbage collect allocate free release "
         "ownership borrow lifetime reference count weak strong cycle "
         "leak audit trace log metric gauge histogram counter sample "
         "percentile latency throughput capacity pressure eviction "
         "preemption admission reserve commit rollback transaction "
         "consensus quorum replica leader follower split brain fence "
         "epoch barrier reduce scatter gather broadcast allreduce "
         "precision floating point overflow underflow denormal rounding "
         "quantize scale bias activation softmax normalize temperature "
         "sampling topk topp nucleus penalty repeat stop sequence end "
         "begin start stop restart shutdown clean flush purge wipe "
         "probe baseline variance sample noise floor ceiling margin "
         "headroom slack cushion pad guard sentinel marker sentinel "
         "checksum digest hash fingerprint signature verify trust anchor").split()


def make_text(n_tokens_approx: int) -> str:
    rng = random.Random()
    chars_target = n_tokens_approx * 7  # this word list tokenizes ~7 chars/token
    parts, total = [], 0
    while total < chars_target:
        w = rng.choice(WORDS)
        parts.append(w)
        total += len(w) + 1
    return " ".join(parts)


def chat_stream(body: dict, read_timeout: int):
    """Open a streaming chat completion; returns (response, usage_holder).
    Caller must iterate the response and close it."""
    r = requests.post(BASE + "/v1/chat/completions", json=body, stream=True,
                      timeout=(10, read_timeout))
    if r.status_code != 200:
        r.close()
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r


def read_stream(r, times):
    """Consume response r's SSE stream; appends arrival timestamps to
    times and returns the list of usage dicts."""
    usage = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        obj = json.loads(data)
        if obj.get("usage"):
            usage.append(obj["usage"])
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                times.append(time.time())
    return usage


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def print_window(name, xs):
    if not xs:
        print(f"  {name:<14} n=0", flush=True)
        return
    n = len(xs)
    print(f"  {name:<14} n={n:<4} mean={sum(xs) / n * 1000:9.1f}ms "
          f"p50={pct(xs, 50) * 1000:9.1f}  p90={pct(xs, 90) * 1000:9.1f}  "
          f"p99={pct(xs, 99) * 1000:9.1f}  max={max(xs) * 1000:9.1f}",
          flush=True)


def main() -> None:
    print(f"probe: model={MODEL} label={LABEL} A~{A_PROMPT_TOKENS}tok "
          f"stream={A_MAX_TOKENS} B~{BIG_TOKENS}tok fire_at={FIRE_AT}",
          flush=True)

    metrics = {"kv": [], "running": [], "waiting": []}
    stop_metrics = threading.Event()

    def sample_metrics():
        while not stop_metrics.is_set():
            try:
                text = requests.get(BASE + "/metrics", timeout=5).text
                for line in text.splitlines():
                    if line.startswith("vllm:kv_cache_usage_perc{"):
                        metrics["kv"].append(float(line.rsplit("} ", 1)[-1]))
                    elif line.startswith("vllm:num_requests_running{"):
                        metrics["running"].append(
                            float(line.rsplit("} ", 1)[-1]))
                    elif line.startswith("vllm:num_requests_waiting{"):
                        metrics["waiting"].append(
                            float(line.rsplit("} ", 1)[-1]))
            except Exception:
                pass
            time.sleep(METRICS_INTERVAL)

    mthread = threading.Thread(target=sample_metrics, daemon=True)
    mthread.start()

    a_text = make_text(A_PROMPT_TOKENS)
    b_text = make_text(BIG_TOKENS)
    a_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      a_text + ("\n\nWrite a detailed essay of at least 400 "
                                "words explaining how a CPU cache hierarchy "
                                "works. Do not stop before 400 words.")}],
        "max_tokens": A_MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    b_body = dict(a_body)
    b_body["messages"] = [{"role": "user", "content":
                           b_text + "\n\nWhat is 19 times 21? Answer with just the number."}]
    b_body["max_tokens"] = B_MAX_TOKENS

    b_state = {}

    def run_b():
        b_state["fire"] = time.time()
        try:
            rb = chat_stream(b_body, read_timeout=900)
            b_state["usage"] = read_stream(rb, b_state.setdefault("times", []))
            rb.close()
            b_state["done"] = time.time()
        except Exception as e:
            b_state["err"] = repr(e)

    a_times = []
    a_usage = []
    ra = chat_stream(a_body, read_timeout=600)
    bthread = None
    try:
        for line in ra.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                a_usage.append(obj["usage"])
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    a_times.append(time.time())
                    if bthread is None and len(a_times) >= FIRE_AT:
                        bthread = threading.Thread(target=run_b)
                        bthread.start()
    finally:
        ra.close()
        if bthread is not None:
            bthread.join(timeout=1200)
        stop_metrics.set()

    if not a_usage:
        raise SystemExit("no usage returned for A")
    a_prompt, a_gen = a_usage[-1]["prompt_tokens"], a_usage[-1]["completion_tokens"]
    if bthread is None:
        raise SystemExit(f"B never fired (A delivered {len(a_times)} batches)")
    if "done" not in b_state:
        raise SystemExit(f"B failed: {b_state.get('err', 'unknown')}")
    b_prompt, b_gen = (b_state["usage"][-1]["prompt_tokens"],
                       b_state["usage"][-1]["completion_tokens"])

    fire_t, b_first, b_done = (b_state["fire"], b_state["times"][0],
                               b_state["done"])
    intervals = [b - a for a, b in zip(a_times, a_times[1:])]
    w1 = [iv for t, iv in zip(a_times[1:], intervals) if t < fire_t]
    w2 = [iv for t, iv in zip(a_times[1:], intervals)
          if fire_t <= t < b_done]
    w3 = [iv for t, iv in zip(a_times[1:], intervals) if t >= b_done]

    b_ttfb = b_first - fire_t
    print(f"A: prompt={a_prompt}tok gen={a_gen}tok "
          f"batches={len(a_times)} tokens/batch={a_gen / len(a_times):.2f}",
          flush=True)
    print(f"B: prompt={b_prompt}tok gen={b_gen}tok "
          f"TTFT={b_ttfb:7.1f}s total={b_done - fire_t:7.1f}s", flush=True)
    print(f"A inter-arrival (ms per batch):", flush=True)
    print_window("pre-B", w1)
    print_window("B-prefill", w2)
    print_window("post-B", w3)
    if w1 and w2:
        choke = pct(w2, 99) / pct(w1, 50)
        print(f"choke ratio (p99 during / p50 before): {choke:8.1f}x",
              flush=True)
    if metrics["kv"]:
        print(f"metrics: max_kv_usage={max(metrics['kv']):.3f} "
              f"max_running={max(metrics['running']):.0f} "
              f"max_waiting={max(metrics['waiting']):.0f}", flush=True)
    else:
        print("metrics: none sampled", flush=True)


if __name__ == "__main__":
    main()
