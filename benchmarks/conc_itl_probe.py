#!/usr/bin/env python3
"""Concurrent inter-token-latency (ITL) probe: how much does a big prefill
stall co-decoding requests?

VICTIMS streaming decodes (small prompt each) start together and fill
`max_num_seqs - 1` server slots; once the first victim has delivered
`FIRE_AT` token batches, request B (huge prompt of unique random text,
streaming) is fired in a background thread to take the last slot. The probe
records each victim's inter-arrival times in three windows (before B fires,
during B's prefill, after B completes), B's TTFT and total time, and samples
server metrics (KV usage, running/waiting) throughout.

Victim count and prompt caps follow the live profile (`profile_config`):
victims default to `VLLM_MAX_NUM_SEQS - 1`, the big prompt is capped to fit
under `VLLM_MAX_MODEL_LEN` with headroom.

MTP note: one streamed batch can carry >1 accepted token, so arrivals are
token *batches*; tokens/arrival is reported from usage.

Usage:
    python3 benchmarks/conc_itl_probe.py [model] [big_prompt_tokens] [label] [victims]

Defaults: model from the live profile, big_prompt_tokens capped to the live
max-model-len, victims = max_num_seqs - 1. Requires a reachable vLLM server
(default http://localhost:8180, override with VLLM_BASE_URL). Runs from the
host or inside the container.
"""
import json
import os
import random
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_config as pc  # noqa: E402

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8180")
MODEL = sys.argv[1] if len(sys.argv) > 1 else pc.served_name()
CFG = pc.load_profile()
MAX_SEQS = pc.max_num_seqs(CFG)
MAX_LEN = pc.max_model_len(CFG)
BIG_TOKENS = (int(sys.argv[2]) if len(sys.argv) > 2
              else pc.fit_prompt(131072, MAX_LEN))
LABEL = sys.argv[3] if len(sys.argv) > 3 else f"big={BIG_TOKENS}"
VICTIMS = (int(sys.argv[4]) if len(sys.argv) > 4
           else max(1, MAX_SEQS - 1))

A_PROMPT_TOKENS = min(16384, pc.fit_prompt(16384, MAX_LEN))  # ~7 chars/token
A_MAX_TOKENS = 512
B_MAX_TOKENS = 16
FIRE_AT = 32              # fire B after a victim delivers this many batches
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


def make_text_chars(n_chars: int) -> str:
    rng = random.Random()
    parts, total = [], 0
    while total < n_chars:
        w = rng.choice(WORDS)
        parts.append(w)
        total += len(w) + 1
    return " ".join(parts)


def make_text(n_tokens_approx: int, ratio: float = 7.0) -> str:
    # this word list tokenizes ~7 chars/token; pass the calibrated ratio
    # once known so near-ceiling prompts don't overshoot max-model-len.
    return make_text_chars(int(n_tokens_approx * ratio))


def calibrate_ratio() -> float:
    """Measure chars-per-token for the word-list text on the live server
    with one small request, so the bully prompt lands just under the
    ceiling instead of 400-ing over it."""
    sample = make_text_chars(30000)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": sample}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(BASE + "/v1/chat/completions", json=body,
                      timeout=(10, 120))
    r.raise_for_status()
    prompt_tokens = r.json()["usage"]["prompt_tokens"]
    return len(sample) / max(1, prompt_tokens)


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
    slots = VICTIMS + 1
    over = slots - MAX_SEQS
    print(f"probe: model={MODEL} label={LABEL} victims={VICTIMS} "
          f"A~{A_PROMPT_TOKENS}tok stream={A_MAX_TOKENS} B~{BIG_TOKENS}tok "
          f"fire_at={FIRE_AT} slots={slots}/{MAX_SEQS} maxlen={MAX_LEN}",
          flush=True)
    if over > 0:
        print(f"probe: WARNING over-subscribes the server by {over} slot(s) "
              f"(victims+B={slots} > max_num_seqs={MAX_SEQS}) — B may queue",
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

    try:
        ratio = calibrate_ratio()
    except Exception as e:
        print(f"probe: calibration failed ({e}); using fallback ratio 7.0",
              flush=True)
        ratio = 7.0
    # 0.97 safety: land under the ceiling even if the sample ratio drifts.
    a_text = make_text(A_PROMPT_TOKENS, ratio * 0.97)
    b_text = make_text(BIG_TOKENS, ratio * 0.97)
    print(f"probe: calibrated chars/token={ratio:.2f}", flush=True)
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

    states = [{"times": [], "usage": [], "done": False, "err": None}
              for _ in range(VICTIMS)]
    fire_lock = threading.Lock()
    bthread = None

    def fire_b():
        nonlocal bthread
        with fire_lock:
            if bthread is None and any(len(s["times"]) >= FIRE_AT
                                       for s in states):
                bthread = threading.Thread(target=run_b)
                bthread.start()

    def run_victim(idx):
        st = states[idx]
        try:
            r = chat_stream(a_body, read_timeout=600)
        except Exception as e:
            st["err"] = repr(e)
            return
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    st["usage"].append(obj["usage"])
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        st["times"].append(time.time())
                        fire_b()
        except Exception as e:
            st["err"] = repr(e)
        finally:
            r.close()
            st["done"] = True

    vthreads = [threading.Thread(target=run_victim, args=(i,))
                for i in range(VICTIMS)]
    for t in vthreads:
        t.start()
    for t in vthreads:
        t.join(timeout=1200)
    if bthread is not None:
        bthread.join(timeout=1200)
    stop_metrics.set()

    for i, st in enumerate(states):
        if st["err"] and not st["times"]:
            raise SystemExit(f"victim {i} failed: {st['err']}")
        if not st["usage"]:
            raise SystemExit(f"no usage returned for victim {i}")
    if bthread is None:
        got = max(len(s["times"]) for s in states)
        raise SystemExit(f"B never fired (victims delivered <={got} batches)")
    if "done" not in b_state:
        raise SystemExit(f"B failed: {b_state.get('err', 'unknown')}")
    b_prompt, b_gen = (b_state["usage"][-1]["prompt_tokens"],
                       b_state["usage"][-1]["completion_tokens"])

    fire_t, b_first, b_done = (b_state["fire"], b_state["times"][0],
                               b_state["done"])
    b_ttfb = b_first - fire_t
    print(f"B: prompt={b_prompt}tok gen={b_gen}tok "
          f"TTFT={b_ttfb:7.1f}s total={b_done - fire_t:7.1f}s", flush=True)
    pool_w1, pool_w2 = [], []
    for i, st in enumerate(states):
        times = st["times"]
        a_prompt = st["usage"][-1]["prompt_tokens"]
        a_gen = st["usage"][-1]["completion_tokens"]
        intervals = [b - a for a, b in zip(times, times[1:])]
        w1 = [iv for t, iv in zip(times[1:], intervals) if t < fire_t]
        w2 = [iv for t, iv in zip(times[1:], intervals)
              if fire_t <= t < b_done]
        w3 = [iv for t, iv in zip(times[1:], intervals) if t >= b_done]
        pool_w1.extend(w1)
        pool_w2.extend(w2)
        tpb = a_gen / len(times) if times else float("nan")
        print(f"victim {i}: prompt={a_prompt}tok gen={a_gen}tok "
              f"batches={len(times)} tokens/batch={tpb:.2f}", flush=True)
        print_window(f"v{i} pre-B", w1)
        print_window(f"v{i} B-prefill", w2)
        print_window(f"v{i} post-B", w3)
    if pool_w1 and pool_w2:
        choke = pct(pool_w2, 99) / pct(pool_w1, 50)
        print(f"pooled choke ratio (p99 during / p50 before, "
              f"{VICTIMS} victim(s)): {choke:8.1f}x", flush=True)
    if metrics["kv"]:
        print(f"metrics: max_kv_usage={max(metrics['kv']):.3f} "
              f"max_running={max(metrics['running']):.0f} "
              f"max_waiting={max(metrics['waiting']):.0f}", flush=True)
    else:
        print("metrics: none sampled", flush=True)


if __name__ == "__main__":
    main()
