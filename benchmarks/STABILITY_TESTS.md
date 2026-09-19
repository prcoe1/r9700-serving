# Stability & Long-Context Tests

Operational health checks for the vLLM server — catches crashes, memory errors,
garbage output, and token-loop degeneration under sustained load.

> Latest full validation:
> [`2026-09-19_qwen3.8-27b_stability_depth_conc.md`](2026-09-19_qwen3.8-27b_stability_depth_conc.md)
> (v0.29.0 + #40709 patch, qwen3.8-27b: stress 400/400, long-context 10/10,
> thinking 10/10, 0 errors). Scripts below name older qwen3.6 models — substitute
> the live served name when running.

## Quick health

```sh
# Verify the server is running and responding
just check && curl -sf http://localhost:8180/health && echo "OK"

# Check GPU health inside the container
just exec rocm-smi

# Follow live logs during testing
just logs
```

## Short benchmark (smoke test)

```sh
just --set model qwen3.6-27b up
just --set model qwen3.6-27b bench
```

Runs `llama-benchy` with pp=2048, tg=32 and tg=128 across 3 runs. Verifies
coherence (no garbage output) and baseline throughput.

## Sustained load stress (400 requests)

Simulates a busy production server with two concurrent workers sending 200
requests each (2048-token generation), catching HTTP errors, timeouts, and crashes.

```sh
just exec bash -c 'python3 << '\''PYEOF'\''
import requests, threading, time, sys

errors = []
successes = []
lock = threading.Lock()

def worker(i):
    for r in range(200):
        try:
            body = {
                "model": "qwen3.6-27b",
                "messages": [{"role": "user", "content": f"Write a detailed essay on topic {i*200+r}."}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            resp = requests.post("http://localhost:8180/v1/chat/completions", json=body, timeout=120)
            if resp.status_code != 200:
                with lock: errors.append(f"W{i} R{r}: HTTP {resp.status_code}")
                break
            with lock:
                successes.append(len(resp.json()["choices"][0]["message"]["content"].split()))
                if (i*200+r) % 20 == 0:
                    print(f"W{i} R{r}/200 OK", flush=True)
        except Exception as e:
            with lock: errors.append(f"W{i} R{r}: {e}")
            break
        time.sleep(0.2)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
for t in threads: t.start()
for t in threads: t.join()

print(f"\nCompleted: {len(successes)} / 400 requests")
if errors:
    print(f"Errors: {len(errors)}")
    for e in errors[:5]: print(f"  {e}")
if successes:
    print(f"Avg words/gen: {sum(successes)/len(successes):.0f}")
    print(f"Min: {min(successes)}, Max: {max(successes)}")
print("Done.")
PYEOF'
```

**Baseline results** (qwen3.6-27b, 2× R9700, vLLM 0.27.0 build; source pin
is now 0.27.1):
- Completed: **400 / 400**, Errors: **0**
- Avg words/gen: ~259, Min: 33, Max: 1451

## Long-context generation (54k input, 10 runs)

Stresses attention over large cached KV contexts. Each request generates up to
4096 tokens from a ~54k-token prompt, checking for:
- HTTP errors or timeouts
- Degenerate token loops (measured via vocabulary richness)
- Crash or OOM under sustained long-context pressure

```sh
just exec bash -c 'python3 << '\''PYEOF'\''
import requests, time, sys, random

results = []

def gen_long_context(base_len):
    topics = [
        "machine learning optimization and gradient descent algorithms",
        "quantum computing error correction and fault tolerance systems",
        "distributed systems consistency models and consensus protocols",
        "climate change modeling and carbon cycle dynamics in ocean-atmosphere systems",
        "neuroscience of visual perception and cortical processing pathways",
        "cryptocurrency consensus mechanisms and proof of stake validation",
        "protein folding predictions and molecular dynamics simulation methods",
        "natural language processing transformer architectures and attention mechanisms",
        "renewable energy grid management and battery storage optimization",
        "spatial statistics and geospatial clustering algorithms for urban planning"
    ]
    parts = []
    for i in range(base_len):
        parts.append(f"Section {i}: Analysis of {random.choice(topics)}. "
                     f"This comprehensive review examines recent developments and findings. "
                     f"The research methodology employed quantitative analysis across multiple datasets. "
                     f"Key results indicate significant correlations between variables. "
                     f"The implications for future work are substantial and warrant careful consideration. "
                     f"Additional experiments confirmed the initial findings with statistical significance.")
    return "\n\n".join(parts)

context = gen_long_context(800)
print(f"Context: {len(context)} chars (~{len(context)//4} est tokens)", flush=True)

for i in range(10):
    print(f"\n--- Request {i+1}/10 ---", flush=True)
    prompt = f"""Analyze the following research compilation. Provide a structured summary:

{context}

Provide your analysis with these sections:
1. Overview  2. Key Themes  3. Methodology Assessment
4. Notable Findings  5. Gaps and Future Directions  6. Conclusion"""

    body = {
        "model": "qwen3.6-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    start = time.time()
    resp = requests.post("http://localhost:8180/v1/chat/completions", json=body, timeout=600)
    elapsed = time.time() - start

    status = resp.status_code
    if status != 200:
        print(f"ERROR: HTTP {status}: {resp.text[:150]}")
        continue

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    input_tokens = usage.get("prompt_tokens", "?")
    output_tokens = usage.get("completion_tokens", "?")
    words = len(content.split())
    unique_words = len(set(content.lower().split()))
    vocab_rich = (unique_words / max(words, 1)) * 100

    results.append({
        "input": input_tokens,
        "output": output_tokens,
        "words": words,
        "vocab_pct": vocab_rich,
        "elapsed": elapsed
    })

    print(f"  Time: {elapsed:.1f}s | Prompt: {input_tokens} | Gen: {output_tokens} | Words: {words} | Vocab: {vocab_rich:.1f}%")
    print(f"  Start: {content[:60].strip()}")
    print(f"  End: ...{content[-60:].strip()}")

print("\n=== RESULTS ===")
ok = [r for r in results if r["output"] != "?"]
print(f"Completed: {len(ok)}/10")
if ok:
    avg_input = sum(r["input"] for r in ok) / len(ok)
    avg_gen = sum(r["output"] for r in ok) / len(ok)
    avg_time = sum(r["elapsed"] for r in ok) / len(ok)
    avg_vocab = sum(r["vocab_pct"] for r in ok) / len(ok)
    print(f"Avg input tokens: {avg_input:.0f}")
    print(f"Avg gen tokens: {avg_gen:.0f}")
    print(f"Avg time: {avg_time:.1f}s")
    print(f"Avg vocab richness: {avg_vocab:.1f}%")
errors = 10 - len(ok)
print(f"Errors: {errors}/10")
print("Done.")
PYEOF'
```

**Baseline results** (qwen3.6-27b, 2× R9700, vLLM 0.27.0 build; source pin
is now 0.27.1):

| # | Time | Input Tok | Gen Tok | Words | Vocab Richness |
|---|------|-----------|---------|-------|----------------|
| 1 | 72.4s | 53,981 | 4,096 | 1,024 | 89.6% |
| 2 | 20.2s | 53,981 | 1,177 | 746 | 56.7% |
| 3 | 16.4s | 53,981 | 918 | 611 | 60.2% |
| 4 | 17.4s | 53,981 | 1,003 | 685 | 56.6% |
| 5 | 22.1s | 53,981 | 1,295 | 799 | 57.7% |
| 6 | 20.1s | 53,981 | 1,117 | 746 | 55.6% |
| 7 | 16.5s | 53,981 | 895 | 618 | 57.9% |
| 8 | 17.3s | 53,981 | 985 | 684 | 57.0% |
| 9 | 19.2s | 53,981 | 1,065 | 669 | 60.2% |
| 10 | 20.1s | 53,981 | 1,033 | 700 | 59.0% |

- **Completed:** 10/10, **0 errors**
- First request ~72s is Triton kernel warmup; subsequent ~16-22s
- Vocab richness ≥55% across all runs (no degenerative repetition)
- All responses coherent, varied conclusions, no garbage output

## Thinking-enabled long generation (32k budget, 10 runs)

Stresses the reasoning path with `enable_thinking: true`. Each request sends a
~13.5k-token prompt with a 32768-token generation budget and checks that
thinking actually happens, output stays coherent, and nothing crashes.

```sh
just exec python3 - <<'PYEOF'
import requests, time, sys, random

results = []

topics = [
    "machine learning optimization and gradient descent algorithms",
    "quantum computing error correction and fault tolerance systems",
    "distributed systems consistency models and consensus protocols",
    "climate change modeling and carbon cycle dynamics in ocean-atmosphere systems",
    "neuroscience of visual perception and cortical processing pathways",
    "cryptocurrency consensus mechanisms and proof of stake validation",
    "protein folding predictions and molecular dynamics simulation methods",
    "natural language processing transformer architectures and attention mechanisms",
    "renewable energy grid management and battery storage optimization",
    "spatial statistics and geospatial clustering algorithms for urban planning",
    "reinforcement learning for robotics control and decision making",
    "computational biology and gene regulatory network inference",
    "autonomous vehicle perception and sensor fusion",
    "federated learning privacy and differential privacy mechanisms",
    "graph neural networks for social network analysis",
    "large language model alignment and safety evaluation",
    "computer vision object detection and image segmentation",
    "time series forecasting and anomaly detection",
    "bayesian optimization for hyperparameter tuning",
    "cryptographic protocols and zero-knowledge proofs"
]

def gen_long_context(base_len):
    parts = []
    for i in range(base_len):
        parts.append(f"Section {i}: Analysis of {random.choice(topics)}. "
                     f"This comprehensive review examines recent developments and findings. "
                     f"The research methodology employed quantitative analysis across multiple datasets. "
                     f"Key results indicate significant correlations between variables. "
                     f"The implications for future work are substantial and warrant careful consideration. "
                     f"Additional experiments confirmed the initial findings with statistical significance. "
                     f"Future directions include expanding the dataset, improving model robustness, "
                     f"and validating the approach on additional benchmarks.")
    return "\n\n".join(parts)

context = gen_long_context(90)
print(f"Context: {len(context)} chars (~{len(context)//4} est tokens)", flush=True)

for i in range(10):
    print(f"\n--- Request {i+1}/10 ---", flush=True)
    prompt = f"""Analyze the following research compilation in extreme detail. Provide an exhaustive, multi-section analysis:

{context}

Provide your analysis with these sections, each with multiple subsections:
1. Executive Summary
2. Key Themes and Patterns
3. Methodology Assessment
4. Notable Findings and Results
5. Comparative Analysis Across Topics
6. Gaps and Future Directions
7. Conclusion and Recommendations

For EACH section write at least 3 substantial paragraphs with concrete details, cross-references between topics, and specific quantitative observations. Be thorough — cover every topic in the compilation."""

    body = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32768,
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": True}
    }

    start = time.time()
    resp = requests.post("http://localhost:8180/v1/chat/completions", json=body, timeout=600)
    elapsed = time.time() - start

    status = resp.status_code
    if status != 200:
        print(f"ERROR: HTTP {status}: {resp.text[:150]}")
        continue

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    reasoning = message.get("reasoning", "")
    usage = data.get("usage", {})

    input_tokens = usage.get("prompt_tokens", "?")
    output_tokens = usage.get("completion_tokens", "?")
    words = len(content.split())
    unique_words = len(set(content.lower().split()))
    vocab_rich = (unique_words / max(words, 1)) * 100

    results.append({
        "input": input_tokens,
        "output": output_tokens,
        "words": words,
        "reasoning_chars": len(reasoning),
        "vocab_pct": vocab_rich,
        "elapsed": elapsed
    })

    print(f"  Time: {elapsed:.1f}s | Prompt: {input_tokens} | Gen: {output_tokens} | Words: {words} | Vocab: {vocab_rich:.1f}% | Reasoning chars: {len(reasoning)}")
    print(f"  Thinking present: {'yes' if reasoning else 'NO'}")
    print(f"  Start: {content[:60].strip()}")
    print(f"  End: ...{content[-60:].strip()}")

print("\n=== RESULTS ===")
ok = [r for r in results if r["output"] != "?"]
print(f"Completed: {len(ok)}/10")
if ok:
    avg_input = sum(r["input"] for r in ok) / len(ok)
    avg_gen = sum(r["output"] for r in ok) / len(ok)
    avg_time = sum(r["elapsed"] for r in ok) / len(ok)
    avg_vocab = sum(r["vocab_pct"] for r in ok) / len(ok)
    thinking = sum(1 for r in ok if r["reasoning_chars"] > 0)
    print(f"Avg input tokens: {avg_input:.0f}")
    print(f"Avg gen tokens: {avg_gen:.0f}")
    print(f"Avg time: {avg_time:.1f}s")
    print(f"Avg vocab richness: {avg_vocab:.1f}%")
    print(f"Thinking enabled: {thinking}/10")
errors = 10 - len(ok)
print(f"Errors: {errors}/10")
print("Done.")
PYEOF
```

**Baseline results** (qwen3.6-35b-a3b, 2× R9700, vLLM 0.27.0 build; source pin
is now 0.27.1):

| metric | result |
|--------|--------|
| Completed | 10/10 |
| Errors | 0 |
| Avg input tokens | 13,500 |
| Avg gen tokens | 5,643 (max 7,029) |
| Avg time | 64.2s |
| Avg vocab richness | 52.1% |
| Thinking enabled | 10/10 (8.9k–19.5k reasoning chars each) |

- **Thinking confirmed:** every request produced a `reasoning` field (thinking
  enabled via `"chat_template_kwargs":{"enable_thinking":true}`), so the
  model's bundled qwen3 template partitions thinking from the final answer correctly.
- **Note on the 32k budget:** the model never hit 32768 output tokens — it
  emitted EOS at ~5.6k avg (max ~7k) once responses reached a coherent stopping
  point. This validates thinking-path stability, but did **not** exercise the
  full generation ceiling. To actually stress 32k output, the prompt must demand
  exhaustive multi-section output (e.g. 100 sections × ~300 words).
- All outputs were structured, coherent, and ended with varied conclusions — no
  garbage or token loops.

## Prefix-cache hit-rate probe (hybrid GDN)

Checks whether `--enable-prefix-caching` actually reuses KV across multi-turn
conversations. Hybrid GDN models run `mamba_cache_mode=align` by default, where
prefix caching is currently broken (vllm-project/vllm#45238): hits stay at 0%
whenever the align-mode Mamba checkpoint lands in request-unique tokens.

```sh
just exec python /home/philip/r9700-serving/benchmarks/prefix_cache_probe.py [model] [turns]
```

Reports per-turn `vllm:prefix_cache_queries_total` / `hits_total` deltas, the
live cache geometry (`block_size`, `mamba_cache_mode` from
`vllm:cache_config_info`), and a final coherence check (the model must retain
a secret planted in turn 1).

**Interpretation.** Trigger condition with the observed `block_size=1600`:
`floor((prompt_len-1)/1600)*1600 > shared_prefix_len` → 0% hits. A **non-zero
hit rate** on this probe is the signal the align-mode checkpoint fix landed
upstream and is worth carrying/keeping (see AGENTS.md watchlist `#45238`).
Record the result after every vLLM bump/restart.

**Baseline** (qwen3.8-27b, vLLM 0.27.1 + patches, 2026-08-20): 0% hits across
all 4 turns (`queries` climbing, `hits=0`), coherence PASS.

## What counts as a failure

| symptom | meaning |
|---------|---------|
| HTTP 500/503 | Server crash or OOM |
| Timeout (>600s) | GPU hung or stuck inference |
| Output is garbled/repetitive | Token-loop bug (MTP or otherwise) |
| `llama-benchy` coherence test fails | Server producing nonsensical tokens |
| `rocm-smi` reports GPU fault | Hardware or driver issue |
