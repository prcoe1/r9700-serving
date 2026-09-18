import json, sys, time, requests

BASE = "http://localhost:8180/v1"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/kv_out.json"

PROMPTS = [
    "Write a detailed explanation of how a paged KV cache reduces memory fragmentation in vLLM.",
    "Explain in detail the difference between e4m3fn and e4m3fnuz fp8 formats and why the choice matters on RDNA4.",
    "Describe how tensor-parallel all-reduce works over PCIe peer-to-peer, step by step.",
    "Write a thorough essay on speculative decoding with DFlash: block diffusion, the selector, and losslessness.",
    "Explain how the attention block size in mamba-cache-mode align interacts with prefix caching on a hybrid GDN model.",
]

results = []
for p in PROMPTS:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 220,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(BASE + "/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    content = d["choices"][0]["message"]["content"]
    results.append({"prompt": p, "content": content,
                    "ttft_ms": int(d.get("usage", {}).get("prompt_tokens", 0)),
                    "elapsed_s": round(time.time() - t0, 2)})
    print(f"[{len(results)}] gen {len(content.split())} words, {results[-1]['elapsed_s']}s", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {OUT}", flush=True)
