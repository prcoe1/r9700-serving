#!/usr/bin/env python3
"""Probe for vllm-project/vllm#55766 (bad Mamba/GDN checkpoint -> NaN logits).

Upstream repro (v0.28.0, bf16, block 816, ngram; we run fp8, block 1600, MTP3):
  A: single user message, rendered prompt length exactly N*block + r with
     r in {4,6,8,10} (r in {6,8} most reliable; r<=3, r>=12, odd r clean).
     The prefill ending 4-10 tokens past a block boundary writes a bad
     Mamba checkpoint at the aligned boundary N*block.
  B: A's messages + A's reply + a fresh ~2000-token user message. B gets a
     block-aligned prefix hit up to N*block, restores the bad checkpoint,
     and must prefill >= 1 full block after it -> NaN logits from step 1
     (token-0 "!" spam to max_tokens, or empty).

Upstream notes speculation was NOT active at the failing step and that the
ngram dependency is untested; we run MTP3. A CLEAN result here is "not
triggered under MTP3", not proof of immunity.

Verdicts:
  CORRUPTED  B's output is !-spam/empty (or corrupted_requests_total moves)
  CLEAN      B produces normal text
  NO-HIT     B received no block-aligned prefix hit (inconclusive)

Controls: the r=20 case (outside the window) must be CLEAN; a cache_salt
re-run of a CORRUPTED B must be CLEAN (isolates the prefix-cache restore).

Usage (inside the container, from the repo):
    python3 benchmarks/nan_checkpoint_probe.py [model] [block_size] [--sweep]
block_size defaults to the live value from /metrics. --sweep adds r in {4,10}
(~50% repro rate upstream, so each gets 2 trials).
Exit: 0 = all clean, 1 = corruption reproduced, 2 = inconclusive (no hit).
"""
import os
import random
import sys
import time

import requests
from transformers import AutoTokenizer

BASE = os.environ.get("VLLM_BASE_URL", "http://localhost:8180")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b"
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SWEEP = "--sweep" in sys.argv
TOKENIZER_DIR = os.environ.get(
    "PROBE_TOKENIZER", "/home/philip/models-local/Qwen3.8-27B-FP8-kvscales")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "chat-templates", "qwen.jinja")
N_BLOCKS = 10
TRIGGER_R = [6, 8]
SWEEP_R = [4, 10]
CONTROL_R = 20
WORDS = (
    "river maple signal copper anchor velvet harbor piston ember quiver "
    "lantern cobble marrow fathom sparrow bazaar cinder grotto prairie "
    "tundra falcon mossy gravel silt amber hollow ridge canyon mesa dune "
    "glacier torrent ravine bluff scree moraine delta estuary fjord atoll "
    "reef lagoon mangrove savanna steppe meadow pasture orchard grove thicket "
    "clearing meadowstone hearthman threshold doorway windowsill rafters "
    "keystone archway buttress gable eaves lintel corbel frieze cornice "
    "balustrade pedestal plinth colonnade peristyle atrium cloister scriptorium"
).split()

rng = random.Random()
tok = None
tmpl = None


def _metric(name: str) -> float | None:
    text = requests.get(BASE + "/metrics", timeout=30).text
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[-1])
    return None


def geometry() -> dict:
    text = requests.get(BASE + "/metrics", timeout=30).text
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            for part in line.split("}")[0].split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip().strip('"')
            break
    return out


def render_len(messages: list[dict]) -> int:
    rendered = tok.apply_chat_template(
        messages, chat_template=tmpl, tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    return len(tok.encode(rendered, add_special_tokens=False))


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def build_content(target_len: int) -> str:
    base_len = render_len([user("x")])
    need = target_len - base_len
    assert need > 100, f"template overhead too large: base={base_len} target={target_len}"
    words = [rng.choice(WORDS) for _ in range(int(need * 2.5) + 64)]
    lo, hi = 0, len(words)

    def plen(k: int) -> int:
        return render_len([user(" ".join(words[:k]))])

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if plen(mid) < target_len - 48:
            lo = mid
        else:
            hi = mid - 1
    body = " ".join(words[:lo])
    npads = target_len - render_len([user(body)])
    for _ in range(5):
        cur = render_len([user(body + " pad" * npads)]) if npads > 0 else render_len([user(body)])
        gap = target_len - cur
        if gap == 0:
            break
        npads += gap
    assert gap == 0, f"could not hit exact length {target_len} (last gap {gap})"
    return body + (" pad" * npads if npads > 0 else "")


def fresh_content(min_tokens: int) -> str:
    parts = []
    total = 0
    for _ in range(40):
        chunk = " ".join(rng.choices(WORDS, k=256))
        parts.append(chunk)
        total += len(tok.encode(chunk, add_special_tokens=False))
        if total >= min_tokens:
            break
    return " ".join(parts)


def chat(messages: list[dict], max_tokens: int, salt: str | None = None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if salt:
        body["cache_salt"] = salt
    t0 = time.time()
    resp = requests.post(BASE + "/v1/chat/completions", json=body, timeout=900)
    dt = time.time() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json(), dt


def full_text(message: dict) -> str:
    return (message.get("reasoning") or message.get("reasoning_content") or "") + (message.get("content") or "")


def is_spam(text: str | None) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(set(t)) == 1:
        return True
    return t.count("!") / len(t) > 0.8


def run_case(r: int, trials: int = 1) -> list[str]:
    verdicts = []
    for trial in range(1, trials + 1):
        target = N_BLOCKS * BLOCK + r
        tag = f"r={r}" + (f" t{trial}" if trials > 1 else "")
        print(f"\n=== case {tag}: target prompt len = {N_BLOCKS}x{BLOCK} + {r} = {target} ===",
              flush=True)
        content = build_content(target)
        print(f"  A content built (verified rendered len = {target})", flush=True)
        mq0, mh0 = _metric("vllm:prefix_cache_queries_total"), _metric(
            "vllm:prefix_cache_hits_total")
        mc0 = _metric("vllm:corrupted_requests_total")
        a, a_dt = chat([user(content)], 300)
        ap = a["usage"]["prompt_tokens"]
        if ap != target:
            print(f"  ABORT case: server prompt_tokens={ap} != local {target} "
                  f"(template mismatch) -- probe invalid", flush=True)
            verdicts.append("TEMPLATE-MISMATCH")
            continue
        aout = full_text(a["choices"][0]["message"])
        if not aout.strip():
            print("  WARNING: A produced no text (itself corrupted?)", flush=True)
        print(f"  A: prompt={ap} gen={a['usage']['completion_tokens']} "
              f"time={a_dt:.1f}s stop={a['choices'][0]['finish_reason']} "
              f"out[:60]={aout[:60]!r}", flush=True)
        fresh = fresh_content(BLOCK + 400)
        b_msgs = [user(content),
                  {"role": "assistant", "content": aout},
                  user(fresh)]
        b, b_dt = chat(b_msgs, 64)
        bq1, bh1 = _metric("vllm:prefix_cache_queries_total"), _metric(
            "vllm:prefix_cache_hits_total")
        mc1 = _metric("vllm:corrupted_requests_total")
        dh = (bh1 or 0) - (mh0 or 0)
        dq = (bq1 or 0) - (mq0 or 0)
        dc = ((mc1 or 0) - (mc0 or 0)) if (mc0 is not None and mc1 is not None) else None
        bout = full_text(b["choices"][0]["message"])
        print(f"  B: prompt={b['usage']['prompt_tokens']} gen={b['usage']['completion_tokens']} "
              f"time={b_dt:.1f}s stop={b['choices'][0]['finish_reason']} "
              f"prefix q={dq:.0f} hits={dh:.0f} corrupted_delta={dc} "
              f"out[:80]={bout[:80]!r}", flush=True)
        if dc is not None and dc > 0:
            verdict = "CORRUPTED"
        elif dh < BLOCK:
            verdict = "NO-HIT"
        elif is_spam(bout):
            verdict = "CORRUPTED"
        else:
            verdict = "CLEAN"
        if verdict == "CORRUPTED":
            print(f"  -> retrying B with cache_salt (control: expect CLEAN) ...", flush=True)
            bs, bs_dt = chat(b_msgs, 64, salt=f"probe-{time.time_ns()}")
            bsout = full_text(bs["choices"][0]["message"])
            salt_ok = not is_spam(bsout)
            print(f"  B(salt): gen={bs['usage']['completion_tokens']} time={bs_dt:.1f}s "
                  f"out[:80]={bsout[:80]!r} -> {'CLEAN' if salt_ok else 'STILL-CORRUPTED'}",
                  flush=True)
            verdict += "" if salt_ok else "+SALT-ALSO-CORRUPTED"
        verdicts.append(verdict)
        print(f"  VERDICT {tag}: {verdict}", flush=True)
    return verdicts


def main() -> None:
    global tok, tmpl, BLOCK
    try:
        import vllm
        print(f"vllm {vllm.__version__}", flush=True)
    except Exception:
        pass
    g = geometry()
    live_block = int(g.get("block_size", 0))
    if not BLOCK:
        BLOCK = live_block
    print(f"model={MODEL} base={BASE}", flush=True)
    print(f"geometry: block_size={g.get('block_size')} "
          f"mamba_cache_mode={g.get('mamba_cache_mode')} "
          f"retention={g.get('prefix_cache_retention_interval')} "
          f"kv={g.get('cache_dtype')}", flush=True)
    if BLOCK != live_block:
        raise SystemExit(f"block size mismatch: arg={BLOCK} live={live_block}")
    if g.get("mamba_cache_mode") != "align":
        print(f"WARNING: mamba_cache_mode={g.get('mamba_cache_mode')} != align; "
              f"probe assumes align mode", flush=True)
    with open(TEMPLATE) as f:
        tmpl = f.read()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"tokenizer ready ({TOKENIZER_DIR}); template={TEMPLATE}", flush=True)

    results = {}
    for r in TRIGGER_R:
        results[r] = run_case(r)
    for r in (SWEEP_R if SWEEP else []):
        results[r] = run_case(r, trials=2)
    results[CONTROL_R] = run_case(CONTROL_R)

    print("\n=== summary ===")
    for r, vs in results.items():
        print(f"  r={r:<3}: {', '.join(vs)}")
    flat = [v for vs in results.values() for v in vs]
    if any(v.startswith("CORRUPTED") for v in flat):
        print("RESULT: CORRUPTION REPRODUCED (#55766 live on this stack)")
        sys.exit(1)
    if any(v == "NO-HIT" for v in flat):
        print("RESULT: INCONCLUSIVE (no block-aligned prefix hit to restore)")
        sys.exit(2)
    if any(v == "TEMPLATE-MISMATCH" for v in flat):
        print("RESULT: INCONCLUSIVE (local/server template mismatch)")
        sys.exit(2)
    print("RESULT: CLEAN (not triggered on this stack; note MTP3 vs upstream ngram)")
    sys.exit(0)


if __name__ == "__main__":
    main()
