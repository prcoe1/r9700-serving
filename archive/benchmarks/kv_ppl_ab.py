#!/usr/bin/env python3
"""Long-context fp8 KV quality A/B: is calibrated KV better than scale 1.0?

Scores a held-out continuation's perplexity under three KV configs of the same
Qwen3.8-27B checkpoint, all TP=2, on the same document:

  - bf16 KV   : full-precision reference (the quality floor / best case)
  - fp8 scale1: fp8 KV served at scale 1.0 (stock checkpoint, no scales)
  - fp8 calib : fp8 KV with calibrated q/k/v scales (ensure-kvscales)

The point: calibration is a *numerics-fidelity* fix, so it should pull the fp8
model's output distribution closer to the bf16 reference. We report per-config
perplexity over the SAME scored token subset, plus |PPL_cfg - PPL_bf16|. If
calibration is correct, PPL_calib < PPL_scale1 and it sits markedly closer to
the bf16 reference (deep-layer V amax ~130 is mis-ranged at scale 1.0).

Memory: prompt_logprobs materializes [seq, vocab] logits on GPU, so we score a
moderate-context window (--ctx-tokens) over many independent segments
(--segments), one generate call per segment, with low KV reservation
(--gpu-memory-utilization) to leave headroom. True-token logprobs are read from
the returned top-K dicts; only tokens present in ALL three configs are scored,
so the comparison is exact and fair.

Usage (inside the vLLM container, TP=2, GPUs free):
    python benchmarks/kv_ppl_ab.py \
        --calib-dir /home/philip/models-local/Qwen3.8-27B-FP8-kvscales \
        --stock-dir /home/philip/.cache/.../snapshots/<sha> \
        --ctx-tokens 4096 --cont-tokens 64 --segments 16
"""
import argparse
import glob
import math
import os

import torch

from vllm import LLM, SamplingParams

REPO = "/home/philip/r9700-serving"
K = 128


def build_document(need_tokens: int) -> list[int]:
    """Build a realistic long document from the repo's own .md/.py text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["CALIB_DIR"])
    chunks = []
    files = (glob.glob(f"{REPO}/**/*.md", recursive=True)
             + glob.glob(f"{REPO}/**/*.py", recursive=True)
             + glob.glob(f"{REPO}/**/*.jinja", recursive=True))
    for f in sorted(files):
        try:
            chunks.append(open(f, encoding="utf-8", errors="replace").read())
        except OSError:
            continue
    ids = tok("\n\n".join(chunks))["input_ids"]
    prose = (
        "Distributed inference on consumer RDNA4 balances arithmetic intensity "
        "against host bandwidth and kernel launch cost. The gated-delta net "
        "folds a linear recurrent state at each step, so long-context recall "
        "depends on how precisely that state is carried and how the sparse "
        "full-attention layers attend over the accumulated sequence. Each "
        "layer caches its keys and values in a paged buffer whose fp8 "
        "representation must keep the full dynamic range of the activations it "
        "stores; a scale tuned to the small-magnitude bulk under-quantizes the "
        "occasional large residual and flattens the sharp attention peaks that "
        "encode the earliest tokens. Calibrating the scale from the observed "
        "per-layer maxima preserves those peaks, trading a little "
        "representational headroom for faithful reconstruction of the cache "
        "that drives the final softmax. The empirical loss of an uncalibrated "
        "cache shows up as a systematic shift in the predictive distribution, "
        "widest where the hidden states are largest."
    )
    while len(ids) < need_tokens:
        ids.extend(tok(prose)["input_ids"])
    return ids[:need_tokens]


def make_segments(doc: list[int], ctx: int, cont: int, n: int) -> list[list[int]]:
    """Return n contiguous windows of ctx+cont tokens, spaced across the doc."""
    stride = (len(doc) - (ctx + cont)) // n
    segs = []
    start = 0
    for _ in range(n):
        segs.append(doc[start:start + ctx + cont])
        start += stride
    return segs


def score_segments(segments: list[list[int]], cont: int) -> list[dict[int, float]]:
    """Under one LLM, score each segment's trailing `cont` tokens.

    Returns one dict per segment mapping token index -> true-token logprob, for
    the last `cont` tokens (each scored against the preceding context). Tokens
    whose true token is missing from the returned top-K dict are omitted; the
    caller intersects across configs so the PPL comparison is exact and fair.
    """
    llm = LLM(
        model=os.environ["CALIB_DIR"] if os.environ.get("USE_CALIB") == "1"
        else os.environ["MODEL_DIR"],
        tokenizer=os.environ["CALIB_DIR"],
        tensor_parallel_size=2,
        max_model_len=max(len(s) + 8 for s in segments),
        enforce_eager=True,
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("GMEM", "0.5")),
        kv_cache_dtype=os.environ.get("KV", "fp8"),
        attention_backend="ROCM_AITER_UNIFIED_ATTN",
        enable_prefix_caching=False,
        max_logprobs=K,
    )
    results = []
    for seg in segments:
        score_start = len(seg) - cont
        outs = llm.generate({"prompt_token_ids": seg},
                            SamplingParams(max_tokens=1, temperature=0.0,
                                           prompt_logprobs=K))
        plp = outs[0].prompt_logprobs
        d = {}
        for i in range(score_start, len(seg)):
            pos = plp[i]
            if not pos:
                continue
            ent = pos.get(seg[i])
            if ent is not None:
                d[i] = ent.logprob
        results.append(d)
    del llm
    torch.cuda.empty_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--ctx-tokens", type=int, default=4096)
    ap.add_argument("--cont-tokens", type=int, default=64)
    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = ap.parse_args()
    os.environ["CALIB_DIR"] = args.calib_dir
    os.environ["GMEM"] = str(args.gpu_memory_utilization)

    doc = build_document(args.segments * (args.ctx_tokens + args.cont_tokens) + 64)
    segments = make_segments(doc, args.ctx_tokens, args.cont_tokens, args.segments)
    print(f"document: {len(doc)} tokens; {args.segments} segments x "
          f"{args.ctx_tokens} ctx + {args.cont_tokens} cont", flush=True)

    # 1) bf16 reference
    os.environ.update({"KV": "bfloat16", "MODEL_DIR": args.calib_dir, "USE_CALIB": "1"})
    bf16 = score_segments(segments, args.cont_tokens)
    print(f"bf16: scored {sum(len(d) for d in bf16)} continuation tokens", flush=True)
    # 2) fp8 scale 1.0
    os.environ.update({"KV": "fp8", "MODEL_DIR": args.stock_dir, "USE_CALIB": "0"})
    scale1 = score_segments(segments, args.cont_tokens)
    print(f"fp8 scale1: scored {sum(len(d) for d in scale1)} tokens", flush=True)
    # 3) fp8 calibrated
    os.environ.update({"KV": "fp8", "MODEL_DIR": args.calib_dir, "USE_CALIB": "1"})
    calib = score_segments(segments, args.cont_tokens)
    print(f"fp8 calib: scored {sum(len(d) for d in calib)} tokens", flush=True)

    # intersect per-segment across configs (exact + fair)
    ce = {"bf16": 0.0, "fp8_scale1": 0.0, "fp8_calib": 0.0}
    common_total = 0
    for a, b, c in zip(bf16, scale1, calib):
        for i in set(a) & set(b) & set(c):
            ce["bf16"] += a[i]
            ce["fp8_scale1"] += b[i]
            ce["fp8_calib"] += c[i]
            common_total += 1
    if common_total == 0:
        raise SystemExit("no common scored tokens")
    ppl = {k: math.exp(-v / common_total) for k, v in ce.items()}
    print(f"\ncommon scored positions: {common_total}", flush=True)
    print("=== SUMMARY (PPL over common subset) ===", flush=True)
    ref = ppl["bf16"]
    for k in ("fp8_scale1", "fp8_calib"):
        print(f"{k:12s}: PPL {ppl[k]:.4f}   |PPL-bf16| = "
              f"{abs(ppl[k] - ref):.4f}", flush=True)
    d1 = abs(ppl["fp8_scale1"] - ref)
    d2 = abs(ppl["fp8_calib"] - ref)
    print(f"calibrated distance to bf16: {d2:.4f} vs scale-1.0 {d1:.4f} "
          f"({'-' if d2 <= d1 else '+'}{(1 - d2 / max(d1, 1e-9)) * 100:.1f}% "
          f"closer to bf16)", flush=True)


if __name__ == "__main__":
    main()
