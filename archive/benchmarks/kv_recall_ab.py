#!/usr/bin/env python3
"""Long-context recall A/B: does calibrated fp8 KV beat scale-1.0 for recall?

Plants a distinctive fact ("The passphrase is <TOKEN>") deep in a long context
and asks the model to extract it, across three KV configs of Qwen3.8-27B TP=2:
  - bf16 KV    : reference
  - fp8 scale1 : fp8 KV at scale 1.0 (stock, no scales)
  - fp8 calib  : fp8 KV with calibrated q/k/v scales

This directly tests the AGENTS note that KV precision matters for long-context
recall. If scale 1.0's deep-layer V miscalibration (amax ~130 mis-ranged) hurts
recall, calibrated should retrieve at least as many needles as bf16 and more
than scale 1.0, especially at the deepest positions.

Memory is normal greedy generation (no prompt_logprobs), so true long contexts
(16K-64K) work. A few depths per config; one model load per config.

Usage (inside the vLLM container, TP=2, GPUs free):
    python benchmarks/kv_recall_ab.py \
        --calib-dir ... --stock-dir ... --ctx-tokens 32768 --depths 0.1,0.5,0.9
"""
import argparse
import glob
import os
import random

from vllm import LLM, SamplingParams

REPO = "/home/philip/r9700-serving"


def filler_text(need: int) -> list[int]:
    """Long realistic filler from repo text + repeated prose, tokenized."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["CALIB_DIR"])
    chunks = []
    for f in sorted(glob.glob(f"{REPO}/**/*.md", recursive=True)
                    + glob.glob(f"{REPO}/**/*.py", recursive=True)):
        try:
            chunks.append(open(f, encoding="utf-8", errors="replace").read())
        except OSError:
            continue
    ids = tok("\n\n".join(chunks))["input_ids"]
    pad = tok(
        "The observation deck houses a redundant array of independent "
        "disks whose striping and parity provide protection against "
        "individual drive failure, and the network fabric carries control "
        "traffic on a dedicated virtual channel so that congestion on the "
        "data plane never stalls the cluster scheduler."
    )["input_ids"]
    while len(ids) < need:
        ids.extend(pad)
    return ids[:need]


def build_prompt(filler: list[int], needle: str, depth_frac: float,
                 max_len: int) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["CALIB_DIR"])
    needle_txt = (f"\n\nImportant context to remember: The passphrase is "
                  f"{needle}. Remember it exactly.\n\n")
    q = ("\n\nQ: What is the passphrase mentioned in the context? "
         "Answer with only the passphrase string.\nA: ")
    needle_ids = tok(needle_txt)["input_ids"]
    q_ids = tok(q)["input_ids"]
    reserve = len(needle_ids) + len(q_ids) + 8
    # place the needle at ~depth_frac of the usable (max_len - reserve) span;
    # never truncate the needle or the question, only middle filler.
    head_len = int((max_len - reserve) * depth_frac)
    head = filler[:head_len]
    tail = filler[head_len:max_len - reserve]
    return head + needle_ids + tail + q_ids


def extract_answer(text: str, needle: str) -> bool:
    return needle in text


def run_config(filler: list[int], needles: list[str], depths: list[float],
               max_len: int) -> list[bool]:
    llm = LLM(
        model=os.environ["CALIB_DIR"] if os.environ.get("USE_CALIB") == "1"
        else os.environ["MODEL_DIR"],
        tokenizer=os.environ["CALIB_DIR"],
        tensor_parallel_size=2,
        max_model_len=max_len,
        enforce_eager=True,
        max_num_seqs=8,
        gpu_memory_utilization=float(os.environ.get("GMEM", "0.85")),
        kv_cache_dtype=os.environ.get("KV", "fp8"),
        attention_backend="ROCM_AITER_UNIFIED_ATTN",
        enable_prefix_caching=False,
    )
    prompts = [build_prompt(filler, n, d, max_len) for n, d in zip(needles, depths)]
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(max_tokens=32, temperature=0.0))
    del llm
    import torch
    torch.cuda.empty_cache()
    ok = []
    for o, n in zip(outs, needles):
        txt = o.outputs[0].text
        ok.append(extract_answer(txt, n))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--ctx-tokens", type=int, default=32768)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--reps", type=int, default=3,
                    help="distinct needles per depth for a steadier estimate")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    os.environ["CALIB_DIR"] = args.calib_dir
    os.environ["GMEM"] = str(args.gpu_memory_utilization)
    depths = [float(x) for x in args.depths.split(",")]

    rng = random.Random(1234)
    filler = filler_text(args.ctx_tokens)
    needles, ds = [], []
    for d in depths:
        for r in range(args.reps):
            # single alphanumeric token (no spaces) so substring extraction is robust
            needle = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz0123456789")
                             for _ in range(10))
            needles.append(needle)
            ds.append(d)
    print(f"context {args.ctx_tokens} tokens; {len(needles)} probes across "
          f"depths {depths}", flush=True)

    # bf16 reference
    os.environ.update({"KV": "bfloat16", "MODEL_DIR": args.calib_dir, "USE_CALIB": "1"})
    ref = run_config(filler, needles, ds, args.ctx_tokens)
    print(f"bf16    recall: {sum(ref)}/{len(ref)}", flush=True)
    # fp8 scale1
    os.environ.update({"KV": "fp8", "MODEL_DIR": args.stock_dir, "USE_CALIB": "0"})
    s1 = run_config(filler, needles, ds, args.ctx_tokens)
    print(f"scale1  recall: {sum(s1)}/{len(s1)}", flush=True)
    # fp8 calibrated
    os.environ.update({"KV": "fp8", "MODEL_DIR": args.calib_dir, "USE_CALIB": "1"})
    ca = run_config(filler, needles, ds, args.ctx_tokens)
    print(f"calib   recall: {sum(ca)}/{len(ca)}", flush=True)

    print("\n=== by depth ===", flush=True)
    for d in depths:
        idx = [i for i, x in enumerate(ds) if x == d]
        print(f"depth {d:>5}: bf16 {sum(ref[i] for i in idx)}/{len(idx)}  "
              f"scale1 {sum(s1[i] for i in idx)}/{len(idx)}  "
              f"calib {sum(ca[i] for i in idx)}/{len(idx)}", flush=True)


if __name__ == "__main__":
    main()
