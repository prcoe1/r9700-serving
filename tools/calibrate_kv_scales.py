#!/usr/bin/env python3
"""Calibrate fp8 KV-cache q/k/v scales for a Qwen3.5/3.6/3.8 hybrid (GDN +
full-attention) checkpoint that ships without k/v_scale values.

On vLLM >= 0.28 (--calculate-kv-scales removed upstream) an fp8 KV cache with
no checkpoint scales serves at scale 1.0 ("correct bytes, wrong numbers"). This
tool runs the checkpoint over a small diverse text corpus with vLLM offline,
hooking Qwen3NextAttention._project_qkv_gate to record the per-layer amax of
post-norm/post-RoPE q, post-norm/post-RoPE k, and raw v. Each worker appends
its observations to a shared log file (CALIB_KV_LOG); the main process
aggregates and writes amax/448 (e4m3fn convention -- vLLM doubles it internally
for the e4m3fnuz runtime scale on RDNA) as scalar sidecar tensors plus the
matching model.safetensors.index.json weight_map entries.

Coverage is BOTH the main full-attention layers (indices 3,7,...,63 for these
GDN hybrids) AND the MTP prediction-head layer(s) (mtp.layers.*): the MTP head
is a stack of its own full-attention decoder layers with separate q/k/v
projections that cache fp8 KV just like the main model. The LLM() below enables
MTP spec-decode specifically so the head layers are instantiated and run --
otherwise they never exist and their KV stays at scale 1.0. Scale tensor names
are derived from the authoritative model.safetensors.index.json q_proj keys
(main: model.language_model.layers.N.self_attn; head: mtp.layers.N.self_attn),
NOT from the hook's attn.layer_name, which uses a different
(language_model.model...) namespace.

Bootstrap ordering: the calibrator must run on a model dir whose index does NOT
yet reference the sidecar (i.e. a fresh setup_kvscales copy), because LLM()
loads weights (index) before the sidecar exists. To re-calibrate an already-
calibrated copy, first restore the pristine index from the HF snapshot and
delete the sidecar.

Usage (inside the vLLM container, TP=2):
    CALIB_KV_LOG=/tmp/kvscale.log python tools/calibrate_kv_scales.py <model_dir>
"""
import argparse
import json
import os
import sys

import torch

# Monkeypatch BEFORE LLM() is constructed so forked TP workers inherit it
# (VLLM_WORKER_MULTIPROC_METHOD defaults to fork). The attention layer is
# Qwen3NextAttention, reused by the qwen3_5 hybrid VLM text tower.
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

_LOG = os.environ.get("CALIB_KV_LOG", "/tmp/kvscale.log")

_orig = Qwen3NextAttention._project_qkv_gate


def _patched(self, qkv, positions):
    q, k, v, gate = _orig(self, qkv, positions)
    # layer prefix lives on the inner Attention module (self.attn.layer_name),
    # e.g. model.language_model.layers.3.self_attn.attn (main) or
    # model.language_model.mtp.layers.0.self_attn.attn (MTP prediction head).
    # Both the main full-attention layers and the MTP head's own full-attention
    # decoder layer(s) cache fp8 KV, so both must be calibrated.
    name = getattr(getattr(self, "attn", None), "layer_name", None) or "?"
    with open(_LOG, "a") as f:
        for tensor, suffix in ((q, "q"), (k, "k"), (v, "v")):
            amax = float(tensor.abs().max().detach().float())
            f.write(f"{name}:{suffix}\t{amax}\n")
    return q, k, v, gate


Qwen3NextAttention._project_qkv_gate = _patched

from vllm import LLM, SamplingParams  # noqa: E402

E4M3FN_MAX = 448.0


def _corpus() -> list[str]:
    prose = (
        "The Radeon R9700 is a workstation GPU built on the RDNA4 architecture, "
        "marketed as the Radeon AI PRO R9700. It pairs the gfx1201 die with 16 GiB "
        "of GDDR6 across a 256-bit bus, delivering roughly 39 TFLOPS of fp32 and "
        "substantially higher fp8 tensor throughput via the WMMA units. Its primary "
        "target is local inference of language models, where the lack of a native "
        "bf16 matrix core path on consumer RDNA4 makes fp8 and int8 quantized GEMMs "
        "the practical compute path. vLLM does not ship support for this card out of "
        "the box, so enthusiasts build a ROCm stack with PyTorch, Triton and AITER "
        "compiled for gfx1201, often pinned to the exact minor versions the vLLM "
        "release was validated against. The cold-start compile of the gated-delta "
        "net Triton kernels used to dominate the first boot, which is why a "
        "persistent kernel cache and pre-warmed JIT infrastructure matter so much "
        "for interactive use."
    )
    code = (
        "def build_schedule(requests, block_size):\n"
        "    blocks, active = {}, []\n"
        "    for r in sorted(requests, key=lambda x: x.priority, reverse=True):\n"
        "        need = (r.tokens + block_size - 1) // block_size\n"
        "        pages = [alloc() for _ in range(need)]\n"
        "        for p in pages:\n"
        "            p.seq_ids.add(r.req_id)\n"
        "        blocks[r.req_id] = pages\n"
        "        active.append(r)\n"
        "    return blocks, active\n"
    )
    jsonish = json.dumps({
        "served_model": "Qwen3.8-27B-FP8",
        "engine": {"kv_cache_dtype": "fp8", "mamba_cache_mode": "align",
                   "tensor_parallel_size": 2, "max_model_len": 262144,
                   "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
                   "speculative": {"method": "dflash", "num_speculative_tokens": 7}},
        "metrics": {"prefix_cache_hits": 0, "ttft_ms": 912},
    }, indent=1)
    mathish = (
        "Let us show that the residual stream remains bounded. Since the input "
        "layernorm projects onto a unit RMS norm, the norm is sqrt(H) after "
        "normalization. The attention output is a convex combination of value "
        "vectors, so its norm is at most the max over tokens of the value norm. "
        "Combining these with the gating term bounded by one, the updated hidden "
        "state satisfies a triangle inequality that, for stable weight matrices "
        "and bounded activations, yields a contraction factor strictly less than "
        "one, proving Lipschitz stability over long sequences."
    )
    code_asst = (
        "<|im_start|>system\nYou are a helpful coding assistant. Return a precise "
        "diff with exact before and after lines.\n<|im_end|>\n"
        "<|im_start|>user\nRefactor the all-reduce to use a one-shot push at TP=2 "
        "and make large payloads quantize to block-scaled fp8 while keeping small "
        "messages bit-exact.\n<|im_end|>\n<|im_start|>assistant\n"
    )
    kernel = (
        "ROCM_AITER_UNIFIED_ATTN unifies prefill and decode on one Triton kernel "
        "set. On RDNA4 the vendor flash kernels carry no gfx1201 device code, so "
        "attention falls to the AITER unified path, whose decode tile must fit the "
        "64 KiB workgroup shared memory even at head_size 256 with a two-byte KV "
        "cache; otherwise warmup fails with an OutOfResources error. The fix caps "
        "the staged tile size for bf16 KV, trading a little tile width for a "
        "workable kernel."
    )
    tool_call = (
        "API::<tool_call>\n"
        '{"name":"search_records","arguments":{"query":"prefix cache hit rate",'
        '"start":"2026-08-01","end":"2026-08-21","aggregate":"per_day"}}\n'
        "</tool_call>\nThe search returned 21 days of telemetry showing the "
        "align-mode prefix cache holding a steady zero-percent hit rate until the "
        "checkpoint fix landed, after which multi-turn probes moved off zero."
    )
    notes = (
        "# Distributed inference notes\n## All-reduce\n"
        "- TP2 fires ~128 all-reduces per generated token; RCCL's ~28us floor is "
        "launch plus protocol, not bandwidth.\n"
        "- A native one-shot P2P push wins at every size at TP2.\n"
        "## KV cache\n"
        "- fp8 e4m3 halves cache bytes; bf16 keeps full precision.\n"
        "- Uncalibrated scales serve at 1.0, which is wrong when K reaches ~24.\n"
        "## Speculative decoding\n"
        "- MTP drafts reuse the target checkpoint; DFlash uses a drafter.\n"
    )
    # repeat each to get a bit more prefill+decode coverage per sample
    out = []
    for sample in (prose, code, jsonish, mathish, code_asst, kernel, tool_call, notes):
        out.extend([sample] * 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None, help="dir to write sidecar+index")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    out_dir = args.out or args.model_dir

    if os.path.exists(_LOG):
        os.remove(_LOG)

    # Enable MTP spec-decode so the MTP prediction-head layer(s) are
    # instantiated; otherwise the head's own full-attention layer (which caches
    # fp8 KV) never runs and its scales would be missing. num_speculative_tokens
    # is the draft window length; the head has num_nextn_predict_layers layers.
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=2,
        max_model_len=8192,
        enforce_eager=True,
        max_num_seqs=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype="auto",
        attention_backend="ROCM_AITER_UNIFIED_ATTN",
        enable_prefix_caching=False,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "attention_backend": "ROCM_AITER_UNIFIED_ATTN",
        },
    )
    prompts = _corpus()
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    print(f"completed generation over {len(prompts)} prompts", flush=True)
    del llm

    if not os.path.exists(_LOG):
        raise SystemExit("no hook records captured; patch did not reach workers")
    # aggregate max amax per ((kind, layer_idx), suffix) across all ranks/calls;
    # kind is "main" (full-attention text layers) or "mtp" (prediction head).
    layer_scales: dict[tuple[str, int], dict[str, float]] = {}
    seen = 0
    with open(_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec, _, amax_s = line.partition("\t")
            base, _, suffix = rec.rpartition(":")
            if ".self_attn.attn" not in base:
                print(f"WARN: unexpected record: {line!r}", flush=True)
                continue
            kind = "mtp" if ".mtp.layers." in base or base.startswith(
                "mtp.layers."
            ) else "main"
            try:
                idx = int(base.split("layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                print(f"WARN: cannot parse record: {line!r}", flush=True)
                continue
            amax = float(amax_s)
            seen += 1
            entry = layer_scales.setdefault((kind, idx), {})
            entry[suffix] = max(entry.get(suffix, 0.0), amax)
    print(f"aggregated {seen} observations", flush=True)

    # Build output names from the authoritative weight_map keys: every KV-cached
    # attention layer (main full-attention + the MTP head) has a
    # `*.self_attn.q_proj.weight` entry. The scale tensor for that layer is the
    # same base with the `_scale` suffix under `.self_attn.attn`. Deriving the
    # name from the index avoids guessing the `model.language_model` vs
    # `language_model.model` prefix (the hook's attn.layer_name differs from the
    # weight-map namespace).
    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))
    by_key: dict[tuple[str, int], str] = {}
    for key in idx.get("weight_map", {}):
        # q_proj weight key across formats: plain fp8/bf16 (".q_proj.weight"),
        # compressed-tensors/Marlin pack-quantized (".q_proj.weight_packed",
        # ".q_proj.weight_scale", ...), GPTQ/AWQ (".q_proj.qweight", ...).
        # The scale tensor name derives from the module prefix, which is
        # format-independent, so any suffix after ".self_attn.q_proj" maps to
        # the same base.
        marker = ".self_attn.q_proj"
        if marker not in key:
            continue
        base = key[: key.index(marker) + len(".self_attn")]
        kind = "mtp" if ".mtp.layers." in base or base.startswith(
            "mtp.layers."
        ) else "main"
        if ".layers." not in base:
            continue
        by_key[(kind, int(base.split(".layers.")[1].split(".")[0]))] = base

    entries, tensor_dict = [], {}
    for (kind, i) in sorted(by_key):
        scales = layer_scales.get((kind, i))
        if not scales:
            print(f"WARN: no capture for {kind} layer {i}, skipping", flush=True)
            continue
        base = by_key[(kind, i)]
        for suffix in ("q", "k", "v"):
            amax = scales.get(suffix)
            if amax is None:
                print(f"WARN: no {suffix} capture for {kind} layer {i}, skipping")
                continue
            scale = amax / E4M3FN_MAX
            tname = f"{base}.attn.{suffix}_scale"
            tensor_dict[tname] = torch.tensor(scale, dtype=torch.float32)
            entries.append((tname, scale))
            print(f"  {tname} = {scale:.6f}  (amax {amax:.4f})", flush=True)

    from safetensors.torch import save_file
    sidecar = os.path.join(out_dir, "model-kvscales.safetensors")
    save_file(tensor_dict, sidecar)
    print(f"wrote {sidecar}", flush=True)

    idx_path = os.path.join(out_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    for tname, _ in entries:
        idx["weight_map"][tname] = "model-kvscales.safetensors"
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"updated {idx_path} with {len(entries)} weight_map entries", flush=True)


if __name__ == "__main__":
    main()
