# Benchmarks

Current-state benchmark setup and index for this stack. All benchmarks use
`llama-benchy` (0.4.0, via `uvx`) against `http://localhost:8180/v1`.
Superseded per-run files (old builds, pre-patch baselines, rejected paths)
live in [`archive/benchmarks/`](archive/benchmarks/); older history in
[`archive/BENCHMARKS.md`](archive/BENCHMARKS.md).

## Setup (current)

vLLM 0.30.0 + local patches (see README "Source-build patches"), torch 2.13,
ROCm 10.0, AITER v0.1.20.post1 unified attention, froggeric chat template
v22.5, MRV2 runner, `--no-async-scheduling` on MTP profiles. `-tp 2`,
`--gpu-memory-utilization 0.95`, `GPU_MAX_HW_QUEUES=1`. KV cache is **bf16
on qwen3.8-27b** (the live default profile, per 2026-09-21 operator decision;
block 832 at 128k max-len) and **bf16 on the 3.6 profiles** (3.6-27b carries
a calibrated fp8 sidecar on disk for opt-in via `VLLM_KV_CACHE_DTYPE=fp8`);
`--max-num-seqs 2` everywhere (the #35288 cap); `--max-num-batched-tokens 1024` on qwen3.8-27b (concurrent-ITL A/B,
2026-09-03), `4096` on 35B-A3B (block-size forced).

Single-request numbers are invariant to `--max-num-seqs`; long-context
concurrency degrades sharply (see the c1-vs-c2 head-to-head in
[`archive/BENCHMARKS.md`](archive/BENCHMARKS.md)).

## Current per-run files

| file | contents |
|:-----|:---------|
| [`benchmarks/2026-09-22_qwen3.8-27b_v0.30.0_bump.md`](benchmarks/2026-09-22_qwen3.8-27b_v0.30.0_bump.md) | **v0.30.0 bump validation** (live): 3 patch rebases + new #56190 import-abort fix, bench pp2048 ~3124 / tg32 ~67 / tg128 ~68 (bf16), prefix-cache still 0%, nan-probe inconclusive/no-hit, thinkoff/tool-truncation/multi-image PASS, conc-ITL choke 17.2x |
| [`benchmarks/2026-09-19_qwen3.8-27b_stability_depth_conc.md`](benchmarks/2026-09-19_qwen3.8-27b_stability_depth_conc.md) | **release validation** (v0.29.0 + #40709 patch, live): stability 400/400 + 10/10 + 10/10, depth d0–d256K matches 08-27 shape, conc-ITL p50 880 ms / p99 1.38 s, coherence PASS everywhere — no regressions |
| [`benchmarks/2026-09-19_qwen3.8-27b_40707_multi_image_probe.md`](benchmarks/2026-09-19_qwen3.8-27b_40707_multi_image_probe.md) | #40707 4-large-image probe: PASS on the patched build (1-image limit lifted) |
| [`benchmarks/2026-09-03_qwen3.8-27b_concurrent_itl.md`](benchmarks/2026-09-03_qwen3.8-27b_concurrent_itl.md) | `--max-num-batched-tokens` 8192/4096/2048 A/B: big-prompt prefill stalled the co-decoder 150–200x (ITL p99 up to 9.8 s) at 8192; 2048 → ~1 s ITL, flat big-prompt TTFT, −3.4% pp2048 → **2048 adopted** on qwen3.8-27b |
| [`benchmarks/2026-09-10_qwen3.8-27b_55766_nan_probe.md`](benchmarks/2026-09-10_qwen3.8-27b_55766_nan_probe.md) | #55766 NaN-checkpoint probe: CLEAN on v0.29.0 + MTP3 (masked by the 2-block hit back-off, not disproven) |
| [`benchmarks/2026-09-10_qwen3.8-27b_awq_trial.md`](benchmarks/2026-09-10_qwen3.8-27b_awq_trial.md) | AWQ-INT4 trial profile record (decode +30–50%, prefill −26%) |
| [`benchmarks/2026-08-27_qwen3.8-27b_depth_no_async.md`](benchmarks/2026-08-27_qwen3.8-27b_depth_no_async.md) | Qwen3.8-27B full depth sweep 0–256K (live config family: fp8 KV, MTP3, no-async) |
| [`benchmarks/2026-08-27_qwen3.8-27b_no_async_scheduling.md`](benchmarks/2026-08-27_qwen3.8-27b_no_async_scheduling.md) | `--no-async-scheduling` (#51571 mitigation): decode parity, c2 coherence smoke clean |
| [`benchmarks/2026-08-25_gfx1201_ua_tuning.md`](benchmarks/2026-08-25_gfx1201_ua_tuning.md) | gfx1201 unified-attention tuning backing the live AITER patches |
| [`benchmarks/2026-08-24_qwen3.6-27b_bf16_mtp4_bench.md`](benchmarks/2026-08-24_qwen3.6-27b_bf16_mtp4_bench.md) | Qwen3.6-27B d0 (bf16 KV, MTP4) — last measured, pre-v0.29.0 build |
| [`benchmarks/2026-08-24_qwen3.6-35b-a3b_bf16_mtp4_bench.md`](benchmarks/2026-08-24_qwen3.6-35b-a3b_bf16_mtp4_bench.md) | 35B-A3B d0 + MTP4 re-test (clean; ~2x decode) — last measured, pre-v0.29.0 build |
| [`benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md`](benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md) | bf16-KV depth sweep (comparison baseline for the fp8 default) |
| [`benchmarks/2026-08-22_qwen3.8-27b_depth_bf16_fp8.md`](benchmarks/2026-08-22_qwen3.8-27b_depth_bf16_fp8.md) | depth A/B: bf16 KV vs calibrated fp8 KV |
| [`benchmarks/2026-08-22_qwen3.8-27b_fp8kv_depth_mtp3_dflash.md`](benchmarks/2026-08-22_qwen3.8-27b_fp8kv_depth_mtp3_dflash.md) | MTP3 vs DFlash2 depth + concurrency-at-depth A/B (DFlash rejection) |
| [`benchmarks/2026-08-22_kv_calibration_quality_ab.md`](benchmarks/2026-08-22_kv_calibration_quality_ab.md) | calibrated vs scale-1.0 fp8 KV: PPL + long-context recall A/B |
| [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md) | fp8-KV depth sweep d4K–d128K (historical baseline) |
| [`benchmarks/STABILITY_TESTS.md`](benchmarks/STABILITY_TESTS.md) | smoke/stress/long-context coherence test scripts + baselines |

## Live probes (scripts)

| script | checks |
|:-------|:-------|
| [`benchmarks/prefix_cache_probe.py`](benchmarks/prefix_cache_probe.py) | prefix-cache hit-rate (AGENTS.md update workflow) |
| [`benchmarks/conc_itl_probe.py`](benchmarks/conc_itl_probe.py) | concurrent-ITL under prefill (re-run on chunk/graph-pool changes) |
| [`benchmarks/nan_checkpoint_probe.py`](benchmarks/nan_checkpoint_probe.py) | #55766 NaN-checkpoint geometry |
| [`benchmarks/thinking_probe.py`](benchmarks/thinking_probe.py) | thinking on/off rendering behavior |
| [`benchmarks/tool_truncation_probe.py`](benchmarks/tool_truncation_probe.py) | stream/non-stream parity on truncated tool calls (#47137 patch) |
| [`benchmarks/thinkoff_probe.py`](benchmarks/thinkoff_probe.py) | parser/template agreement on thinking-off (thinkoff patch) |

## NCCL channels (current hardware)

Two R9700s on separate PCIe 5.0 x8 root ports, P2P disabled
(`NCCL_P2P_DISABLE=1`). `all_reduce_perf` (rccl-tests, out-of-place busbw GB/s):

| channels | 1M | 4M | 8M | 32M | 64M |
|:---------|-------:|------:|-------:|-------:|-------:|
| 1 | 8.04 | 9.51 | 11.09 | 11.81 | 11.94 |
| 2 | 8.88 | 11.19 | 12.15 | 12.54 | 12.61 |
| **4** | **9.21** | 11.50 | 11.86 | **12.80** | **12.91** |
| 8 | 8.88 | 11.50 | 11.79 | 12.72 | 12.87 |
| 16 | 8.20 | 11.67 | 12.10 | 12.76 | 12.89 |
| 32 | 7.52 | 10.75 | 11.70 | 12.51 | 12.74 |
| 112 | 9.13 | 11.07 | 12.16 | 12.52 | 12.60 |

4 channels is fastest or near-fastest at every size; serving A/B confirms
+12-19% tg128 decode on 4-ch vs 112-ch. This is why
`NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` are pinned to 4.
