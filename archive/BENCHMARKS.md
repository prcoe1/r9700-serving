# Benchmarks (archived snapshot, 2026-08-12)

> **Archived.** Snapshot of `BENCHMARKS.md` as of the vLLM 0.27.0 build, kept
> for reference when the live file was reduced to current-state content. The
> current setup and per-run index live in [`../BENCHMARKS.md`](../BENCHMARKS.md);
> per-run tables referenced below now live in [`benchmarks/`](benchmarks/).
> Concurrency/depth findings that still apply to the current stack are
> summarized in the README ("Long-context concurrency").

All benchmarks use `llama-benchy` (0.4.0, via `uvx`) against
`http://localhost:8180/v1`. Per-run tables live in [`benchmarks/`](benchmarks/).

## Setup

`--max-num-batched-tokens 4096`, `--max-num-seqs 1`, `--gpu-memory-utilization
0.85`, `-tp 2`, `GPU_MAX_HW_QUEUES=1`. Single-request numbers are invariant to
`--max-num-seqs`; the server runs at `--max-num-seqs 1` because concurrency
loses to serial on this stack (see Concurrency). The model's bundled default
chat template is used (reasoning → `reasoning`, answer → `content`).

## Current (2026-08-12, vLLM 0.27.0 build, torch 2.13, triton 3.8.0, bf16 KV + tuned MoE + tuned dense, MTP off on 35B)

> **Note on versions:** the tables below were measured on the vLLM 0.27.0
> build. The current source-build pin is **0.27.1** (see README config table);
> the 0.27.0 → 0.27.1 bump is a packaging/pin update and these numbers are
> expected to carry over unchanged.

**BF16 KV** restored via AITER LDS-fit patch + **tuned fused MOE configs**
(`fused_moe_configs/E=256,N=256,...json`) for Triton 3.8.0. MTP disabled on
35B-A3B per vLLM #47087 workaround; 27B retains MTP4. Both models also carry
the tuned dense w8a8 block-FP8 GEMM configs (see README "Key tuning decisions").

| model | test | t/s |
|:------|-----:|----:|
| 35B-A3B BF16+MoETuned+MtPOff | pp2048 | ~8788 |
| 35B-A3B BF16+MoETuned+MtPOff | tg32 | ~87.8 |
| 35B-A3B BF16+MoETuned+MtPOff | tg128 | ~87.1 |
| 35B-A3B +DenseTuned (same)   | pp2048 | ~8510 |
| 35B-A3B +DenseTuned (same)   | tg32 | **91.0** |
| 35B-A3B +DenseTuned (same)   | tg128 | **91.3** |
| 27B BF16+MTP4                | pp2048 | ~2471 |
| 27B BF16+MTP4                | tg32 | ~80.6 |
| 27B BF16+MTP4                | tg128 | ~63.7 |
| 27B BF16+MTP4+DenseTuned     | pp2048 | ~2500 |
| 27B BF16+MTP4+DenseTuned     | tg32 | **90.8** |
| 27B BF16+MTP4+DenseTuned     | tg128 | ~69 |
| Qwen3.8-27B fp8-KV+MTP3+DenseTuned (default) | pp2048 | ~2633–2661 |
| Qwen3.8-27B fp8-KV+MTP3+DenseTuned (default) | tg32 | **59.7** |
| Qwen3.8-27B fp8-KV+MTP3+DenseTuned (default) | tg128 | **68.8** |

### v0.26 → v0.27 upgrade

| model          | metric  | v0.26.2.dev0 | v0.27.0 | delta |
|:---------------|:--------|-------------:|-----------:|------:|
| 27B            | pp2048  |      ~2927   |    ~2916   | flat  |
| 27B            | tg32    |        ~75   |      ~87   | **+16%** |
| 27B            | tg128   |        ~66   |      ~76   | **+15%** |
| 35B-A3B        | pp2048  |     ~10864   |   ~11143   | +2.6%  |
| 35B-A3B        | tg32    |       ~182   |     ~189   | +3.8%  |
| 35B-A3B        | tg128   |       ~144   |     ~151   | +4.9%  |

The dense 27B benefits most from v0.27's spec-decode improvements and
multi-layer MTP refactor. Both models show consistent gains with no regressions.

## Baseline comparisons

| description                      | pp2048 t/s | tg32 t/s | vLLM  |
|:---------------------------------|-----------:|---------:|------:|
| 27B andy upstream (MTP3, fp8 KV) |     2750   |   81.9   | v0.25 |
| 35B-A3B no MTP, stock config     |   ~10075   |    ~83   | v0.26 |
| 27B (MTP4, fp8 KV, all opts)    |    ~2916   |    ~87   | v0.27 |
| 35B (MTP4, fp8 KV, all opts)    |   ~11143   |   ~189   | v0.27 |

The 35B-A3B MoE model is 3.8× faster on prefill and 2.2× faster on decode than
the dense 27B. MTP4 + NCCL tuning + v0.27 deliver 2.3×
decode throughput over the no-MTP baseline.

## MTP impact (35B-A3B)

Enabling MTP roughly doubles decode speed. Draft-token count was tuned on the
35B-A3B. The table below is MTP3-era; MTP is currently **disabled on 35B-A3B**
per vLLM #47087 (see README "Archived issues") and 27B runs MTP4.

| MTP | pp2048 (t/s) | tg32 (t/s) | acceptance |
|:----|-------------:|-----------:|-----------:|
| off |        10075 |       82.9 |         -  |
| 2   |   8000 ± 527 | 145.54 ± 6.51 | 59.8% |
| 3   | 9354 ± 171.64 | 143.80 ± 6.85 | **72.3%** |

## NCCL channels

Two R9700s on separate PCIe 5.0 x8 root ports, P2P disabled
(`NCCL_P2P_DISABLE=1`). `all_reduce_perf` (rccl-tests, out-of-place busbw GB/s):

| channels |     1M |    4M |     8M |    32M |    64M |
|:---------|-------:|------:|-------:|-------:|-------:|
| 1        |   8.04 |  9.51 |  11.09 |  11.81 |  11.94 |
| 2        |   8.88 | 11.19 |  12.15 |  12.54 |  12.61 |
| **4**    | **9.21** | 11.50 | 11.86 | **12.80** | **12.91** |
| 8        |   8.88 | 11.50 |  11.79 |  12.72 |  12.87 |
| 16       |   8.20 | 11.67 |  12.10 |  12.76 |  12.89 |
| 32       |   7.52 | 10.75 |  11.70 |  12.51 |  12.74 |
| 112      |   9.13 | 11.07 |  12.16 |  12.52 |  12.60 |

4 channels is fastest or near-fastest at every size. Serving A/B confirms
+12-19% tg128 decode on 4-ch vs 112-ch. Recommendation: pin both
`NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` to 4.

## Concurrency

> **Note:** The depth sweep and concurrency benchmarks referenced below were
> run with **bf16 KV** and **tuned fused_moe configs** — the same stack as the
> current running config. The concurrency *behavior* (c1 vs c2, depth scaling)
> is independent of the later tuned-dense GEMM configs and still valid.

35B-A3B. The table below is the original MTP3-era data; current MTP4 findings
follow. `total` = aggregate across all concurrent requests, `req` = per
request.

| test                | c1 (t/s) | c4 total (t/s) | c4 req (t/s) | scaling |
|:--------------------|---------:|---------------:|-------------:|--------:|
| pp2048 / tg128      |  157.73 ± 9.18 |  279.00 ± 7.90 |   91.20 ± 13.65 | 1.77x |
| pp2048 / tg512      |  140.69 ± 6.98 |  326.72 ± 8.35 |   89.91 ± 5.00 | **2.32x** |
| pp8192 / tg256      |  136.03 ± 1.74 |  187.88 ± 5.21 |   67.76 ± 15.30 | 1.38x |

Single-request decode is flat ~140-158 t/s regardless of generation length.
Longer generations scale best at concurrency 4 (2.32× for tg512). Longer prompts
cut scaling (prefill competes with decode for the `max_num_batched_tokens`
budget).

The same limit hits long *contexts* even with short generations: a c4 depth
sweep (tg32) keeps full-context prefill scaling well (~11-12k t/s aggregate at
d1024-d8192) but decode aggregate collapses from 347 t/s @ d1024 to 3.6 t/s @
d64000 — the 64k-token contexts can no longer share a batch within the 4096
token budget, and e2e TTFT reaches ~23.9 s. See
[`08_10_qwen3.6-35b-a3b_mtp4_depth_c4.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_depth_c4.md).

A c2 depth sweep finds the sweet spot for long-context serving: aggregate decode
matches c4 (geomean 86.1 vs 86.0 t/s) but e2e TTFT @ d64000 is ~15× better
(1.6 s vs 23.5 s), since the 64k prefill no longer serializes against a second
deep prefill in the batch budget. Per-request throughput is still 23-68% below
c1 at depth. A/Bs (fp8 KV, MTP2, 8192 batch) all lost to the baseline — the
deep-context decode cost is inherent to attending over a huge cached KV, not
fixable by those knobs. See
[`08_10_qwen3.6-35b-a3b_mtp4_depth_c2.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_depth_c2.md).

### c1 vs c2 head-to-head (2026-08-10)

A same-boot A/B (`--max-num-seqs 1` vs 2, identical depth sweep, see
[`08_10_qwen3.6-35b-a3b_mtp4_c1_vs_c2_depth.md`](benchmarks/08_10_qwen3.6-35b-a3b_mtp4_c1_vs_c2_depth.md))
settles it: **serial (c1) is strictly more efficient**. The earlier c2 sweep
was not a bad-process-state artifact — it reproduces closely (c2 total geomean
95 vs 86 before). Concurrency never reaches single-request decode:

| depth | c1 | c2 total | c2/req | c2 total / c1 |
|:------|---:|---------:|-------:|:--------------|
| d1024 | 180.0 |    127.2 |  113.1 | 71% |
| d16384| 156.1 |     97.4 |  104.1 | 62% |
| d32000| 175.7 |     76.7 |   97.4 | 44% |
| d64000| 171.2 |     46.9 |   77.0 | 27% |
| geomean | 169.5 |   94.8 |  100.4 | - |

Two concurrent requests (c2 total) move *fewer* tokens/s than one request alone
at every depth, so two deep requests finish faster back-to-back, and c2's
latency is worse (incremental TTFT @ d64000 1070 vs 1562 ms; full-context load
9292 vs 14178 ms). The server runs at `--max-num-seqs 1`. Use c2 only when
multiple users must progress simultaneously at ~45-55% lower per-request
decode.

## Depth sweep (35B-A3B, MTP4, fp8 KV, stock MoE)

Single-request speeds at increasing prompt depth (d = prefix tokens before the
2048-token prompt). `pp2048` = **incremental** prefill of only the
2048 fresh tokens (depth prefix cached via `--enable-prefix-caching`); each new
token attends over the full cached KV, so this falls with depth. `ctx_pp` =
full-context prefill (`depth + 2048`), comparable to pre-PC "pp2048 @ dXXX".

| depth | ctx_pp (t/s) | ctx_tg (t/s) | pp2048 (t/s) | tg32 (t/s) | ctx_e2e (ms) |
|------:|-------------:|-------------:|-------------:|-----------:|-------------:|
| 0 | — | — | 9381 | 185 | — |
| 1024 | 7462 | 170 | 6847 | 195 | 3072 |
| 4096 | 9695 | 185 | 3474 | 173 | 6144 |
| 8192 | 10189 | 166 | 3433 | 173 | 10240 |
| 16384 | 9652 | 172 | 3366 | 162 | 18496 |
| 32000 | 9038 | 158 | 2710 | 157 | 34816 |
| 64000 | 7876 | 132 | 2314 | 149 | 66816 |
| 128000 | 6315 | 94 | 1379 | 114 | 130176 |

Incremental prefill drops with depth (attention span over the cached prefix
grows): ~6847 t/s at d1024 → ~1379 t/s at d128K. Decode is mostly flat
114-195 t/s, with the expected shallow-context bump from full-budget cudagraphs.
Full-e2e TTFT scales linearly with total context loaded (cache-friendly 6.3K
t/s ctx_pp up to ~6K depth, degrading at 128K to 6.3K with 20.3s TTFT).

## Depth sweep (35B-A3B, bf16 KV, tuned MOE, MTP off, 2026-08-11)

Same stack as current but with **BF16 KV** + **tuned fused MOE configs**, MTP
disabled per vLLM #47087. `pp` = incremental prefill of 2048 fresh tokens
(depth prefix cached).

| depth | tg32 (t/s) | tg128 (t/s) |
|------:|-----------:|------------:|
| 0     | 86.8 | — |
| 4096  | 86.8 | 86.7 |
| 8192  | 86.5 | 86.1 |
| 16384 | 85.4 | 84.7 |
| 32768 | 81.7 | 81.6 |
| 65536 | 78.2 | 77.3 |
| 128000| 71.4 | 70.7 |

**Key finding:** tg32/tg128 are nearly identical (~88→86 t/s at d4K, ~78 t/s
at d64K, ~71 t/s at d128K). The tg128 bump over tg32 at 0-depth (~189 t/s
MTP4 baseline) erodes once MTP is removed — without draft tokens, decode speed
is capped by the MoE kernel throughput regardless of output length. KV cache
loading dominates prefill at depth, and decode falls with depth as the attention
over the cached prefix grows (71 t/s at d128K vs 87 at d0 = −18%).

## Depth sweep (35B-A3B, MTP off, tuned dense vs stock, 2026-08-12)

Deep-context decode is dominated by attention over the cached KV, so the GEMM
tuning benefit narrows with depth. Same-boot A/B (bf16 KV, tuned MoE, thinking
off); the `stock` column is the table above (no tuned dense configs).

| depth | stock tg32 | +tuned dense tg32 | uplift |
|------:|-----------:|------------------:|:------|
| 0     | 86.8 | **90.3** | +4% |
| 4096  | 86.8 | **91.4** | +5% |
| 65536 | 78.2 | **79.1** | +1% |
| 128000| 71.4 | **72.3** | +1% |

## Depth sweep (Qwen3.8-27B-FP8, fp8 KV, MTP3, 2026-08-18)

Current default stack: **fp8 KV** (`VLLM_KV_CACHE_DTYPE=fp8`), MTP3 spec-decode,
`--max-model-len 262144`. Full-context prefill sweep (no prefix caching: each
run prefills `depth + 2048`).

| depth | ctx_pp (t/s) | ctx_tg32 (t/s) | e2e TTFT (s) |
|------:|-------------:|---------------:|-------------:|
| 4096  |     2698 |         48.7 |           2.3 |
| 8192  |     2754 |         51.3 |           3.7 |
| 16384 |     2699 |         51.3 |           6.8 |
| 32768 |     2588 |         46.5 |          13.5 |
| 65536 |     2364 |         59.8 |          28.6 |
| 128000|     2032 |         40.3 |          64.0 |

Decode is flat ~47–51 t/s from d4K–d16K (the d64K point is 2-run noise),
falls to 40.3 t/s at d128K, and full-context prefill degrades 2698 → 2032 t/s
(−25%) with e2e TTFT scaling linearly to 64 s at d128K. fp8 KV halves the KV
bytes moved per attention step — which deep-context decode is bound by — so
decode at depth holds up against the 57.6 t/s d0 bf16-KV baseline
([`08_14_qwen3.8-27b_bf16_mtp4_bench.md`](benchmarks/08_14_qwen3.8-27b_bf16_mtp4_bench.md)).
See [`benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_18_qwen3.8-27b_fp8kv_mtp3_depth.md)
for full tables and observations.

## Superseded Qwen3.8-27B rows (Sept–Oct 2026, moved from README 2026-10-04)

| model | MTP | KV | pp2048 t/s | tg32 t/s | tg128 t/s |
|:------|:----|:---|-----------:|---------:|----------:|
| 2026-09-22 (v0.30.0, bf16) | MTP3 | bf16 | ~3124 | ~67 | ~68 |
| 2026-09-27 (v0.30.0+55390/58368, fp8) | MTP1 | fp8 | ~3440 | ~50 | ~52 |
| 2026-09-28 (v0.30.1rc0, spec off) | off | fp8 | ~3045–3353 | ~34.6 | ~34.3 |
| 2026-10-01 AM (v0.31.0rc3, bf16, spec off) | off | bf16 | ~3493–3519 | ~34.3 | ~34.5 |
| 2026-10-01 PM (v0.31.0rc3, fp8, MTP3 re-enabled) | MTP3 | fp8 | ~3418 | ~61.6 | ~70.8 |
| 2026-09-18 (v0.29.0, fp8) | MTP3 | fp8 | ~3275 | ~68 | ~71 |
| 2026-08-28 (pre-v0.29.0, fp8) | MTP3 | fp8 | ~3160 | ~62 | ~61 |
| 2026-08-25 (pre-v0.29.0, bf16) | MTP3 | bf16 | ~3060 | ~67 | ~68 |

Notes: 09-22 delta vs 09-18 is the bf16 KV switch (operator decision), not a
regression. 09-27 K was cut 3→1 for #47602 acceptance decay at depth. 10-01
MTP3 re-enable: ~1.8–2.1x decode vs off at ~−4% prefill, acceptance ~64%.
Per-run records in `benchmarks/2026-09-2*.md`, `benchmarks/2026-10-01_*.md`.
