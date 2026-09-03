# Archived notes: dead ends, resolved/superseded issues, stale triage

> **Archived.** Material moved out of the README when it was reduced to
> current-state content (2026-08-21). Nothing here is actionable on the
> current stack; kept for reference. The live upstream-issue watchlist and
> known-to-ignore list are maintained in `AGENTS.md` ("Checking for Updates",
> last triage 2026-08-21).

## Dead ends

- **V2 model runner as the Qwen3.8-27B default** (tried 2026-09-03, rolled
  back same day). `VLLM_USE_V2_MODEL_RUNNER=1` on the pinned v0.28.1rc0: fully
  correct (bench coherence, 54K × 10, image+MTP, c2 condense smoke — all
  passed) and pp2048 +3.6–4% (consistent, zero overlap), but decode tg32/tg128
  flat and MTP acceptance unchanged (2.7–3.4 on both runners) — the #54498
  acceptance hypothesis (V1 M-RoPE draft-slot bug) did not materialize. Not a
  platform-default path upstream ("slower with MRV2" ROCm TODO). Revisit per
  the conditions in
  `benchmarks/2026-09-03_qwen3.8-27b_v1_vs_v2.md`.
- **DFlash2 speculative decoding as the Qwen3.8-27B default** (tried 2026-08-21,
  reverted 2026-08-22). DFlash2 (`incoai/Qwen3.8-27B-DFlash2`, block 8,
  `num_speculative_tokens 7`) was briefly the 3.8 default after depth-0 decode
  beat MTP3 (~88 vs ~62 tg32). A clean depth A/B on the same calibrated
  v0.28.0rc2 build showed the win is **short-context only**: DFlash2 decode
  decays hard with depth (tg32 60.9@d4K → 37.5@d32K → 24.4@d64K → 14.8@d128K)
  and collapses under concurrency at depth (2.2@d64K c2), while MTP3 holds
  ~50-60 out to d64K and ~37@d128K with far lower variance. Reverted to MTP3.
  Two additional DFlash boot caveats: the drafter must NOT pin
  `attention_backend` to `ROCM_AITER_UNIFIED_ATTN` (non-causal attention not
  supported → server crash-loops on boot; leave it unset so vLLM auto-selects a
  non-causal-capable backend), and DFlash2 uses text-only draft inputs on
  multimodal (images verified coherent, but draft ignores vision). Depth data:
  `benchmarks/2026-08-22_qwen3.8-27b_fp8kv_depth_mtp3_dflash.md`.
- **AITER MoE/FP8 backend on gfx1201**: vLLM aborts at startup. Enable once
  upstream AITER adds RDNA4 support.
- **`--enable-expert-parallel` on top of `-tp 2`**: regresses decode ~7-12% on
  the 35B-A3B (tg32 160-175 vs ~181-191, tg128 135-137 vs ~146) with flat
  prefill. EP's AllToAll doesn't pay off for a 3B-active MoE at tp=2. Skip at
  this scale; revisit only for much larger active-parameter MoEs.

## Resolved / superseded issues

- **MTP token loops on Qwen3-MoE** ([#47087](https://github.com/vllm-project/vllm/issues/47087)):
  deep agentic conversations on Qwen3-MoE degenerated into garbled token loops.
  **Resolved upstream by #51113 (in v0.27.1).** MTP remains disabled on 35B-A3B
  (`VLLM_SPEC_DECODE=` empty) pending a re-test of MTP on that profile, which
  would recover ~185 tg32 (MTP4) vs ~83 tg32 (no MTP).
- The earlier "garbage output" seen with BF16 KV was this MTP token loop, not
  the AITER BF16 LDS-fit patch.

## Upstream watchlist (snapshot, checked 2026-08-19)

Superseded by the live list in `AGENTS.md`.

- **`#52872`** GDN/hybrid prefill peak under-predicted;
  `--max-num-batched-tokens` also sizes the CUDA-graph pool (open).
- **`#47602`** MTP acceptance rate decays with context length (open, on
  Qwen3.6-27B).
- **`#51250`** prefix caching is a silent no-op on GDN hybrid (open).
- **`#45238`** hybrid prefix caching drops to 0% in align mode (open),
  **`#52520`** align-mode admission livelock near the KV-pool ceiling (open),
  **`#51562`** GDN metadata misclassifies a stateless first chunk as decode
  (open, no fix yet).

## Known to ignore (snapshot, checked 2026-08-19)

Superseded by the live list in `AGENTS.md`.

- **`#52475`** MTP repetition collapse with turboquant KV — NVIDIA sm120 KV
  quant; this stack is ROCm gfx1201 with standard fp8/bfloat16 KV.
- **`#52480`** `qwen3_5_mtp` fails at TP≥2 — Qwen3.5 drafter, not the Qwen3.6/3.8
  MTP heads used here.
- **`#52583`** prefix caching hangs on Qwen3.8-**VL** multimodal inputs — this
  stack serves text-only Qwen3.8-27B.
- **`#51752`** hybrid block-size alignment skipped on PP ranks — this stack is
  TP2, no pipeline parallelism.
- **`#51530`** DeepSeek-V4 sparse indexer on ROCm, **`#51957`** AITER FP8 BMM
  with DP attention, **`#40017`** NIXL P/D disaggregation, **`#51805`**/
  **`#51766`** KV-connector paths — all require connectors, DP, DSpark, or
  disaggregation this stack does not use.
- **`#51971`** Qwen3 MoE GPTQ `qzeros` on gfx1201 — GPTQ path; profiles use FP8
  / BF16 checkpoints.
- **`#51571`** async MTP align accepted-count race — async scheduling is
  auto-disabled for MTP (`config/vllm.py:1108`), unreachable here.
- **`#52793`** fp8 KV on hybrid models falls back to scale 1.0 — **verified
  non-issue on this stack** (2026-08-19): Qwen3.8-27B-FP8 runs fp8 KV at
  scale 1.0, but a d200K/d256K probe passed coherence at 258k total tokens
  with clean logs and monotonic decode decay. Re-check only if a model with
  larger K/V dynamic range or a different fp8-KV profile is added.
- **`#52312`** BF16 MLA with `ROCM_AITER_FA` on gfx950, **`#52833`**/**`#48568`**
  GLM-5.2 MTP on MI-series — different GPU/model combo than gfx1201 + Qwen.

## Historical performance table rows (superseded by the current README table)

Measured on older builds (vLLM 0.25/0.26/0.27.0, pre-patch); kept as upgrade
history. Per-run files (where they exist) are in [`benchmarks/`](benchmarks/).

| model | MTP (draft #) | pp2048 t/s | tg32 t/s | tg128 t/s |
|:------|:--------------|-----------:|---------:|----------:|
| Qwen3.6-27B (Andy & upstream baseline) | MTP3, fp8 KV | 2750 | 81.9 | — |
| Qwen3.6-27B-FP8 (v0.26) | MTP4, fp8 KV | ~2927 | ~75 | ~66 |
| Qwen3.6-27B-FP8 (v0.27) | MTP4, fp8 KV | ~2916 | ~87 | ~76 |
| Qwen3.6-35B-A3B-FP8 (v0.26) | MTP4, fp8 KV | ~10864 | ~182 | ~144 |
| Qwen3.6-35B-A3B-FP8 (v0.27) | MTP4, fp8 KV | ~11143 | ~189 | ~151 |
| Qwen3.8-27B-FP8, pre-patch (08-18) | MTP3, fp8 KV | ~2633–2661 | 59.7 | 68.8 |

### v0.26 → v0.27 upgrade delta

| model | metric | v0.26.2.dev0 | v0.27.0 | delta |
|:------|:-------|-------------:|-----------:|------:|
| 27B | pp2048 | ~2927 | ~2916 | flat |
| 27B | tg32 | ~75 | ~87 | +16% |
| 27B | tg128 | ~66 | ~76 | +15% |
| 35B-A3B | pp2048 | ~10864 | ~11143 | +2.6% |
| 35B-A3B | tg32 | ~182 | ~189 | +3.8% |
| 35B-A3B | tg128 | ~144 | ~151 | +4.9% |

### Pre-patch Qwen3.8-27B depth sweep (08-18, superseded by the patched 08-19 sweep in the README)

| depth | ctx_pp t/s | tg32 t/s | e2e TTFT (s) |
|------:|-----------:|---------:|-------------:|
| 4096 | 2827 | 55.2 | 2.2 |
| 8192 | 2768 | 55.5 | 3.7 |
| 16384 | 2703 | 59.6 | 6.8 |
| 32768 | 2589 | 57.3 | 13.4 |
| 65536 | 2373 | 51.8 | 28.5 |
| 128000 | 2038 | 46.5 | 63.8 |
