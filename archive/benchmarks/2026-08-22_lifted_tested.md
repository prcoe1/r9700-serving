# Session 2026-08-22 — lifted-and-tested items (qwen3.8-27b only)

Source repos reviewed: `stilldeadcode/vllm-radiance` and `tcclaviger/vllm`
(Docker Hub), plus `patcarter883/rdna4-vllm` and the tcclaviger docs. No
library downgrades; stack stays vLLM v0.28.0rc2, torch 2.13.0+rocm7.14.0,
AITER v0.1.20. Everything below is on **Qwen3.8-27B-FP8** only.

## 1. fp8 KV scale calibration (tcclaviger's "scale 1.0 is wrong" warning)

- Both `Qwen3.8-27B-FP8` and `Qwen3.6-27B-FP8` ship **no** `k_scale` /
  `v_scale` / `q_scale`. On vLLM ≥0.28 (`--calculate-kv-scales` removed,
  #37201/#49389) fp8 KV serves at **scale 1.0**; boot log confirmed:
  `Using KV cache scaling factor 1.0 for fp8_e4m3`.
- Built `tools/calibrate_kv_scales.py`: runs the checkpoint over a small
  diverse text corpus with vLLM offline (TP=2), hooks
  `Qwen3NextAttention._project_qkv_gate` (fork-inherited) to record per-layer
  amax of post-norm/post-RoPE q, post-norm k, raw v; writes `amax/448`
  (e4m3fn convention; vLLM doubles internally for e4m3fnuz on RDNA) as a scalar
  `model-kvscales.safetensors` sidecar + `model.safetensors.index.json`
  `weight_map` entries (16 full-attn layers at 3,7,…,63; 48 scalars).
- Calibration found deep-layer **V amax up to 132** (layer 63) — vs the ~1-24
  range scale 1.0 effectively assumes, confirming the miscalibration.
- Sidecar loads correctly (scale-1.0 warning gone), correct e4m3fnuz convention
  (no clipping: effective scale = amax/224 < fnuz max 240), no bench regression.
- **Determinism evidence the effect is real:** two scale-1.0 instances are
  100% token-identical; same-instance reruns are 100% identical; but
  calibrated vs scale-1.0 deterministic (temp 0) outputs diverge **~20-27%**
  word-match. So scale 1.0 genuinely changes KV numerics and generation.
- **Now the default** for both `qwen3.8-27b` and `qwen3.6-27b`: each profile
  declares `VLLM_MODEL_ID` (HF source) + `VLLM_MODEL`
  (`~/models-local/<model>-kvscales`), and `just up` runs the new `ensure-kvscales`
  recipe which creates the local copy (`tools/setup_kvscales.py`, symlinks into
  the HF cache) and calibrates it (throwaway container) when the sidecar is
  missing. `just clear-kvscales` forces a recalibration. `calibrate_kv_scales.py`
  now derives the full-attention layer list from `config.json` (`layer_types`)
  so it works for any Qwen3.5/3.6/3.8 hybrid; both models measured V amax ~130+
  at the deepest layer.

## 2. DFlash boot bug found while testing (fixed in `env/qwen3.8-27b.env`)

The committed config forced `attention_backend: ROCM_AITER_UNIFIED_ATTN` on the
DFlash draft model. DFlash needs **non-causal** attention (block diffusion) and
AITER unified cannot do non-causal, so the server crash-looped on boot
(`Selected backend ... non-causal attention not supported`). Removing the
drafter's `attention_backend` fixes boot and reproduces the recorded DFlash2
bench (~92 t/s tg32). The DFlash model-card example also omits it.

## 3. PCIe P2P must stay off (radiance / tcclaviger P2P all-reduce)

| config | tg32 (t/s) | tg128 (t/s) | pp2048 (t/s) |
|:-------|-----------:|------------:|-------------:|
| baseline (legacy=1, P2P off) | ~92 | ~82 | ~2650 |
| **P2P on** (legacy=0, `NCCL_P2P_DISABLE=0`) | **~9** | **~8** | ~1650-2030 |
| legacy=0, P2P off | ~76 | ~84 | ~2701 |

Enabling P2P collapses decode ~10× despite RCCL reporting `isAllCudaP2p 1` and
8 P2P channels — the two R9700s are on separate PCIe root ports (README note
confirmed). `HSA_ENABLE_IPC_MODE_LEGACY` is irrelevant once P2P is off. This
de-prioritizes the tier-3 P2P all-reduce HIP kernels on this host.

## 4. Images + DFlash spec-decode — coherent

`benchmarks/image_spec_probe.py`: image prompt + follow-up both coherent
(correctly identifies a red square and blue circle). DFlash logs
`...does not support external multimodal embeddings... using text-only draft
inputs`, so the radiance "MTP drafter multimodal mask alignment" concern does
not apply to DFlash.

## 5. tcclaviger env vars — neutral

`HSA_ENABLE_INTERRUPT=1`, `HSA_ENABLE_MWAITX=1`, `OMP_NUM_THREADS=8`,
`NCCL_PROTO=Simple` as a batch: pp2048 ~2729 vs baseline ~2679 (within noise),
tg32 ~71 vs ~92 (within the wide baseline stddev). No clear win → not adopted.

## Reproducers added

- `tools/calibrate_kv_scales.py` — build the KV-scale sidecar.
- `benchmarks/kv_compare.py` — deterministic output A/B (calibrated vs scale 1).
- `benchmarks/image_spec_probe.py` — image + spec-decode coherence.
