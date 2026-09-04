# vLLM on Radeon AI PRO R9700

Build and run vLLM from source for AMD Radeon AI PRO R9700 GPUs. The default
configuration targets two R9700s (`gfx1201`) and serves a model through vLLM's
OpenAI-compatible API.

## Requirements

- Docker with the Compose plugin (`docker compose`), or Podman (`podman
  compose`); `just` recipes default to Docker (Podman caveats below)
- [`just`](https://just.systems/)
- [`git`](https://git-scm.com/) (to fetch the source)
- SELinux hosts need no special relabeling: bind mounts mount unlabeled because
  the container runs with `label=disable`
- One or more R9700 GPUs; the included configuration assumes two

**Podman caveats** (verified on podman 5.7 / buildah 1.42):
- Runtime ops (`up`/`down`/`logs`/`run`/`exec`) need the podman API service
  running — `systemctl --user enable --now podman.socket` (or
  `podman system service &`). Without it, `podman compose` fails to connect
  to `/run/user/$UID/podman/podman.sock`. `just check` (config only) works
  without it.
- Builds work, but buildah silently ignores `RUN --mount=type=cache` (the
  target dir is pre-created but not shared across steps), so `just rebuild`
  loses pip/ccache cache reuse and runs slower than the Docker estimate.

## Quick start

Get the source (skip if you already have the repo checked out; all commands
below run from inside it):

```sh
git clone https://github.com/prcoe1/r9700-serving.git
cd r9700-serving
```

```sh
cp .env.example .env  # Build version pins + default model profile (untracked)
just build       # Build localhost/vllm-fullbuild:latest
just check       # Validate the compose config for the selected profile
just up          # Start vLLM in the background (default: Qwen3.8-27B-FP8)
just --set model qwen3.6-27b up     # Switch to Qwen3.6-27B-FP8 (dense)
just --set model qwen3.6-35b-a3b up  # Switch to MoE 35B-A3B model
just logs        # Follow service logs
just down        # Stop and remove containers
```

To use Podman: `just --set runtime podman build` or `RUNTIME=podman just up`
(see the Podman caveats in Requirements). Run `just --list` to see all recipes
including `rebuild` (force-rebuild) and `clear-vllm-caches` (wipe host-side
Triton/Inductor/AITER caches; preserves the HuggingFace model cache).

Always go through `just`: `compose.yaml` interpolates the model arguments from
`env/<profile>.env`, which the recipes pass to compose via `--env-file`. A bare
`docker compose up` fails with a required-variable error rather than starting a
server with no model.

The vLLM OpenAI-compatible API is available at `http://localhost:8180/v1`.
Other containers on the same compose network can reach it via the `llm-backend`
network alias instead of the host port.

## Configuration

Build versions are pinned in `.env` (untracked; copy `.env.example` to create it).
Host-specific settings also live there: `USER_UID`/`USER_GID` (the user the
container runs as, keeping host cache dirs user-owned) and `RENDER_GID` (the
host render group gid for `/dev/dri` access — check with `getent group render`).

| component    | version |
|:-------------|:--------|
| ROCm         | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch      | 2.13.0+rocm7.14.0 |
| vLLM         | 0.28.1rc0 |
| AITER        | v0.1.20 |
| Flash Attention | @ 1cc7ff67 |

ROCm 7.14 is on AMDs "TheRock" technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/`gfx1201` support. AITER `v0.1.20` is the
latest tagged release; vLLM is 0.28.1rc0 (421 commits past v0.28.0, including
the `--prefix-cache-retention-interval` flag from #52216/#53504) since
`gfx1201` requires source builds.

The default (active) model is `Qwen/Qwen3.8-27B-FP8` (`qwen3.8-27b`, the
newest dense 27B hybrid linear/full-attention architecture, MTP trained,
vision). Alternatives: `Qwen/Qwen3.6-27B-FP8` (`qwen3.6-27b`, dense) and
`Qwen/Qwen3.6-35B-A3B-FP8` (`qwen3.6-35b-a3b`, 35B total / 3B active MoE).
Model selection is controlled by `MODEL_PROFILE` in `.env` — override inline
with `MODEL_PROFILE=qwen3.6-27b just up`.

Runtime environment is split across files:
- `env/2xr9700.vllm.common` — two-GPU ROCm config (arch, NCCL, HSA, compile caches)
- `env/aiter-unified-attention.env` — enables AITER unified attention only
- `env/qwen3.6.env.common` — shared qwen3.6/3.8 config (KV cache dtype, MTP spec-decode, tool choice)
- `env/qwen3.6-35b-a3b.env` — MoE model config (path, tokenizer,
  `--max-num-batched-tokens 4096` cap)
- `env/qwen3.6-27b.env` — dense 27B model config
- `env/qwen3.8-27b.env` — Qwen3.8-27B-FP8 dense model config (same
  architecture as 3.6-27B, so it shares the 3.6 common settings and tuned
  per-shape fp8 GEMM configs)

### Chat template

All profiles mount and use [froggeric's Qwen-Fixed-Chat-Templates]
(`chat-templates/qwen.jinja`, pinned to **v22.5** — `qwen3.8-froggeric-v22.5`,
fetched from the repo's `main`). It is applied to every model via `--chat-template` in `compose.yaml`,
overriding each model's bundled template. It fixes rendering bugs, KV-cache
invalidation, and token waste in the official Qwen templates, and adds
tool-error retry warnings plus `tool_call_format` / `reasoning_effort` kwargs.
Since **v22.4** (retained in v22.5), history re-rendering is byte-identical to generated tokens on
thinking-off turns, which keeps `--enable-prefix-caching` hits intact across
multi-turn conversations (this stack runs thinking-off). Thinking is
partitioned into the `reasoning` field and the answer into `content`;
`--reasoning-parser qwen3` is required for the split. v22.5 additionally
conditions the tool-call instruction's `<think>` block on `enable_thinking`
(fixes spurious `<think>` when thinking is off) and avoids truncating JSON
`tool_response` payloads.

Refresh the overlay from upstream when a newer version ships (compare the
`template_version` line of `chat-templates/qwen.jinja` against the repo's
`main`):

```sh
curl -L -o chat-templates/qwen.jinja \
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
```

Template bumps need no image rebuild: after refreshing, bump the pin note
above and `just down && just up` (the in-memory prefix cache is cleared on
restart anyway).

[froggeric's Qwen-Fixed-Chat-Templates]: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

### Non-standard vLLM flags

- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`** (`VLLM_TOOL_CHOICE`, all profiles): OpenAI
  tool-calling with Qwen's `qwen3_coder` parser; `--reasoning-parser qwen3` is
  required for the template's `reasoning`/`content` split.
- **`--limit-mm-per-prompt '{"image": 99, "audio": 0, "video": 0}'`**: up to
  99 images per prompt, audio/video disabled. Previously capped at 1 to block
  the 2+-large-images engine deadlock on these GDN hybrids (upstream #40707,
  fix #40709 not merged — see AGENTS.md watchlist); with 99 that deadlock
  can hang the engine (request hangs forever, engine never recovers) if a
  prompt contains 2+ large images — use at own risk.
- **`--override-generation-config`**: server-side sampling defaults
  (`temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0, no penalties).
- **`--enable-prefix-caching`**: reuse KV for shared prompt prefixes (known
  limitations on this hybrid — AGENTS.md watchlist).
- **`--max-model-len`** (default `131072`; Qwen3.8-27B overrides to `262144`),
  **`-tp 2`**, **`--gpu-memory-utilization 0.92`** (`VLLM_GPU_MEM_UTIL`, was
  0.95; the lower default leaves VRAM headroom for GPU co-tenants — raise it
  if the GPUs are single-tenant), **`--max-num-seqs 2`** (the universal #35288
  cap, set explicitly per profile to keep it visible).
- **`--kv-cache-dtype bfloat16`** (`VLLM_KV_CACHE_DTYPE`, default `bfloat16`
  on all profiles; the AITER BF16 LDS-fit patch
  `patches/aiter/unified-attention-bf16-kv.patch` is required). Opting into fp8
  KV (`VLLM_KV_CACHE_DTYPE=fp8`) halves KV memory and automatically serves the
  calibrated local copy built by `just up` (`ensure-kvscales`): the stock
  checkpoints ship no KV scales, and uncalibrated scale-1.0 is miscalibrated.
  Calibration is a correctness fix, not a measured quality win — see the
  `#52793` note in AGENTS.md and
  [`benchmarks/2026-08-22_kv_calibration_quality_ab.md`](benchmarks/2026-08-22_kv_calibration_quality_ab.md).
- **`--attention-backend ROCM_AITER_UNIFIED_ATTN`** + `--speculative-config`
  (MTP4 on Qwen3.6-27B, **MTP3** on Qwen3.8-27B, **MTP4 on 35B-A3B**). MTP3 is
  the Qwen3.8-27B default: DFlash2's decode win is short-context only (it
  decays hard with depth) and was rejected after a 2026-08-22 depth A/B — see
  [`archive/DEADENDS.md`](archive/DEADENDS.md).

### Runtime overlays (bind-mounted source fixes)

Version-locked patches applied at runtime by read-only bind-mounts in
`compose.yaml` (no image rebuild needed). Refresh the overlay files when bumping
the pinned dependency.

- **Tolerate empty `tools` arrays** (`patches/vllm/protocol.py`, pinned to
  `VLLM_REF` v0.28.1rc0): some clients send `{"tools": [], "tool_choice":
  "none"}`, which upstream rejects with a 400. The overlay treats `tools: []`
  as a no-tools request when `tool_choice` is `"none"`/omitted, while still
  rejecting genuinely invalid combos.

### Source-build patches (applied at image build time)

Local backports of upstream fixes not in `VLLM_REF` v0.28.1rc0, applied by
`Dockerfile.fullbuild` from `patches/vllm/*.patch` (mirrors the aiter patch
loop). Re-verify each patch applies cleanly on the new ref when bumping
`VLLM_REF` — and drop any whose fix has since landed (see
AGENTS.md §4). (#51812/#51837 were carried as patches on v0.27.1, merged
upstream 2026-08-11, and dropped with the v0.28.0 bump.)

- **Honor `drop_eagle_block` in `MambaManager`**
  (`patches/vllm/48375-mamba-drop-eagle-block.patch`,
  [#48375](https://github.com/vllm-project/vllm/pull/48375), open upstream):
  without it, MTP + prefix caching on hybrid GDN models can leave the final
  matched page holding recurrent state written over draft positions that
  verification later rejects — silent corruption spread to every later request
  sharing the prefix (#43559, #50188). The fix lowers the cache-hit search
  ceiling by one page.

- **Keep packed GDN decode beta in FP32**
  (`patches/vllm/53877-gdn-packed-decode-beta-fp32.patch`,
  [#53877](https://github.com/vllm-project/vllm/pull/53877), merged upstream
  2026-08-30, first release: v0.29.0rc1): the packed GDN decode kernel
  rounded the FP32 `sigmoid(beta)` to the input dtype (bf16) before folding it
  into the FP32 recurrent state. That kernel is the exact decode path
  Qwen3.8-27B runs on ROCm (the aiter RDNA fast path only serves
  Qwen3-Next's interleaved GQA layout), and the per-step rounding error
  compounds with decode length, degrading long-context accuracy. **Impact**:
  the MTP3 spec-decode path (fused sigmoid-gating kernel) was already FP32 and
  is unchanged; the fix corrects the packed/non-spec decode path (used at
  CUDA-graph capture and non-spec decode), with no measured throughput cost —
  verified 2026-09-02 (8/8 packed-decode tests incl. the upstream regression
  test, coherence passed, decode at parity), see
  [`benchmarks/2026-09-02_qwen3.8-27b_53877_backport.md`](benchmarks/2026-09-02_qwen3.8-27b_53877_backport.md).

### AITER source-build patches (applied at image build time)

`Dockerfile.fullbuild` applies `patches/aiter/*.patch` to the pinned
`AITER_REF` (v0.1.20) before building the wheel. Together they make aiter's
unified attention work and run well on RDNA4 (`gfx1201`):

- **`unified-attention-bf16-kv.patch`** — with bf16 KV the staged K/V tiles of
  the 2D-decode and 3D kernels overflow RDNA's 64 KiB workgroup LDS
  (`"out of resource: shared memory, Required: 65792, Hardware limit: 65536"`,
  a hard startup abort). Caps `TILE_SIZE` to 32 (and drops `attn_stages` to 1
  on 3D) when `kv_cache_dtype == bfloat16`, gated on `arch.is_rdna`. This is
  the fix for upstream [ROCm/aiter#4329]
  (https://github.com/ROCm/aiter/issues/4329) / [vllm#48723]
  (https://github.com/vllm-project/vllm/issues/48723), still open upstream.
- **`unified-attention-gfx1201-tune.patch`** — per-arch launch-config tuning
  for gfx1201, mirroring the in-file `gfx1151` precedent. 3D long-context
  decode (the kernel behind decode at depth): `attn_warps` 2 → 4 makes the
  attention kernel **~1.4–1.9× faster** at 16k–128k context, bitwise-identical,
  for both bs=1 and the MTP batch-decode shape (q_len>1). 2D large-prefill:
  `num_warps` 4 → 8 gives ~7% on the attention prefill kernel. These are
  attention-kernel wins; end-to-end decode is dominated by the 48 GDN layers +
  MTP + TP, so the system-level effect is within noise (see the tuning doc). See
  [`benchmarks/2026-08-25_gfx1201_ua_tuning.md`](benchmarks/2026-08-25_gfx1201_ua_tuning.md)
  for the full sweep.
- **`allowed-archs-gfx1201.patch`** — accept gfx1201 (and the rest of the RDNA
  family) in `csrc/cpp_itfs/utils.py` `allowed_archs` so a
  `GPU_ARCHS=gfx1201` build-time prebuild path doesn't hard-assert (matches the
  runtime JIT list in `aiter/jit/core.py`; inert here since this repo runs
  `PREBUILD_KERNELS=0`).

Re-verify each patch applies cleanly on the new ref when bumping `AITER_REF`
(they are version-locked to v0.1.20).

### Runtime env knobs

Non-standard environment set across `compose.yaml`, `Dockerfile.fullbuild`,
and `env/2xr9700.vllm.common` (loaded via `env_file`):

| var | value | why |
|:----|:------|:----|
| `GPU_MAX_HW_QUEUES` | `1` | required: multiple queues cause a 55-63% decode regression on RDNA4 |
| `NCCL_P2P_DISABLE` | `1` | required on this host: the GPUs sit on separate PCIe root ports, and enabling P2P collapses decode ~10× even though RCCL establishes P2P channels (also rules out the P2P all-reduce HIP kernels) |
| `NCCL_MIN/MAX_NCHANNELS` | `4` | bandwidth sweet spot for two PCIe 5.0 x8 root ports, P2P off (data in BENCHMARKS.md) |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `1` | needed for the ROCm stack |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | avoid scratch reallocation stalls |
| `HIP_FORCE_DEV_KERNARG` | `1` | force device-side kernel args |
| `LD_PRELOAD` | `libamd_smi.so` | expose GPU metrics via amd_smi |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | prefer hipBLASLt GEMMs |
| `SAFETENSORS_FAST_GPU` | `1` | fast safetensors load on GPU |
| `PYTORCH_NVML_BASED_CUDA_CHECK` | `1` | NVML-based CUDA check on ROCm |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | enable Triton FA on AMD |
| `TOKENIZERS_PARALLELISM` | `false` | avoid HF tokenizer thread churn |
| `HOME` | `$HOME` (compose `user:`) | container runs as host user; whole home mounted, caches redirected under `~/.cache` (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `AITER_JIT_DIR`, `TILELANG_CACHE_DIR`) |
| `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` | `0,1` | select the two R9700s |
| `HIP_ARCHITECTURES`/`AMDGPU_TARGETS`/etc. | `gfx1201` | target the R9700 ISA |

The `VLLM_ROCM_USE_AITER_*` flags in `env/aiter-unified-attention.env` enable
only AITER's unified attention; MoE/linear/RMSNorm stay on stock vLLM kernels
(AITER's MoE/FP8 backends don't support `gfx1201` yet). The attention backend
is configured for gfx1201 by the aiter patches above (bf16-KV LDS caps +
per-arch tuning); `tools/tune_ua_config.py` re-runs the config sweep to
re-validate or retune after an `AITER_REF` bump.

Key tuning decisions:
- **MTP speculative decoding** (dense profiles): MTP4 on Qwen3.6-27B (~72%
  acceptance, ~doubles decode). Qwen3.8-27B peaks at **MTP3** (its MTP head
  accepts drafts poorly past position 3, so more drafts waste compute —
  bf16-KV sweep: MTP3 57.6, MTP2 56.0, MTP1 45.6, MTP4 49.2, no-MTP 32.0
  tg32). **MTP4 is now enabled on 35B-A3B** (2026-08-24): the #47087 MoE
  token-loop bug (fixed upstream by #51113 in v0.27.1) was re-tested clean on
  the v0.28.0 build and delivers a ~2x decode win (tg32 194.9 vs 87.8 MTP-off);
  the old "disabled" state is documented in [`archive/DEADENDS.md`](archive/DEADENDS.md).
- **bf16 KV cache** (current default, `VLLM_KV_CACHE_DTYPE=bfloat16`): higher
  KV fidelity than fp8. It uses more K/V bytes than fp8, so fp8 (with the
  calibrated-copy scale fix, halving KV memory) remains the option when context
  length is the binding constraint. The 2026-08-22 quality A/B found fp8
  (calibrated or scale-1.0) indistinguishable from bf16 on PPL and long-context
  recall, so bf16's extra fidelity costs nothing measurable and it is the
  safer default; the prior "garbage" output was MTP token loops, not the KV
  dtype.
- **Tuned dense w8a8 block-FP8 configs** (`fp8_configs/N=*,K=*,device_name=AMD_Radeon_R9700,...json`):
  the 5 per-GPU weight shapes for both 35B-A3B and 27B (TP=2) are now tuned for the
  R9700 via `tools/tune_fp8_dense.py`. Sweeps 576 Triton tile configurations per shape
  with fp32-reference numeric gating (eliminating structurally invalid configs — BK=256
  mixes 128-wide scale groups). Same-boot A/B vs stock defaults: **35B tg32 +4%, tg128 +5%;
  27B tg32 +19%, pp2048 +3%** (tg128 flat).
- **Tuned fused MOE configs** (`fused_moe_configs/E=256,N=256,...json`): tuned via
  `tools/tune_fused_moe.py`. vLLM keys the config file on the per-GPU geometry at
  TP size 2 (`E=256,N=256` = local experts × local intermediate); an earlier
  `E=256,N=512` file never matched, so the server silently ran the stock MoE
  config. Enabled via `VLLM_TUNED_CONFIG_FOLDER=/app/fused_moe_configs`.
- **`--max-num-batched-tokens 4096`** is required for the MoE model (its
  gated-delta layers force an attention block size of 2112 tokens).
- **`--max-num-batched-tokens 2048`** on Qwen3.8-27B (2026-09-03 A/B): at the
  8192 default, a 100K+ prefill running alongside a decoding request stalls
  that request's token generation 150–200x (47 ms → p50 3.5–4.4 s, p99 up to
  9.8 s — each scheduler step is one big prefill chunk). 2048 caps the step at
  one Mamba-block grid stop → ~1 s ITL during prefill, flat big-prompt TTFT,
  −3.4% pp2048 (+27 ms). Below 2048 gains nothing (step floored at the
  1600-token Mamba checkpoint grid). Full record:
  [`benchmarks/2026-09-03_qwen3.8-27b_concurrent_itl.md`](benchmarks/2026-09-03_qwen3.8-27b_concurrent_itl.md).
- **V1 model runner (V2 tested and rolled back, 2026-09-03)**:
  `VLLM_USE_V2_MODEL_RUNNER=1` on the pinned v0.28.1rc0 is fully correct on
  this stack (coherence, 54K × 10, image+MTP, c2 condense smoke) and prefill
  is +3.6–4%, but decode and MTP acceptance are flat — the #54498
  acceptance-driven decode hypothesis did not materialize, and V2 is not a
  platform-default path for this model family upstream. The line stays
  commented out in `env/qwen3.8-27b.env`; revisit conditions are in
  [`benchmarks/2026-09-03_qwen3.8-27b_v1_vs_v2.md`](benchmarks/2026-09-03_qwen3.8-27b_v1_vs_v2.md).

### MTP concurrency bug

[#35288](https://github.com/vllm-project/vllm/issues/35288): MTP spec-decode
corrupts output when 4+ decode sequences share a batch (garbage header →
repetition loop → `max_tokens`). **Workaround**: `--max-num-seqs 2` everywhere
(compose default + explicit per profile), so the batch never reaches the
threshold — verified with the #35288 repro (4/6/8 concurrent requests → all
coherent) and the 400-request stress test. See the AGENTS.md watchlist for
status.

### Upstream issues

The live upstream-issue watchlist, the known-to-ignore list, and the update-check
workflow (pin bumps, patch re-verification, triage filters) are maintained in
`AGENTS.md` ("Checking for Updates"). Resolved/superseded issues, dead ends, and
stale triage snapshots live in
[`archive/DEADENDS.md`](archive/DEADENDS.md).

## Performance

Measured on 2× R9700 (gfx1201), single request, thinking off, vLLM 0.28.1rc0 +
the local patch, torch 2.13 (ROCm 7.14.0), tuned MoE/dense GEMM configs. The
top Qwen3.8-27B row is the current default stack (**MTP3**, 256K context,
**fp8 KV**, 2026-08-28). Since 2026-08-27 the MTP profiles
also pass `--no-async-scheduling` (vLLM turns async on by default for MTP,
which is the open `#51571` accepted-count race + the `#54039`/`#32275` ROCm-CI
hang combination); re-bench shows decode parity — see
[`benchmarks/2026-08-27_qwen3.8-27b_no_async_scheduling.md`](benchmarks/2026-08-27_qwen3.8-27b_no_async_scheduling.md).
Since 2026-09-02 the stack also carries the #53877 GDN decode-beta FP32
backport (build-time patch; correctness fix, no perf change — decode parity
re-verified, see [`benchmarks/2026-09-02_qwen3.8-27b_53877_backport.md`](benchmarks/2026-09-02_qwen3.8-27b_53877_backport.md)).
The Qwen3.6 rows are the latest
measurements on the v0.28.0 build (2026-08-24); **35B-A3B now ships MTP4** (the #47087 MoE
token-loop fix was re-validated clean — see below). Full methodology, per-run
files, and history: [`BENCHMARKS.md`](BENCHMARKS.md) and [`archive/`](archive/).

| model                     | MTP (draft #) | KV   | pp2048 t/s | tg32 t/s | tg128 t/s |
|:--------------------------|:--------------|:-----|-----------:|---------:|----------:|
| Qwen3.8-27B-FP8 (default, 2026-08-28)³ | **MTP3** | fp8 |  ~3160 |   ~62 |    ~61 |
| Qwen3.8-27B-FP8 (default, 2026-08-25) | **MTP3** | bf16 |  ~3060 |   ~67 |    ~68 |
| Qwen3.6-27B-FP8 (2026-08-24)²         | MTP4 | bf16 |   ~1730 |   **80.5** |    ~69 |
| Qwen3.6-35B-A3B-FP8 (2026-08-24)      | **MTP4** | bf16 |   ~5700 |   **194.9** |   **161.3** |

² no-async scheduling. ³ vLLM 0.28.1rc0 + the retention-interval workaround,
current live profile (fp8 KV).

### Depth sweep (Qwen3.8-27B-FP8)

Current live profile: fp8 KV + **MTP3** + 256K max-model-len, full-context
prefill at depth (2026-08-27, `--no-async-scheduling`): pp256K 1277 t/s /
TTFT 202 s, tg32 holds 41–60 t/s at every depth, coherence passed at every
depth — full table in
[`benchmarks/2026-08-27_qwen3.8-27b_depth_no_async.md`](benchmarks/2026-08-27_qwen3.8-27b_depth_no_async.md).

Earlier bf16-KV sweep (2026-08-22) for comparison:

| depth | pp2048 (t/s) | tg32 (t/s) | e2e TTFT (s) |
|------:|-------------:|-----------:|-------------:|
| 4096  |     2708.14 |     57.49 |          2.27 |
| 8192  |     2654.98 |     54.53 |          3.86 |
| 16384 |     2531.50 |     64.71 |          7.28 |
| 32768 |     2290.63 |     59.00 |         15.20 |
| 65536 |     1909.03 |     51.50 |         35.40 |
| 128000|     1445.16 |     61.12 |         89.99 |
| 200000|     1119.89 |     50.46 |        180.42 |
| 256000|      953.17 |     44.97 |        270.73 |

bf16 KV doubles the K/V bytes per attention step, so **deep prefill/TTFT is
slower than the fp8 sweep** (pp256K 953 vs 1563 t/s, ~39% down; TTFT 271 vs
165 s at d256K). Decode holds 51–65 t/s out to d200K and 45 t/s at d256K
(recorded here higher than the old fp8 figures — see the cross-build caveat in
the depth doc). Coherence passed at every depth; MTP acceptance ~33% unchanged.
Full tables and the fp8-vs-bf16 comparison in
[`benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md`](benchmarks/2026-08-22_qwen3.8-27b_bf16kv_depth_mtp3.md);
the earlier fp8-KV depth sweeps are in
[`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_depth.md)
and [`benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md`](benchmarks/08_19_qwen3.8-27b_fp8kv_mtp3_d0.md).

### 35B-A3B depth sweep and concurrency (archived)

The 35B-A3B depth sweep (tuned dense vs stock) and the long-context
concurrency head-to-head (serial/c1 wins; the `--max-num-seqs` 2 cap exists
for the #35288 MTP bug, not throughput) are archived in
[`archive/BENCHMARKS.md`](archive/BENCHMARKS.md), with per-run tables in
[`archive/benchmarks/`](archive/benchmarks/).

## Stability tests

Smoke tests, sustained-load stress, and long-context generation checks for
catching crashes, memory errors, and token-loop degeneration.

See [`benchmarks/STABILITY_TESTS.md`](benchmarks/STABILITY_TESTS.md) for scripts
and baseline results. Quick health check:

```sh
just check && curl -sf http://localhost:8180/health && echo "OK"
```
