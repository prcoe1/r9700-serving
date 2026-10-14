# AGENTS

## Rules

- **Never commit, push, open PRs, or make any remote changes targeting `andysalerno/r9700-serving` (remote: `upstream`/`andysalerno`) without explicit, direct instructions from the user — and confirm intent before doing so.**
- Always target `origin` (your own fork) for commits, pushes, and PRs unless explicitly told otherwise.

## Tools

Use `just` for all build/run workflows. Commands are defined in `justfile`.

| recipe | purpose |
|:-------|:--------|
| `just check` | validate the compose config for the selected model profile |
| `just build` | build the Docker image |
| `just rebuild` | force-rebuild (no cache) |
| `just up` | start the vLLM server (runs `check`, `ensure-cache-dirs`, `prewarm`, starts container, waits for readiness, runs warmup) + start the dashboard on :8083 |
| `just down` | stop and remove the server container and the dashboard |
| `just dashboard-down` / `just logs-dashboard` | stop only the dashboard / follow its logs (`up` restarts it) |
| `just prewarm` | build shared aiter JIT kernels in one throwaway container (runs automatically before every `up`) |
| `just bench` | benchmark the selected model via `llama-benchy` (pp2048, tg32+128) |
| `just logs` | follow container logs (compose `logs -f`) |
| `just exec <cmd>` | run a command inside the running container (e.g. `just exec bash`) |
| `just ensure-cache-dirs` | pre-create host cache dirs owned by the current user |
| `just clear-vllm-caches` | wipe compile caches (triton, torchinductor, aiter, etc.) |

`compose.yaml` interpolates `VLLM_MODEL`/`VLLM_TOKENIZER`/`VLLM_SERVED_NAME`/
`VLLM_SPEC_DECODE` from `env/<profile>.env`, which the recipes pass to compose
via `--env-file` (alongside `.env` for the build pins). Bare `docker compose up`
fails with a required-variable error by design — always go through `just`.
`.env` is untracked; create it with `cp .env.example .env`.

To switch models, set `MODEL_PROFILE` or use `--set`:

```
MODEL_PROFILE=qwen3.6-27b just up
just --set model qwen3.6-27b up
```

## Build Caveats

### Rebuild Timeouts
- `just rebuild` (no-cache Docker build) takes **40-60+ minutes** for a full vLLM
  compilation cycle (framework-base → flash-attention → aiter → vllm → runtime).
- When invoking rebuild via automation, set timeout to **at least 3600s (1h)**;
  budget **4500s (75m)** for safety on first-run or after base-image changes.
- Incremental `just build` (layer-cached) is significantly faster. Only use
  `rebuild` when base images, dependency pins, or source patches change.

### Cache Clearing
- The vLLM container runs as the host user (`compose.yaml` `user:` + `HOME` env),
  and `just up` pre-creates the host cache dirs via `just ensure-cache-dirs`, so
  Docker's daemon never recreates them as root. Cache dirs stay user-owned and
  `just clear-vllm-caches` needs no sudo.
- If the dirs were ever created by an older root-running setup (or by a bare
  `docker compose`), they may be root-owned; `just ensure-cache-dirs` detects
  this and asks for a one-time sudo to chown them back.
- `compose.yaml` mounts the whole home (`${HOME}:${HOME}`) plus
  `${HOME}/.vllm-workspace:/workspace`; cache dirs are redirected under
  `~/.cache` via env (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `AITER_JIT_DIR`, `TILELANG_CACHE_DIR`) and created lazily by the container as
  the host user, so `just ensure-cache-dirs` only pre-creates `~/.cache` and
  `~/.vllm-workspace`.
- Cache dirs managed by `just clear-vllm-caches`: `~/.cache/{vllm,triton,
  torchinductor,aiter,comgr,tvm-ffi,tilelang}` (huggingface kept: model weights).
- Always clear caches after updating `VLLM_REF`/`VLLM_VERSION` or changing
  `AITER_REF` to avoid stale kernel artifacts causing runtime errors.

### Non-root aiter JIT (required)
- aiter's JIT build falls back to `~/.aiter/jit` when site-packages isn't
  writable (non-root runs), but that dir is only added to `sys.path` when
  `AITER_JIT_DIR` is set. Without it, `import aiter.jit.module_aiter_core` and
  `aiter.ops.triton.unified_attention` fail with `ModuleNotFoundError` and the
  unified-attention backend breaks. `compose.yaml` sets
  `AITER_JIT_DIR=${HOME}/.cache/aiter/jit`.
- `just up` runs `just prewarm` first: it builds the shared aiter kernels
  (module_aiter_core, unified-attention) in one throwaway container before the
  server starts. On a fresh cache this is required — otherwise the model-
  inspection subprocess and both TP workers race to build them
  (`ModuleNotFoundError: aiter.ops.triton.unified_attention`) and a subprocess
  dying at exit leaves a stale aiter baton lock that deadlocks startup.
- A stale baton lock (`~/.cache/aiter/jit/build/lock_*` referencing a dead
  PID/container) is not auto-cleared by aiter; `just prewarm` removes it first.
- Toolchain: the runtime JIT must resolve `hipcc` to `/opt/rocm/bin`
  (`env/2xr9700.vllm.common` pins `PATH` with `/opt/rocm/bin` first; the
  `Dockerfile.fullbuild` ENV matches for future rebuilds). The venv TheRock
  SDK shim (`rocm_sdk_core`, first in the stock PATH) cannot find its device
  library on the 10.1 stream — device files moved to `rocm-sdk-device-*` +
  devel reconcile and `rocm[devel]` is not installed — so every GPU compile
  fails ("cannot find ROCm device library") and `import aiter` aborts at
  `module_aiter_core`. Symptom since the 2026-10-12 rebuild (fresh pip
  resolve); `/opt/rocm/bin/hipcc` is the same HIP 7.16 / clang 24 toolchain,
  zero skew. Repro: `hipcc --offload-arch=gfx1201` on a trivial `.hip` file
  in any throwaway container. If a future `rocm[devel]` flow stabilizes,
  re-evaluate (benchmarks/2026-10-12_aiter-0.1.25_bump.md).

### Version Pins
- All build pins live in `.env` (untracked) and `.env.example` (tracked template).
- When upgrading vLLM, update both `VLLM_REF` and `VLLM_VERSION` in **both files**.
- Key dependencies to cross-check against vLLM release notes:
  - `AITER_REF` — ROCm kernels library
  - `FLASH_ATTN_REF` — flash attention (commit pin)
  - `TORCH_VERSION` / `TORCHVISION_VERSION` — PyTorch stack
- If `just rebuild` terminates (signal 15 / timeout), check the image with
  `docker inspect localhost/vllm-fullbuild:latest` (or `podman inspect`) to
  verify completion before attempting `just up`.

### Podman (verified on podman 5.7 / buildah 1.42, docker-compose provider)
- `just` takes a runtime: `RUNTIME=podman just up` / `just --set runtime
  podman up`. `just check` (compose config) works out of the box.
- Runtime ops (`up`/`down`/`logs`/`run`/`exec`) need the podman API service:
  `systemctl --user enable --now podman.socket` (or `podman system service &`).
  Without it, `podman compose` fails to connect to
  `/run/user/$UID/podman/podman.sock`. `up`/`ps`/`down` were verified working
  with the service running.
- Builds work, but buildah silently ignores `RUN --mount=type=cache` (target
  dirs are pre-created but not shared across steps), so `just rebuild` loses
  pip/ccache cache reuse and runs slower than the Docker estimate above.

## Checking for Updates

When asked to "check for updates" (or when a new release is suspected), compare
the pins in `.env`/`.env.example` against upstream, then look for **patches
that affect this GPU setup and model combo** before recommending a bump.

### 1. Upstream release state

```sh
# vLLM — pin VLLM_REF=v0.32.0rc0 (2026-10-12; ROCm 10.1 + torch 2.13).
# Final vs rc3 is 6 commits, all HiSparse/minimax/transformers-bound —
# nothing in our paths; the final's `transformers < 5.18.0` pin (#59614)
# downgrades the 5.18.0 the rc3 image ran to 5.17.x (accepted, upstream-tested
# dep set). Carries over
# v0.29.0/v0.30.0: #55760+#55861 (dense
# retention default — the #53504-family root-cause fix), #53877 (packed GDN
# decode FP32), #53821 (AITER unified-attn metadata across graph replay),
# #54994 + #52041 (multimodal prefix-cache worker paths), ROCm perf #52033 +
# #53712 + #53818, #54713 + #55450 (prefix family), #54826 (draft
# attention_backend on MRV2), #53388 (trailing-block-drop opt-out),
# #53945 (last-block replay), #54251 (GDN warmup), #54965 (ROCm W4A16 skinny
# GEMM), #47562 (47137 content half), #48606 (Quark W4A16), #51565 (#51562
# GDN fix), #55390 + #58368 (the #54360 fix + regression fix), #59146 +
# #59175 (sparse-retention prompt-end checkpoint + prefill checkpoint
# reservation), #58434 (MRV2 padded tails as spec rows), #51694
# (incremental multimodal hashing), #51899 (source-tagged extra keys),
# #55178 (padded-tail Mamba state). v0.31.1rc0 exists (tagged 2026-10-07,
# e37e51dd2 — full main snapshot, 466 commits past v0.31.0): contains #58021
# (→ drops 58021-prefix-match-unit-single-group.patch) but NOT #55601
# (3d6853b32 merged 10-07, after the rc0 tag; re-verified on main
# 2026-10-11) — a v0.31.1+ FINAL is the bump where the carried 58021 +
# 55601 patches drop (plus carried 52619 — see 2026-10-14 state below). In-scope rc0 content: #59536 (MRV2
# GDN checkpoint metadata per cache group), #59759 (stale Mamba spec blocks
# in prefill-checkpoint steps), #58457 (block-size validation hardening),
# #59132 (new opt-in ROCM_SEGMENTED_ATTN backend — future A/B candidate);
# #56531 is ngram-only (we run MTP, N/A); #59781 removed no VLLM_* envs;
# #58997 drops mamba_cache_mode "all" (no profile passes it).
# 2026-10-14 state: the v0.31.1 final is still UNTAGGED (latest final =
# v0.31.0); v0.32.0rc0 tagged 2026-10-12 (fe341a53 — verified 2026-10-14
# via ancestor checks to contain #55601, #58021, #60533, #55504, #59853,
# #60736, #57560, #59759, #59536, #58457, #59132, #60381, #59447 and
# #52619, our carried 52619-rdna-w4a16-medium-skinny.patch) — so either
# final (v0.31.1 or v0.32) drops all three carried patches
# (58021 + 55601 + 52619).
# New in-scope content since rc0: #60533 (merged 2026-10-08, post-rc0 —
# hybrid Mamba prefix-cache boundary fixes, see the #45238 entry), #55504
# (GDN spec-decode query-width bound), #59853 (Qwen3-VL/3.5/3.8 4x M-RoPE
# headroom drop — bit-exact without YaRN, ~32 MiB/GPU win for Qwen3.8),
# #57560 (opt-in AITER FlyDSL GDN prefill backend — gfx950-validated;
# future prefill A/B candidate if gfx1201 support lands), #60736 (MRV2
# int64 row-offset fix); bump-time watch: #59447 (ROCm setup.py develop
# fail-fast build change), #60381 (transformers 5.19), #58918 (Transformers
# AOT→graph cache rework — see the #50891 entry for the carried-patch
# rebase risk). Next target:
# v0.31.1 final, else v0.32 final (watch #53479, #50409, #51599). MRV2
# is the default for all our profiles (#53183).
gh release list -R vllm-project/vllm --limit 8

# AITER — pin AITER_REF=v0.1.25 (2026-10-12; carries upstream #5790, the
# gfx1201 UA-2D D=256 tune, plus #3698 V-mask-by-value-head-size in the shared
# UA kernel and the RDNA UA LDS guard #4868). The v0.1.25 rebase DROPPED our
# old warps-8 2D-prefill key: #5790's D/SW schema axes make D-first lookup
# prefer upstream's narrower D_LEQ_* keys for every realistic shape (old key
# unreachable), and measurement shows upstream's operative
# D_LEQ_256.Q_GEQ_64.DT_bf16_any (w4/eu2/TILE-16) beats it 1.06x on
# long-context chunked prefill — see benchmarks/2026-10-12_aiter-0.1.25_bump.md
# (a w8/TILE-16 hybrid collapses to 0.78x, so no combining). Kept: 3D decode
# key + narrowed Q_LEQ_1 TILE-32 guard (stock v0.1.25 still selects TILE-64
# Q_LEQ_1 for D=256 small-Q without it). Next bump: re-check the tune patch +
# re-run tools/tune_ua_config.py (config-tree API since the v0.1.23 rebase).
# aiter#5035 (3D LDS clamp on gfx1201) CLOSED as resolved 2026-10-07 —
# covered by #4868 in the pinned tree (per-arch JSON config).
# aiter#6108 (non-causal UA support for the aiter#5683 image-input gap) still
# OPEN — ua_image_correctness_probe.py stays the guard; re-run it on any bump
# touching the UA or vision path.
gh release list -R ROCm/aiter --limit 8

# Flash Attention — pinned to a commit; compare HEAD to FLASH_ATTN_REF
# (HEAD a369df7, 2 commits past pin, both Blackwell-only — no bump).
git ls-remote https://github.com/ROCm/flash-attention.git HEAD

# ROCm base image — ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.1.0-full (10.1.0
# published 2026-10-05; torch 2.13.0+rocm10.1.0 / torchvision 0.28.0+rocm10.1.0
# / torchaudio 2.11.0.2+rocm10.1.0 all on whl-next). 10.x releases via
# TheRock releases page.
# NOTE: torchaudio is hardcoded in Dockerfile.fullbuild (not from .env) —
# bump its +rocm suffix together with ROCM_IMAGE.
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-ubuntu-24.04/tags?page_size=100&name=10." | jq -r '.results[].name' | sort -V

# Froggeric chat template — pin is the first line of chat-templates/qwen.jinja
# (currently qwen3.8-froggeric-v22.5). Compare against upstream main:
curl -sL https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja | head -1
head -1 chat-templates/qwen.jinja
```

Report what's newer than the current pins and whether the bump is worth it
(see relevance filters below). Do **not** auto-bump pins. Chat-template bumps
are lower-risk than build pins: it's a pure Jinja swap, no rebuild — refresh
`chat-templates/qwen.jinja`, bump the README pin note, `just down && just up`
(the in-memory prefix cache is cleared on restart anyway).

### 2. Scan for open issues affecting this setup

Beyond checking the watchlist (below), actively search for **new** open
issues/PRs each time. Report anything that changes the picture; do **not**
auto-apply fixes.

```sh
# Re-check watchlist status (open/closed/resolved) + any new labels.
# Pruned 2026-10-14 to stack-direct items only (gfx1201 + tracked Qwen models +
# MTP + align prefix-cache + UA + build/parser path). Excluded items live in
# git history; re-add if the stack changes (new model, drafter, backend, or cap).
for n in 35288 48375 52872 47602 52520 45238 49125 53479 48815 61281 50551 51571 51599 54360 55495 55497 57707 50891 50409 55533 55894 55766 60008 55196 52817 48007 60911; do
  gh issue view $n -R vllm-project/vllm --json state,title,updatedAt 2>/dev/null \
    | jq -r '"\(.state) | \(.updatedAt) | \(.title)"'
done

# New open issues by theme (MTP, hybrid, ROCm, prefix caching):
gh search issues -R vllm-project/vllm --state open --limit 25 "MTP" \
  --json number,title,updatedAt | jq -r '.[] | "\(.number) | \(.updatedAt) | \(.title)"'
gh search issues -R vllm-project/vllm --state open --limit 25 "hybrid" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "ROCm" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "mamba" --json number,title
# RNDA4 (and correctly-spelled RDNA4) + gfx1201 — surfaces Radeon/RDNA-family
# issues that the ROCm/hybrid/MTP themes above may miss:
gh search issues -R vllm-project/vllm --state open --limit 25 "RNDA4" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "RDNA4" --json number,title
gh search issues -R vllm-project/vllm --state open --limit 25 "gfx1201" --json number,title
```

Apply the relevance filters from step 3 when triaging results: track **only**
issues that affect this stack (`gfx1201`/ROCm, hybrid GDN/Mamba path, MTP/
speculative decoding, prefix caching align mode, fp8 KV, AITER unified
attention) **and** one of the tracked models (Qwen3.6-27B, Qwen3.6-35B-A3B,
Qwen3.8-27B). Everything else is out of scope — NVIDIA/CUDA-only issues
included, even if the model matches — and goes to the not-applicable list
below. For a candidate issue, read its body and comments: confirm the root
cause matches a path this stack actually reaches (e.g. check whether an
option the issue requires — async scheduling, KV connectors, DSpark,
turboquant KV, NVFP4 weights, explicit `--block-size` — is even enabled here)
before adding it to the watchlist.

### 3. Relevance filters — does the update matter here?

This stack is not a stock vLLM install. A fix/perf change only matters if it
touches one of:

- **GPU**: `gfx1201` (RDNA4, 2× R9700), ROCm 10.0. ROCm-only issues and
  AITER unified-attention paths are in scope; NVIDIA/CUDA-only fixes are not.
- **Models**: Qwen3.6-27B (dense, MTP4), Qwen3.6-35B-A3B (MoE, MTP4),
  Qwen3.8-27B (hybrid GDN, MTP3, 256K context, fp8 KV). Anything touching:
  hybrid Mamba/GDN models, MTP/speculative decoding, prefix caching
  (align mamba cache mode), fp8 KV, or `ROCM_AITER_UNIFIED_ATTN` is in scope.
- **Chat template**: froggeric `chat-templates/qwen.jinja` (pinned, e.g.
  v22.5). A newer version matters when it changes prompt rendering in ways
  this stack hits: history re-render must stay byte-identical to generated
  tokens (KV-cache/prefix-cache invariance) for thinking-off multi-turn,
  tool-argument formatting for the `qwen3_coder` XML parser, or reasoning/
  tool-error heuristics. Template bumps need no rebuild (see step 1).
- **Known-bug watchlist** (search/check these before recommending a vLLM bump):
  - `#35288` MTP concurrency corruption — mitigated by `max-num-seqs 2`
    on MTP profiles; qwen3.8-27b runs MTP1 at cap 2 since the 2026-10-08
    operator decision (bf16 KV + c2; the 2026-10-06 4-wide hold check —
    acceptance 79–93%, no garbling —
    benchmarks/2026-10-06_mtp1_c4_hold.md — stands as evidence for any
    future raise, still below the 4+-decode-sequence corruption threshold
    at K=1 depth);
    precise root cause is the async+MTP align race `#51571` (fix `#51599`),
    and the corruption requires async scheduling, which we disable.
  - `#48375` MambaManager ignores `drop_eagle_block` (MTP + prefix caching
    corrupts hybrid recurrent state, #43559/#50188) — **carried as a local
    patch** (upstream PR open). Drop when a release contains the fix.
   - `#52872` GDN/hybrid prefill peak under-predicted; `--max-num-batched-tokens`
     also sizes the CUDA-graph pool. qwen3.8-27b pins 1024 (concurrent-ITL A/B:
     8192 default stalled the co-decoder 150–200x; 2048 → ~1 s ITL, tightened
     to 1024 on 2026-09-21, reaffirmed by the 2026-10-07/10-08 chunk ladders;
     −3.4% pp2048 at the 2048 step). Re-check headroom + re-run `benchmarks/conc_itl_probe.py` on any
     bump that changes chunk/graph-pool sizing.
  - `#47602` MTP draft acceptance decays with context length (Qwen3.6-27B)
  - `#49125` stale partial prefix-cache hash resurrected after full-block
    promotion (pure-Python `BlockPool` bug in the fine-grained/partial
    prefix-caching path `#45939`/`#46384`). Only reachable once `#45238` is
    fixed and prefix caching actually hits — monitor post-fix
  - `#52520` align-mode admission livelock near KV-pool ceiling (open)
  - `#45238` hybrid prefix caching drops to 0% in align mode (open) — the
    binding constraint on this stack (same family: `#51250`, `#51198`). Root cause:
    `BlockPool.cache_full_blocks` skips Mamba align-mode null blocks, so only
    ~1 checkpoint hash per request is registered and a missing Mamba
    checkpoint vetoes every attention-group hit. Live geometry: `block_size=832`
    on bf16 KV (1600 on fp8; live moved bf16 on 2026-10-08), so incremental
    multi-turn prefixes never hit —
    measured **0% on the multi-turn probe (most recently 2026-10-12, 4-turn,
    v0.31.0 + aiter v0.1.25, MTP1, bf16)**. Note the cumulative
    `vllm:prefix_cache_hits_total` is non-zero: caching *does* hit on
    repeated-identical-prompt workloads — the failure is specific to the
    incremental shared-prefix pattern, not a global no-op. Fixes in flight:
    `#48815` (MTP align retention),
    **`#53479`
    (the leading candidate — retention-aware boundary materialization +
    removal of the speculative one-block back-off; open, conflicting, author
    confirms the head is a superseded branch since merged `#54713` touched
    the same path — correctly NOT carried as a local patch)**,
    `#52789` (internal prefill checkpoints — merged 2026-08-22, Kimi-K3/
    FlashKDA TTFT win, not a fix for this geometry),
    **`#50551`/`#60603`/`#60760` stack (finalized Mamba decode checkpoints
    [1/N base since 2026-07-31, unreviewed — last human activity 2026-08-14]
    + hidden-state drafting [2/N,
    conflicting/needs-rebase] + match-unit-aligned retention [3/N, 2026-10-09,
    no reviews, empty validation] — evaluated 2026-10-09: NOT carried, ~1900
    lines of unreviewed core scheduler/KV-manager + worker changes colliding
    with our `#50409`/`#48375` patches; monitor for merge, then take
    via version bump), `#60533` (merged to main 2026-10-08, post-rc0 — in
    v0.32.0rc0 (verified 2026-10-14), in no final release yet: fixes
    inconsistent hybrid Mamba prefix-cache boundaries — the
    Mamba prompt checkpoint no longer double-applies the `P-1` cap and the
    EAGLE block drop, attention lookup may inspect the cached EAGLE proof
    beyond the final hit limit, and Mamba checkpoint calc/registration is
    made consistent across scheduler and workers; extracted from #59826.
    NOT carried — unreviewed on our geometry and it lands in the same
    scheduler/KV territory as our carried `#48375`/`#50409`
    patches, so re-verify those against it on the bump; monitor for the
    final release that contains it),
    `#61281` (NEW 2026-10-12, open, kimi author — GPU-independent reproducer:
    the align prompt-boundary Mamba state is freed mid-request via three
    paths (`remove_skipped_blocks` retiring the hashed boundary block, the
    running-state swap, the CoW-copy free) and recycled by concurrent
    prefills before the next turn arrives, plus LRU-order eviction at
    request end — a registration fix alone (`#53479` family) will NOT
    restore multi-turn hits until this eviction race is fixed; no fix PR
    yet). **When a real fix
    merges**: prefer the version bump; carry a local patch only if no
    available release contains it. Related: `#52897` (priority-scheduling
    variant — N/A, we don't use priority), `#53749` (same family on two more
    hybrids — reinforces the checkpoint fix is the binding constraint).
     - `#55766` (2026-09-07, open): Qwen3.8/3.5 hybrid GDN + mamba align +
      prefix caching — a prefill ending **4–10 tokens past a block boundary**
      writes a bad Mamba/GDN checkpoint; a later block-aligned prefix-cache hit
      restores it → **NaN logits** from step 1 (token-0 `"!"` spam to
      max_tokens; identical retries fail until the cache turns over).
      **This stack's exact model + cache mode + TP2 + bf16** (repro at block
      816 on v0.28.0; ours is 832 on bf16 KV since 2026-10-08, was 1600 on
      fp8). **Currently masked**: the carrier is a
      prefix-cache *restore*, and our incremental multi-turn pattern is the
      `#45238` 0%-hit no-op (repeated-identical prompts DO hit — the window
      is narrow but real). No in-repo mitigation; the client-side workaround
      is `cache_salt` (or pad the prompt **start**) when the prior prompt
      length mod block_size ∈ {4,6,8,10}. **Monitor**; run
      `benchmarks/nan_checkpoint_probe.py` on any NaN / `"!"`-spam / empty
      reply, on any bump touching the hit back-off geometry (#53388/#53945/
      #53479 family), or when spec decode changes (no-spec hits reach further
      than MTP's 1–2-block back-off — the masking analysis thins). History:
      CLEAN on v0.29.0+MTP3 (2-block back-off, unrestorable), CLEAN on
      v0.30.0+MTP1 with live 14400-token restores (1-block back-off);
      full diary in `benchmarks/2026-09-10_qwen3.8-27b_55766_nan_probe.md`.
  - `#52817` RFC: hybrid SSM + SpecDec + APC re-runs the last full block on a
    prefix hit (832 tokens here on the bf16-KV profile; 1600 on fp8),
    bounding the prefix-cache win for MTP even after `#45238` is fixed.
    Implementation `#53945` landed 2026-09-16 and IS in v0.30.0 — re-run the
    prefix-cache + conc-ITL probes on the v0.30.0 bump to measure the bound;
    follow-up `#57329` adapts it for Mamba2 (merged; in v0.31.1rc0 — Mamba2 scope, no action here).
  - `#51571` async MTP align accepted-count race (open): async scheduling +
    MTP + hybrid GDN + `mamba-cache-mode align` → accepted-token D2H counts
    gathered from a mutated `InputBatch` after `condense()` (repeated/dropped/
    garbled tokens). Mitigation: `compose.yaml` passes `--no-async-scheduling`
    for all spec-decode profiles (tracks `VLLM_SPEC_DECODE`); re-check upstream
    before removing. Fix PR `#51599` (open, retargeted `[Bugfix][MRv1]` — all
    our profiles run MRV2, which shrinks the exposed surface, but
    `--no-async-scheduling` stays): if it lands in a release we adopt, re-test
    before dropping `--no-async-scheduling`. **New external repro 2026-10-06**
    (NVIDIA RTX 4090, `dbirks/Qwen3.8-27B-W4A16-AutoRound` — our model arch,
    V1 runner + async + align): repeated tokens at temp 0; root-caused to a
    double-permutation of the accepted counts after a row reorder (`#53613`'s
    wait-earlier alone is insufficient — removing the `prev_positions`
    gather is the fix); reporter: "still present in 0.31.0, patch applies
    cleanly to 0.28/0.30/0.31/main". Our mitigations (MRV2 +
    `--no-async-scheduling`) still hold — monitor. (Same-combination question
    `#54039` — ROCm CI disables async+MTP while the default enables it —
    folded here; watch for a merged default change.)
  - `#54360` (open): spec decode drives prefix-cache hits to **0** on
    Qwen3.8-27B hybrid GDN align — **nightly-only** (not in our pin; our
    probe hits since the `#55390`+`#58368` pair, both in-pin since v0.31;
    1-block MTP back-off, validated live 2026-09-27). Re-run the
    prefix-cache probe after any bump touching this path. New PR `#52244`
    (open, conflicting — MTP+GDN hits at hash-unit boundaries) evaluated
    N/A here: gated on `--prefix-match-unit`, which no profile passes. Do
    not carry; re-check only if fine-grained hashing is ever enabled.
   - `#55495` (open): qwen3_coder engine parser emits a truncated `arguments`
     string with a leaked `</parameter` tag when the model writes a
     `<parameter=>` element slightly off (name missing / `>` missing / stray
     `=` — typical for shell-command values with `$`/newlines); the streamed
     partial converter swallows everything up to the closing tag's `>` as a
     key while the final conversion does not, so the prefix invariant breaks
     and the client keeps an unterminated non-JSON string (raw tags leak
     into content; Responses clients then 400 on every later turn).
     **Seen live on this stack 2026-10-01** (raw `</parameter>` in an
     opencode session + a one-call→two-mangled-calls split in another).
     **Fixed upstream by #55497** (open + conflicting, in no release —
     verified NOT in v0.31.0rc3) **and carried as a local patch**
     (`patches/vllm/55497-qwen3-malformed-parameter-drop.patch`, both
     upstream commits, source-only: shared `([^>\s<]+)` param-name group +
     a warning — tool name + lengths only — in `_flush_arg_converter`
     instead of the silent `None`). Drop when a pinned `VLLM_REF` contains
     the merge. **Validated pre-build 2026-10-01**: upstream
     `TestMalformedParameterElements` FAILS 7/10 against the running
     v0.31.0rc3 container (positive control) and the full
     `tests/parser/engine/test_qwen3.py` PASSES 61/61 with the fix applied
     at runtime. Related open (same `_flush` area — not carried):
     `#58407` (copied tool closers kept inside values; semantic tension
     with drop-malformed, monitor). Live no-regression guard:
     `benchmarks/tool_truncation_probe.py` complex-values check.
  - `#57707` (open): qwen3 closes `</function>` with the final
    `</parameter>` unclosed → trailing argument lost/clipped (fixes
    `#57699`). Same engine-parser path as the carried `#55497` patch,
    different lines. When it merges, re-run the guard above and check
    interaction with the `#55497` patch.
  - `#50891` (open, added 2026-09-24): AOT compile-cache key ignores
    `limit_mm_per_prompt` (`ModelConfig.compute_hash()` lists it in
    `ignored_factors`), so same-model runs with different image caps collide
    and crash in `profile_run` (`AttributeError: 'NoneType' object has no
    attribute 'size'`; can also flip attention-backend selection — wrong
    twice over). Surfaced via dup `#58203` (Qwen3.8-27B-FP8 + images + MTP,
    NVIDIA/B200). **Not exposed here today**: all qwen3.8-27b runs share
    image-cap 99 (raised from 1 on 09-19 once the #40707 fix landed — then
    via the local #40709 patch, now via #51603 in the pin;
    `multi_image_probe.py` PASS) and profile model names differ, so hashes
    don't collide.
    Hygiene rule: never alternate image caps / `--language-model-only` for
    the same model on a shared compile cache without clearing it first.
    Monitor for a fix (key the cache on the multimodal config).
    **Carried as a local patch since 2026-10-02**
    (`patches/vllm/50891-mm-cap-compile-cache-key.patch`, own fix — no
    upstream PR exists): keys the folded
    `multimodal_config.limit_per_prompt` explicitly via `normalize_value`
    (dropping the ignore entry alone is a silent no-op — the caps arrive as
    an InitVar, never retained on the instance; verified hashes still
    collided before the explicit factor, DIFFER after, in-container).
    Drop when a pinned `VLLM_REF` keys the compile cache on the multimodal
    config. **Bump-time risk 2026-10-14**: `#58918` (merged to main
    post-v0.32.0rc0 — replaces Transformers AOT caching with graph caching)
    lands in the same compile-cache territory; expect a rebase conflict or
    an obsolete patch on the v0.32 bump — re-verify the hash-collision
    behavior (InitVar `limit_per_prompt` vs `normalize_value` factor)
    before rebuilding.
  - `#55894` (2026-09-08, open): hybrid Mamba + MTP silently corrupts
    requests (0.2–1% of a mixed workload, up to 7% targeted) when a
    request's first decode step is scheduled alongside a long chunked
    prefill — `calculate_reorder_batch_threshold` takes the min over all
    attention groups, and a drafter whose backend auto-selects to a lower
    threshold (FlashInfer=1 vs Mamba=1+k) puts the draft-decode rows behind
    the continuing prefill chunk; the Mamba prefill kernels then run on
    them (slot 0 only, the k speculative slots never written → garbage
    recurrent state from ~token 5 on, word salad / special-token loops).
    Repro is Mamba2/Nemotron-3.5 on NVIDIA. **Structurally mitigated for
    this stack**: every profile pins the drafter
    `attention_backend: ROCM_AITER_UNIFIED_ATTN` in `VLLM_SPEC_DECODE`
    (same as the target) — the issue's own table shows drafter-backend
    pinning → 0/400. **Monitor**; if a profile ever unpins its drafter
    backend, re-test before shipping.
  - `#51599` (2026-09-01, PR, open): the `#51571` fix — see the `#51571`
    entry above (retitled `[Bugfix][MRv1]`).
  - `#55601` (merged 2026-10-07, `3d6853b32` — **in the pinned
    `v0.32.0rc0`**): align-mode `add_request` used to seed the running-state
    block column by the scheduler block size instead of the Mamba block size
    (silent wrong-state read or IMA on prefix-cache resumes; ex-`#53798`,
    same line as the small-block-drafter `#55600`). No patch, no watch —
    the fix rides the pin. Re-verify on any bump that reworks the precopy
    path.
  - `#50409` (2026-08-31, PR, open): when the prompt length is an exact
    multiple of the block size, align prefill runs as one chunk and the only
    cached Mamba state sits at `num_tokens`, which `get_computed_blocks`
    caps below → 0-token Mamba hit. Adds the replay boundary as a mandatory
    chunk stop. **Carried as a local patch**
    (`patches/vllm/50409-mamba-replay-boundary-chunk-stop.patch`, re-cut for
    `v0.32.0rc0`). **Caveat**: the old inertness evidence was fp8/1600
    geometry (`max_num_batched_tokens=1024` < 1600 block); live is bf16
    (block 832) since 2026-10-08, so re-measure on 832 geometry
    (exact-multiple 832/1664/2496 repeat ladder). Drop when a pinned
    `VLLM_REF` contains the merge. Bump-time note: `#60533` (in-pin) and
    PR `#52244` (open, conflicting) land in the same scheduler territory —
    expect a rebase.
   - `#55196` (2026-09-03, RFC, open): fp8 KV yields only 1.00x–1.84x (not
     2x) cache capacity on Mamba/GDN hybrids — the never-quantized bf16
     Mamba page pins the unified hybrid block (`#37121`/`#40696` family), so
     `--kv-cache-dtype fp8` barely moves `num_blocks`. Caveat on fp8-KV
     capacity expectations (alongside the `#52793` calibration note).
     Proposed levers (quantize Mamba state, decouple per-group page size)
     unimplemented anywhere in vLLM. Monitor; no action.
   - `#55533` (2026-09-06, open): Hybrid GDN (Qwen3.5/3.8 27B-class) + MTP
     scheduler caps at ~3 concurrent sequences at batch ≥ 4 — acceptance/
     throughput collapse (8-wide batch runs `[2,2,2]` only; `bs ≤ 3` healthy).
     Root: mamba cache budget shared between target states + MTP draft slots in
     scheduler accounting (scales with GDN layer count). **Relevant** — the
     cap was raised 2→4 with MTP1 on 2026-10-06 after a clean 4-wide hold
     check (no cap-out, acceptance 79–93%; K=1 slips under the pathology),
     then returned to 2 on 2026-10-08 (bf16 KV + c2 operator decision; the
     hold evidence stands for any future raise).
    WIP fix `PR #55617` (2026-09-06; 2026-09-12: still OPEN +
    CONFLICTING/WIP). Re-probe before raising MTP depth past K=1 at 4-wide.
  - `#60008` (2026-10-05, open): align-mode hybrid prefix caching costs
    11–16% throughput even at 0% hits — per-group eager block-table ops,
    per-step align kernels outside CUDA graphs, block-boundary prefill splits
    (Nemotron-3.5-Lightning repro). This is the tax this stack pays on every
    multi-turn request while `#45238` holds at 0% hits; the `#45238` fix would
    unmask the hits but the per-step align cost stands on its own. Monitor;
    no fix PR yet.
  - aiter#5683 (open, AMD gfx12 team): gfx1201 AITER unified attention
    returns wrong results on image input (non-causal attention path).
    Measured by a #57838 commenter on an R9700 2026-10-08: a 6-item
    vision/OCR check drops 6/6→3/6 with `VLLM_ROCM_USE_AITER=1`, text-only
    unaffected; AMD confirmed (2026-09-20) they will work on the non-causal
    path. **Isolated 2026-10-09** (#57838 thread): `VLLM_ROCM_USE_AITER=1`
    + `--attention-backend TRITON_ATTN` restores 6/6 — the corruption is the
    UA non-causal path, not AITER FP8 linear. AMD's fix work: aiter#6108
    '[Triton/Gluon] Add non-causal support to unified attention' (OPEN; the
    feature request aiter#5827 was closed 2026-10-01 in favor of it).
    **Our path does not reproduce it**:
    `benchmarks/ua_image_correctness_probe.py` PASS 12/12 + two-image on the
    live v0.31.0 server 2026-10-08 (record:
    `benchmarks/2026-10-08_qwen3.8-27b_ua_image_correctness.md`). Re-run the
    probe on any bump touching the UA or vision path, and when aiter#6108
    lands (a fix must keep it green).
  - `#59132` (merged 2026-10-06, in v0.31.1rc0): new opt-in
    `ROCM_SEGMENTED_ATTN` backend — segmented Triton attention for
    RDNA3/3.5/4, paged decode/prefill/verification, native FP16/BF16 KV,
    FP8 KV gated to gfx12, heads 64/128/256, GQA 1–16, causal + non-causal;
    startup autotune OFF by default (enable via
    `--kernel-config.enable_rocm_segmented_attn_autotune=True`). Upstream's
    suggested answer to `#50264`. Monitor; future A/B candidate against
    pinned UA — attractive as a hedge while aiter#5683's non-causal gfx12
    gap is open.
  - `#48606` (merged, in-pin since v0.31 — patch dropped): native Quark
    W4A16 load path (`QuarkW4A16Int4`). Live value is the alternative
    `qwen3.8-27b-awq` profile (record:
    `benchmarks/2026-09-10_qwen3.8-27b_awq_trial.md`) — check ROCm/gfx1201
    kernel coverage on any future bump, not just the merge.
  - `#48007` (PR, open + conflicting): the upstream args half of the #47137
    truncated-tool-call divergence — **carried as a local patch**
    (`patches/vllm/47137-tool-truncation-parity.patch`, adapted from
    magiccodingman/vllm-radiance; verified live 2026-09-18 via
    `benchmarks/tool_truncation_probe.py`). The content half landed upstream
    via #47562 (in v0.30.0) — local hunk dropped on the v0.30.0 rebase, only
    the args hunk is carried now. Drop when a pinned `VLLM_REF`
    contains the equivalent.
  - thinkoff parser/template disagreement (no upstream issue/PR): the
    froggeric template pre-closes `<think>` for `reasoning_effort` none/off
    but `Qwen3Parser` only read `enable_thinking` → `content=None` with the
    answer stranded in `reasoning`. **Carried as a local patch**
    (`patches/vllm/qwen3-thinkoff-kwarg-parity.patch`, adapted from
    GGZ14/vllm-mxfp4; verified live 2026-09-18 via
    `benchmarks/thinkoff_probe.py`). Drop when a pinned `VLLM_REF` derives
    `thinking_enabled` from the same kwargs.
 
 Issues known **not** to apply (checked; re-check only if the stack changes),
 grouped by reason:

 NVIDIA/CUDA-only (incl. Blackwell/SM100/SM120/SM121/Ampere cards, CUDA
 kernels absent from ROCm builds, FlashInfer paths): #52475, #52583 (VL),
 #53180 (turboquant k8v4+MTP — same silent-corruption family, re-check if
 turboquant KV is ever tried), #52480 (qwen3_5_mtp TP≥2 load), #53387
 (compressed-tensors WNA16 MTP drafter load — we use FP8), #53887 (MTP second
 vocab embedding OOM on 24 GB), #55775 (MTP+FlashInfer long-context IMA),
 #52682 (Qwen3.8 FP8 graph-capture hang on Ampere), #53323 (DFlash2 collapse
 with ROCM_ATTN drafter — we run pinned-UA MTP), #54094 (DFlash2+YaRN zero
 prefix reuse — we run MTP), #52539/#53462 (fused GDN MTP decode kernel —
 CUDA-only, N/A on gfx1201 despite our v/k ratio now being supported),
 #50264 (Triton paged-attention fallback — we run AITER unified attn;
 upstream's 2026-10-06 answer is the new #59132 ROCM_SEGMENTED_ATTN
 backend, tracked in the watchlist),
 #54906 (`thinking_token_budget` ignored — field we never send), #56419 (CPU
 backend), #56701 (KV offload+MTP — no offload), #56774 (hidden-state
  extraction), #55291 (Qwen3.6-27B-FP8 "!"-collapse — v0.21.0/L20 report, and
  a bounded 0.28.0 repro attempt found zero collapses; N/A, but tracked model
  + sticky corruption: glance if a modern repro appears), #56736 (hybrid
  Mamba/GDN + spec decode Xid 31 in the align precopy path — DFlash2 drafter
  on a v0.13-era NVIDIA fork with async on; same precopy family as in-pin
  #55601, signal only), #57838 (RowWise FP8 linear 5–24% slower than
  ChannelWise on gfx1201 — requires per-tensor/channel dynamic FP8 weights;
  our Qwen FP8 checkpoints are block-scaled → live log selects
  `TritonFp8BlockScaledMMKernel`, different path; re-check if a non-block
  FP8 checkpoint is ever served; the 2026-10-08 thread's image/OCR
  corruption finding — tracked in aiter#5683, isolated to the UA non-causal
  path 2026-10-09 (TRITON_ATTN + AITER on → 6/6), fix work aiter#6108 (open)
  — our probe does NOT reproduce it; the `aiter.gemm_a8w8`-on-gfx1201 dev
  thread targets per-token/per-channel checkpoints (gemma4) — N/A: our Qwen
  FP8 checkpoints are block-scaled and the profile disables AITER linear — is
  recorded in the watchlist entries above), #59724 (SM120
  FLASHINFER_MLA_SPARSE MTP acceptance 0%), #58422 (Qwen3.8 GDN-hybrid MoE
  TP1 engine wedge — SM120/NVFP4/FlashInfer path, signal only).

 Non-Qwen models (different arch/checkpoint format): #52833/#48568 (GLM),
 #51530 (DeepSeek), #56605 (GLM-5.3 word salad), #55280 (GLM kpool split IMA,
 gfx942), #54924/#54451/#56380 (GLM ROCm), #56506/#52911/#57149/#57230
 (DeepSeek/gfx950 perf), #58409/#59575/#59820/#58606 (GLM-5.3/gemma/Hy4
 gfx950 perf tracking), #54114 (GLM-5.1 reasoning), #55357 (Flash-Next MTP
 collapse — #54928-family signal only), #56088/#55922 (Flash-Next), #56832
 (Flash-Next NVFP4 + marlin moe-backend — we set no moe-backend), #57532
 (GLM NVFP4 MTP load on main), #55496 (ModelOpt NVFP4 MTP experts), #54926
 (Gemma + NIXL PD), #54504 (nemotron_h prefix no-op), #54547 (Quark MXFP4
 multimodal naming), #54775 (KDA chunked-scan OOM — GLM/KDA, we run Qwen
  GDN at 2048), #57267/#57266 (Mamba2 prefix-cache + explicit
  `--mamba-block-size` — we run GDN/align, never pass block sizes), #57721
  (Mamba2 mamba-page padding ignores `num_spec` → loud startup assert with
  spec decode at K≥2 — Granite/Mamba2 + ngram repro; our servers start fine,
  so the GDN path + padding slack covers us; monitor if a fix PR touches GDN
  scope), #60791 (MiniMax-M3 PP/NIXL), #60333 (MiMoV2MTP FP8 KV scales),
  #59151 (mxfp4 MoE gfx90a backend), #60581 (gfx950 Quark W4A16 slow Triton),
   #58688 (Qwen3.8-Flash-Next-FP8 load — different model family), #59755
   (GLM-5.3-Flash prefill checkpoints),
  #47277 (Qwen3.5 native MTP vs no-MTP baseline — not a tracked model; our
  own benches show MTP wins on this stack).

 Paths not reached here: PP ranks (#51752, #55951, #54709), DP attention
 (#51957, #48255), DCP (#54761, #57228), P/D disaggregation (#54392, #54926),
 KV connectors (#51805/#51766/#40017/#53505/#53514, #56972 Mooncake,
 #45407 LMCache, #54165, #60124 MultiConnector/NIXL hybrid, #60316
 connector-forced ROCM_ATTN decode), priority scheduling (#52897, #57580 —
 priority-preemption checkpoint visibility, same family), EP (#41862),
 GPTQ qzeros on gfx1201 (#51971 — we run FP8), gfx950 MLA (#52312),
 turboquant (#53180, #53334), numa-bind (#55416), XPU (#55425), prebuilt
 rocm/vllm image (#56945 — we build from source; #58937 ROCm nightly image
 publication gap same reason), GLM page-align/offload
 (#54458, #54831), gfx1030/gfx1100 (#54728, #54438), DeepEP (#54281),
 MTP `n_predict`/token auto-defaults (#55322/#55323 — we pass explicitly),
 client `stop` strings in think output (#53066 — our clients don't send
 stop), KV offloading (#52773, #58653 SimpleCPU offload block-size
 misalignment, #58034 CPU P/D conv-layout assert), `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`
 (#53136 — never set; fault was gfx942 TP=8-specific), embedding/rerank-only
 kernels (#58060 — generative-only stack), multi-layer MTP
 layer-0 reuse (#52688/#53397 — both 27B models have
 `mtp_num_hidden_layers=1`, so N/A), in-flight model PR metadata
 (#53983/#53982 — QSA/Kpool side caches not on main), ngram spec decode
 (#56077 — we run MTP; CLOSED 2026-10-08 via #56531, in v0.31.1rc0),
 ROCm 7.2.x libhsa (#56521 — we're on 10.0),
 gfx1151 (#57493/#57494 — different GPU), gfx1201 FP8 fall-through request
 (#28649 — OP retracted: already routes to W8A8), RunAI ModelStreamer
  (#60691 — we load from local disk), pooling-task prefix cache (#60947 —
  generative-only stack), NIXL MLA descriptors (#60561), NixlConnector
  heterogeneous TP (#59110), CI image build (#60258 ci_base layer size).

 Requires an option we never pass: #53142 (align precopy IMA — needs
 explicit `--block-size`; #54199 retracted as its duplicate; the cluster
 of fix variants #55507/#55601/#55688 around the old #53798 line
 settled when #55601 merged 2026-10-07 — in the pinned `v0.32.0rc0`,
 no patch),
  #57032 (dflash drafter KV group — we run MTP; same annotation family as
  #54360), #58080 (MTP draft doesn't inherit `--hf-overrides`/YaRN — we pass
  no overrides; both 27B models run native context), #60838
  (`--mamba-block-size` inert after the `mamba_cache_mode="all"` removal —
  we never pass block sizes), #60346 (ReplaySSM ring assert on non-divisible
  group counts — use_replayssm=False), #58692 (dynamic SD
  `num_speculative_tokens_per_batch_size` + MTP ZeroDivision — we pass
  explicit K), #52317 (mamba_cache_mode "all" + DSpark startup crash —
  mode "all" removed in #58997, we run GDN/align + MTP), #61003 (DFlash
  drafter fp8 `ignored_layers` load path — we run MTP).

  Already resolved/stale for this stack: #40980 (R9700 TP2 deadlock —
  v0.19-era; AMD confirmed TP2 working, ours serves), #49851 (multimodal
  gfx1201 load failure — v0.25.1-era; we serve images), #47194 (hybrid+MTP
  tool/needle corruption report — OPEN but NVIDIA-only: 2× RTX 2080 Ti SM75,
  custom GPTQ-Pro checkpoint, `gptq_marlin` path, `sha256_cbor` hashing;
  N/A, signal only for the hybrid+prefix+MTP family),
 #54106 (KV group n:1 split — ours is 3:1, fine), #54690 (draft-only fp8
 KV crash — NVIDIA/FlashInfer paths), #56021 (RDNA force-select
 unified-attn — N/A: we pin the backend explicitly, and the reported
 stock-kernel LDS overflow confirms our LDS-cap patch is load-bearing).
  RFC/feature monitor-only: #54080 (TreeWY), #53786 (fine-grained SWA hits),
  #55916 (RDNA4 FlyDSL all-reduce), #57111 (checkpoint-aware eviction for
  hybrids — same family as #53041/#55697), #58638 (KV-cache grouping for
  hybrid + spec drafters — DFlash-only pathology, MTP unaffected; signal if a
  profile ever moves off MTP).
 - `#56190` (upstream profiler fix, new in v0.30.0): preloads
   `libtorch_cpu.so` RTLD_GLOBAL at `import vllm` for kineto/rocprofiler
   registration — **fatal on this stack** (TheRock torch 2.13 + `rocm` pip
   package): libtorch's static LLVM vs `_rocm_sdk_core`'s → duplicate
   `spirv-expand-step` registration → LLVM ERROR → abort at import (exit
   139, crash-loop with no logs). **Carried as a local patch**
   (`patches/vllm/56190-rocm-skip-libtorch-global-promotion.patch`: skip
   the preload when `rocm_sdk` is installed; costs only kineto GPU
   profiling, unused here). Drop when a pinned `VLLM_REF` guards/reverts
   #56190. No upstream issue exists — the failure is silent (no log past
   the LLVM line), so check `import vllm` in a throwaway container first
   on any bump if startup crash-loops. **Upstream fix in flight: PR
    #60911** "[ROCm][Bugfix] Run TheRock's ROCm SDK preload before
    promoting libtorch_cpu.so" (open, updated 2026-10-10) — drop the
    patch when a pinned `VLLM_REF` contains it.

### 4. Local patches vs upstream

`patches/vllm/*.patch` and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time. Before bumping any pin:

- The two parser patches (`47137-tool-truncation-parity.patch`,
  `qwen3-thinkoff-kwarg-parity.patch`) are adaptations from third-party
  forks (radiance / GGZ14 lineage), not upstream PRs — each has a live probe
  (`benchmarks/tool_truncation_probe.py`, `benchmarks/thinkoff_probe.py`).
  Re-run its probe after any bump touching `vllm/parser/`.

- The aiter patches (version-locked to `AITER_REF` v0.1.25) are **RDNA4-local
   work**, not upstream cherry-picks: `unified-attention-gfx1201-tune.patch`
   (additive gfx1201 JSON keys in the v0.1.22+ config tree: warps 4/waves 6
   on `D_LEQ_256.DT_any_bf16` 3D decode ~1.4-1.9x, plus a narrowed
   `Q_LEQ_1.DT_any_bf16` TILE-32 LDS guard for the small-Q 2D bucket
   upstream #4868 leaves at 64) and `allowed-archs-gfx1201.patch`
   (build-path arch acceptance; rebased 2026-10-12 for upstream's added
   `gfx1250` list entry). The old warps-8 `Q_GEQ_256.DT_bf16_bf16` 2D-prefill
   key was dropped on the v0.1.25 rebase — superseded by upstream #5790
   (unreachable under D-first lookup AND slower: 1.06x for
   `D_LEQ_256.Q_GEQ_64.DT_bf16_any` on chunked prefill). The old code-level
   `unified-attention-bf16-kv.patch` was retired on the v0.1.23 bump
   (superseded by #4868; unappliable since the config-tree refactor —
   archived at `archive/patches/`). After any `AITER_REF` bump re-verify
   the tuning still wins — re-run `tools/tune_ua_config.py` (config-tree
   API; with `just down` first) and re-check the LDS guard. See
   `benchmarks/2026-08-25_gfx1201_ua_tuning.md` and
   `benchmarks/2026-10-12_aiter-0.1.25_bump.md`.
- Check whether a newer `VLLM_REF` **already contains** a carried patch (the
  fix landed upstream). If so, the patch should be **dropped**, not kept.
  Verify: `gh pr view <pr> --repo vllm-project/vllm` and check the PR's merged
  status + which release tag includes it (compare tag commits via
  `git ls-remote --tags https://github.com/vllm-project/vllm.git`).
- After any pin change, verify each patch still applies cleanly on the new
  ref before building; a failed `git apply` in `Dockerfile.fullbuild` aborts
  the build. Bump the version-lock comment in each patch header too.
- Always `just clear-vllm-caches` after a `VLLM_REF`/`VLLM_VERSION`/`AITER_REF`
  change, then `just rebuild` (see Rebuild Timeouts).

### 5. Recommended bump checklist

1. Diff `.env.example` vs `.env` — keep both in sync.
2. Update `VLLM_REF` + `VLLM_VERSION` together; verify `AITER_REF` and
   `FLASH_ATTN_REF` are compatible with the new vLLM release notes.
3. Check `TORCH_VERSION`/`TORCHVISION_VERSION` against the vLLM release's
   supported ROCm/PyTorch stack.
4. Re-check the patch watchlist (step 2/3) and drop/rebase local patches.
5. Refresh `chat-templates/qwen.jinja` if froggeric shipped a newer version
   (step 1): curl from upstream `main`, bump the README pin note, then
   `just down && just up`. No rebuild or cache clearing needed.
6. `just clear-vllm-caches && just rebuild && just up`, then `just bench` to
   confirm no regression vs `README.md`/`benchmarks/` baselines.
7. Re-run the prefix-cache hit-rate probe (`benchmarks/prefix_cache_probe.py`)
   after any vLLM bump/restart and record whether
   `vllm:prefix_cache_hits_total` moves off 0% (see `#45238`). A non-zero hit
   rate on the multi-turn probe is the signal the align-mode checkpoint fix
   landed and is worth carrying/keeping.
8. Update `README.md` (patches, pins, bench tables) and commit to `origin`.

