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
| `just up` | start the vLLM server (runs `check`, `ensure-cache-dirs`, `prewarm`, starts container, waits for readiness, runs warmup) |
| `just prewarm` | build shared aiter JIT kernels in one throwaway container (runs automatically before every `up`) |
| `just bench` | benchmark the selected model via `llama-benchy` (pp2048, tg32+128) |
| `just logs` | follow container logs (compose `logs -f`) |
| `just down` | stop and remove the container |
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
# vLLM — current pin VLLM_REF=v0.29.0 (final, released 2026-09-09T08:54Z —
# the latest stable; v0.28.1 final was never cut). Bumped from v0.28.1rc0 on
# 2026-09-09 (commit 8c07ffea3d) alongside ROCm 7.14 -> 10.0 + torch 2.13.
# What the final adds vs the old rc0 pin: #55760+#55861 (dense
# prefix_cache_retention_interval default for Mamba+EAGLE / is_hybrid+MTP —
# the #53504-family root-cause fix; PREFIX_CACHE_RETENTION_INTERVAL pins
# dropped from all env files), #53877 (packed GDN decode beta FP32 — local
# backport patch dropped), #53821 (AITER unified-attn metadata preserved
# across graph replay), #54994 (multimodal SHM worker cache for
# prefix-covered items), #52041 (MM tensors not broadcast to workers when
# prefix-cache-covered), plus ROCm perf #52033 (dual-stream hipgraph decode),
# #53712 (ROCr/CLR graph-replay segfault fix, up to ~20% TPOT), #53818
# (CUDA-graph capture on current stream). Still NOT fixed in v0.29.0:
# #54716/#51599/#53479/#48375/#40709 (all still open as of 2026-09-09).
# **Runner change**: MRV2 is now the default for all models (#53183); on ROCm
# the only MRV1 defaults are DeepseekV32/DeepseekV4 (ROCM_DEFAULT_MRV1_
# ARCHITECTURES in vllm/config/vllm.py), so all three profiles now run MRV2 —
# qwen3.8-27b pins VLLM_USE_V2_MODEL_RUNNER=1 explicitly (a54a74ca13, also
# the #54498 mitigation); qwen3.6-27b and qwen3.6-35b-a3b migrated
# implicitly with the bump and are confirmed fine on runner 2 (user-verified
# 2026-09-09).
gh release list -R vllm-project/vllm --limit 8

# AITER — current pin AITER_REF=v0.1.20.post1 (TheRock 10.0 hipcub fix;
# bumped with the ROCm 10.0 migration 2026-09-09). Upstream: v0.1.21
# (2026-09-02, bi-weekly), v0.1.21.post1 (2026-09-03: #5222 MLA
# _fold_seqlen_indptr cudagraph-safe fix — N/A, no MLA), v0.1.21.post2
# (2026-09-09) = v0.1.21 + 30 FlyDSL/CI/gfx950/gfx1250/Kimi-K3/MiniMax
# commits. +210 commits vs our pin, dominated by gfx1250/gfx950/FlyDSL;
# includes the unified-attention refactor #5088 (merged 2026-09-02:
# _UAParams, backend param, per-kernel subwrappers,
# get_unified_attention_config) and #4761 (UA prefill/decode perf,
# 2026-09-09) — both touch the code our three local unified-attention
# patches modify, so a bump means rebase + re-running
# tools/tune_ua_config.py. #4329 (bf16-KV LDS cap) still NOT fixed upstream
# (no LDS commits in the range); no gfx1201/Qwen patches in the release.
# Only ROCm-10-relevant commit is #4853 (topk_per_row hipcub::Traits fix) —
# inert, our pin verified working on ROCm 10.0. No bump (re-checked
# 2026-09-09: v0.1.21.post2 latest, unchanged).
gh release list -R ROCm/aiter --limit 8

# Flash Attention — pinned to a commit, so compare HEAD to FLASH_ATTN_REF
# (2026-09-09: HEAD a369df7 unchanged since 2026-09-06, 2 commits past our
# pin, both flash_attn/cute/ SM100-Blackwell CuTe fixes — N/A on
# ROCm/gfx1201, no bump)
git ls-remote https://github.com/ROCm/flash-attention.git HEAD

# ROCm base image — current ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.0.0-full
# (migrated 7.14.0 -> 10.0.0 on 2026-09-09 per the official vLLM-on-ROCm
# guide: ROCm 10.0 + PyTorch 2.13 via stable.repo.amd.com/rocm/whl-next).
# 10.0.x releases via TheRock (github.com/ROCm/TheRock releases);
# 10.0.0-full is the only 10.0 tag so far (re-checked 2026-09-09). The
# 7.14 line (7.14.1 point release) is no longer relevant to this stack.
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-ubuntu-24.04/tags?page_size=100&name=10.0" | jq -r '.results[].name' | sort -V

# Froggeric chat template — current pin is the first line of chat-templates/qwen.jinja
# (template_version = "qwen3.8-froggeric-v22.5", upstream unchanged as of
# 2026-09-09). Compare against upstream main:
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
# Re-check watchlist status (open/closed/resolved) + any new labels:
for n in 35288 47087 48375 52872 47602 51250 52520 45238 51562 51812 51837 40707 52527 52789 48815 52817 52959 51198 49125 53479 51571 54039 54360 54498 53504 53488 51599 54076 53798 50409 54163 55600 55533; do
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
- **Models**: Qwen3.6-27B (dense, MTP4), Qwen3.6-35B-A3B (MoE, MTP off),
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
  - `#35288` MTP concurrency corruption (still mitigated by `max-num-seqs 2`).
    2026-09-01: new cross-reference names `#51571` (async MTP align
    accepted-count race) as the precise root cause — the corruption requires
    async scheduling, which we disable, so the mitigation stands; track the
    fix in `#51599`
  - `#47087` MTP token loops on Qwen3-MoE (resolved by #51113, in v0.27.1 —
    **re-test PASSED 2026-08-24**: MTP4 on 35B-A3B is clean on v0.28.0rc2 —
    coherence PASSED + manual 512-token gen varied, ~2x decode win (tg32
    194.9 vs 87.8 MTP-off); MTP4 re-enabled by default on 35B-A3B)
  - `#51812` Qwen GDN gate/spec-token alignment — **resolved**: merged
    upstream 2026-08-11 (`5af7c8d`), present in v0.28.0rc2; local patch
    dropped 2026-08-21
  - `#51837` ROCm KV-first attention blocks sharing pages with Mamba —
    **resolved**: merged upstream 2026-08-11 (`3e372c5`), present in
    v0.28.0rc2; local patch dropped 2026-08-21. Inert on this stack (AITER
    unified attn is blocks-first, `block_dim == 0`); only matters if a
    KV-first backend is ever selected
  - `#48375` MambaManager ignores `drop_eagle_block` (MTP + prefix caching
    corrupts hybrid recurrent state, #43559/#50188) — **carried as a local
    patch** (upstream PR still open; re-checked 2026-09-09 still `OPEN`,
    updated 2026-09-08T21:31Z; local patch re-verified clean on v0.29.0 —
    `MambaManager.find_longest_cache_hit` target hunk unchanged)
   - `#52872` GDN/hybrid prefill peak under-predicted; `--max-num-batched-tokens`
     also sizes the CUDA-graph pool. **2026-09-03**: qwen3.8-27b now pins
     `VLLM_MAX_BATCHED_TOKENS=2048` (concurrent-ITL A/B — a 100K+ prefill
     stalled the co-decoder 150–200x at the 8192 default; 2048 → ~1 s ITL,
     flat big-prompt TTFT, −3.4% pp2048). Smaller pool/peak; re-check the
     prefill-peak headroom and re-run
     `benchmarks/conc_itl_probe.py` on any bump that changes chunk/
     graph-pool sizing. See
     `benchmarks/2026-09-03_qwen3.8-27b_concurrent_itl.md`.
  - `#47602` MTP draft acceptance decays with context length (Qwen3.6-27B)
  - `#51250` prefix caching is a silent no-op on GDN hybrid (same family as
    `#45238`)
  - `#51198` newer restatement (2026-08-21) of the `#45238`/`#51250` hybrid
    prefix-cache 0%-hit no-op — confirms the family is still open (monitor)
  - `#49125` stale partial prefix-cache hash resurrected after full-block
    promotion (pure-Python `BlockPool` bug in the fine-grained/partial
    prefix-caching path `#45939`/`#46384`). Only reachable once `#45238` is
    fixed and prefix caching actually hits — monitor post-fix
  - `#52520` align-mode admission livelock near KV-pool ceiling (open)
  - `#45238` hybrid prefix caching drops to 0% in align mode (open) — the
    binding constraint on this stack. Root cause:
    `BlockPool.cache_full_blocks` skips Mamba align-mode null blocks, so only
    ~1 checkpoint hash per request is registered and a missing Mamba
    checkpoint vetoes every attention-group hit. Live geometry:
    `block_size=1600` on the fp8-KV profile (832 on bf16), so
    incremental multi-turn prefixes never hit — measured **0% on the 30-turn
    qwen3.8-27b probe (re-confirmed 2026-08-23)**. Note the cumulative
    `vllm:prefix_cache_hits_total` is non-zero: caching *does* hit on
    repeated-identical-prompt workloads — the failure is specific to the
    incremental shared-prefix pattern, not a global no-op. Fixes in flight:
    `#52527` (metrics), `#48815` (MTP align retention), **`#53479`
    (2026-08-25, the leading candidate — retention-aware boundary
    materialization + removal of the speculative one-block back-off; makes the
     store side consistent with the `#52216` retention-0 default; open, not
     merged — 2026-09-06: still CONFLICTING + REVIEW_REQUIRED (head b99d152),
     no progress (re-checked 2026-09-06);
    interacts with our `--prefix-cache-retention-interval` pin so
    re-validate after any bump that lands it)**, and a 2026-08-27 adaptive
    single-checkpoint prototype (demand-driven, not yet a PR; positioned as an
     alternative/complement to `#53479`); `#52789` (internal
    prefill checkpoints) merged 2026-08-22 — **verified present in
    v0.28.1rc0** (merge `9eb9d9d` is an ancestor of the tag, checked
    2026-08-29); Kimi-K3/FlashKDA-specific TTFT win, not a fix for the 0%-hit
    geometry. **When a real fix merges**: prefer
    the version bump; carry a local patch only if no available release
    contains it. 2026-08-24 data points (independent 27B-scale repro on sm80,
    Qwen3.8-27B):
    a cache miss needs ~3 sightings, not 2 (Marconi only checkpoints after a
    prefix is known-common), and turn-2 at 40K was *slower than cold*
    (insertion/eviction churn in the block-aligned chunk-split path);
    `--prefix-match-unit 400` (in rc2) does not fix the turn-2 miss but halves
    eventual-hit TTFT. Related trigger `#52897` (0 hits with
    `--scheduling-policy priority`) is N/A here — we don't use priority.
    2026-08-27 `#53749`: two more hybrids (Nemotron-3.5-L 30B, Ling-3.0-flash
    int4) show exactly 0 hits below one attention block (auto-raised to
    2096/1920); same family — no action, reinforces the checkpoint fix is the
    binding constraint.
  - `#53504` (2026-08-24) with MTP, the **first** repeat of an identical
    prompt misses the prefix cache entirely on Qwen3.8-27B hybrid GDN align —
    this stack's exact geometry (TP2, 3 Mamba groups + 1 attn, 256K); the
    second repeat hits. Cause: the EAGLE-adjusted reusable boundary is not a
    boundary the default *sparse* Mamba retention keeps — a side effect of
    `#52216`, which promotes `prefix_cache_retention_interval` to a CLI arg
    and flips the default from `None` (dense) to `0` (semantic checkpoints
    only). **Mitigated on v0.28.1rc0 (2026-08-28)**: `#52216` is in the rc, so
    all three profiles pin `--prefix-cache-retention-interval <block_size>`
    via `PREFIX_CACHE_RETENTION_INTERVAL` in their env files (1600 on the fp8
    qwen3.8-27b profile, 832 on bf16 KV, 2112 on 35B-A3B). The compose var is
     deliberately NOT named `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` — that
     env var is vLLM-managed, deprecated in v0.28.1, removed in v0.29 (the
     CLI flag remains). Validate with the prefix-cache probe on any bump.
     **Root-cause fix landed (2026-09-09)**: `#55760` + `#55861` merged
     2026-09-08 and are in the v0.29.0 final — we are pinned to it
     (8c07ffea3d) and the `PREFIX_CACHE_RETENTION_INTERVAL` pins are dropped
     from all env files (unset now means dense for hybrid+MTP). Re-run the
     prefix-cache probe after any future bump to confirm the first-repeat
     still hits.
   - `#55766` (2026-09-07, open, new): Qwen3.8/3.5 hybrid GDN + mamba align +
     prefix caching — a prefill that ends **4–10 tokens past a block boundary**
     writes a bad Mamba/GDN checkpoint; a later block-aligned prefix-cache hit
     restores it → **NaN logits** from step 1 (token-0 `"!"` spam to
     max_tokens; `corrupted_requests_total` increments; identical retries fail
     until the cache turns over). **This stack's exact model + cache mode +
      TP2 + fp8** (repro at block 816 on v0.28.0; ours is 1600). Silent-
     corruption family (cf. `#53912`, `#55291`, `#39273`). **Currently
     masked**: the carrier is a prefix-cache *restore*, and our incremental
     multi-turn pattern is the `#45238` 0%-hit no-op, so the bad checkpoint
     isn't restored there — but repeated-identical prompts DO hit, and it
      becomes live the moment `#45238` is fixed. No in-repo mitigation; the
      client-side workaround is `cache_salt` (or pad the prompt **start**) when
      the prior prompt length mod block_size ∈ {4,6,8,10}. **Monitor**; run a
      probe if we ever see NaN / `"!"`-spam / empty replies. **2026-09-09**:
      upstream dev actively investigating (version-controlled SM89 repro in
      progress; no PR yet). **Exposure increased on v0.29.0**: the
      dense-retention default (`#55760`/`#55861`) makes the *first*
      identical-prompt repeat hit the cache, widening the window where a bad
      checkpoint (prior prompt length mod 1600 ∈ {4,6,8,10}) can be restored —
      run a targeted probe (repeated prompts at those lengths).
      **2026-09-10 probe** (`benchmarks/nan_checkpoint_probe.py` +
      `benchmarks/2026-09-10_qwen3.8-27b_55766_nan_probe.md`): **CLEAN on
      v0.29.0 + MTP3 — masked, not disproven.** Measured hit geometry backs
      off **2 blocks** from the last aligned boundary (12800-token hit on a
      16006-token prompt = 10×1600+6; consistent across r=6/8/20): the
      EAGLE/MTP last-block drop (`use_eagle_block_drop`, #53388 family) plus
      the speculative one-block back-off (#53479, unmerged) both stand, and
      the bad checkpoint sits AT the last aligned boundary → it is
      **unrestorable under MTP3**, so the dense-retention "widened window"
      does not apply to this carrier while MTP3 + 2-block back-off stand.
      Upstream repro used ngram (no EAGLE drop → its hit reached the last
      aligned boundary); the "is ngram required?" question is moot here.
      Note: the live profile is **fp8 KV / block 1600**. **Re-probe** if spec decode changes
       (MTP off → hit reaches the last aligned boundary → carrier live), on a
       bump landing #53479/EAGLE-drop changes, or on any NaN / `"!"`-spam /
       empty reply in the field.
   - `#53041` RFC: tiered SWA/Mamba checkpointing (HBM tail + periodic store)
     + recompute backfill for divergent hybrid prefix hits (same family as
     `#52959`/`#52789`; monitor)
   - `#55697` (2026-09-09, RFC, open): application-directed prefix
     checkpoints for Mamba/hybrid prefix caching — same family as `#53041`/
     `#52959`; monitor-only
  - `#53488` `prompt_logprobs` silently corrupted under MTP + chunked prefill
    (Qwen3.5-family, two builds) — we don't request prompt_logprobs; monitor
  - `#50729` Mamba state-copy overlap race in `vllm/v1/worker/mamba_utils.py`
    (same-block conv/SSM shift copies were memmove-unsafe) — **merged
    2026-08-17 and present in v0.28.1rc0** (verified `a02cfcc` is an ancestor
    of the tag). `#53077` GDN metadata reset of the spec-decode count on an
    empty draft schedule — **merged 2026-08-20 and present in v0.28.1rc0**
    (verified `6df7adc`). Both former "main-only, ride the next bump" fixes
    (checked 2026-08-24) are now in the current pin — no bump needed to gain
    them.
  - `#52817` RFC: hybrid SSM + SpecDec + APC re-runs the last full block on a
    prefix hit (1600 tokens here on the fp8-KV profile; 832 on bf16),
    bounding the prefix-cache win for MTP even after `#45238` is fixed. Monitor
    for a merged implementation.
  - `#51562` GDN metadata misclassifies stateless first chunk (open)
  - `#52959` RFC: internal state checkpoints for Mamba align mode (same
    family as `#52789`; in flight, not merged)
  - `#40707` hybrid Mamba scheduling deadlock with 2+ large images in one
    prompt (align block-split collapses to 0 → request hangs forever, engine
    never recovers). **Previously mitigated 2026-08-28** via
    `--limit-mm-per-prompt image: 1` (2+-image trigger unreachable, multi-image
    rejected with 400); **cap re-raised to 99 on 2026-09-08 at user request** —
    deadlock risk re-exposed. Fix PR `#40709` is **not merged** (absent from
    v0.28.0rc2) — monitor it.
  - `#51571` async MTP align accepted-count race (open): async scheduling +
    MTP + hybrid GDN + `mamba-cache-mode align` → accepted-token D2H counts
    gathered from a mutated `InputBatch` after `condense()` (repeated/dropped/
    garbled tokens). **Relevant, was mislabeled N/A.** On v0.28.0 `async
    scheduling resolves to ON by default when the spec method is MTP` (MTP is
    in `EagleModelTypes`; verified in `vllm/config/vllm.py` at the tag) — the
    "async is auto-disabled on MTP" note was wrong. Mitigation:
    `compose.yaml` now passes `--no-async-scheduling` for all spec-decode
    profiles (tracks `VLLM_SPEC_DECODE`); re-check upstream before removing.
     **Fix in flight (2026-09-01)**: PR `#51599` (open, decouples the async
     Mamba-align D2H accepted-count copy from `InputBatch` row shifts, closes
     #51571; also cited as the root cause of `#35288` — re-checked 2026-09-09
     still `OPEN`, updated 2026-09-08T23:49Z, retitled `[Bugfix][MRv1]`,
     i.e. retargeted at the V1 runner; all our profiles now run MRV2, which
     shrinks the exposed surface, but `--no-async-scheduling` stays) — if it
     lands in a release we adopt, re-test before dropping
     `--no-async-scheduling`.
  - `#54039` (2026-08-27, question): vLLM's own ROCm CI disables async+MTP
    (#32275, unroot-caused shm-broadcast hang) while the default still
    enables that combination; asks for a default-resolution fix or at least a
    warning. Same combination we now disable via `--no-async-scheduling`;
    monitor for a merged default change.
  - `#54360` (2026-08-29, open) on **nightly** (`v0.28.1rc1.dev43`, main past
     our rc0 pin): any spec decode (MTP or dflash) drives prefix-cache hits to
     **0** on Qwen3.8-27B hybrid GDN align — same model family and align-mode
     path as `#45238`/`#53504`. Comment data point: on 0.27.1-era builds spec
     decode loses exactly one block of reachable prefix (4→3, hit rate
     69.4%→42.5%), consistent with the EAGLE-boundary/one-block-back-off
     family, and nightly regresses further to zero. Not in our pin (main was
     100 commits past v0.28.1rc0, unreleased, checked 2026-08-29). **Monitor**:
      forward-looking regression signal for the eventual v0.28.1 final —
      re-run the prefix-cache probe if a bump lands that includes it.
   - `#54498` (2026-08-27, open, checked 2026-09-03): V1 EAGLE/MTP drafter
    feeds the M-RoPE **temporal** dim (`positions[0]`) to the KV-slot
    computation on `SupportsMRoPE` targets — on any prompt with an image the
    temporal coord lags the absolute token index, so each draft step writes
    draft K/V into a **prompt** slot (overwriting real prompt K/V) while
    attention reads the full span: acceptance drops and the error compounds
     with K. **Affects this stack's model**: Qwen3.8-27B is M-RoPE (local
     config: `mrope_section [11,11,10]`, `mrope_interleaved`), we run MTP3 and
     we serve images (capped 1). Buggy line verified in v0.28.1rc0
     (`llm_base_proposer.py:787`). Text-only prompts are unaffected (dims ==
     absolute index). Upstream measurements: Qwen3.8-27B K=6 ≈ -5.7% mean
     acceptance vs the V2 runner; the gap grows with K (K=3 ≈ -0.3%, K=15
     -24.8% on a VL-derived 27B). The prompt-K/V overwrite implies an
     unmeasured output-quality risk on image+MTP requests (upstream measured
     acceptance only). **Mitigated 2026-09-09**: qwen3.8-27b switched to the
     V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`, a54a74ca13; A/B on
     ROCm10/v0.29/torch2.13: pp +1.8%, tg mix, coherence PASSED) — the bug is
     in the V1 proposer, so no profile on this stack runs it (v0.29.0
     defaults every non-DeepseekV32/V4 architecture to MRV2 on ROCm; the
     qwen3.6 profiles are on V2 too — user-verified fine 2026-09-09). Fix PR
     `#54519` was **closed unmerged 2026-09-01** (superseded), leaving
     `#54716` as the sole fix (open, in no release — re-checked 2026-09-09,
     still `OPEN`, updated 2026-09-09T19:52Z).
    Open review defect in `#54716` (flagged 2026-09-01 by the superseded
    PR's author; no response as of 2026-09-03): its `step3p5.py` re-derives
    the max-len `exceeds` condition *after* `seq_lens` has advanced in
    place, so at the boundary the draft token is written to slot 0 of the
     first block (live prompt KV) — same corruption class, relocated to the
     overflow path; its tests also skip on CPU CI.
     **Monitor** (no longer blocking: all profiles run MRV2, which has no
     V1 proposer — backport/bump only matters if a profile ever returns to
     the V1 runner). (Companion `#54555`/`#54621` xDRoPE positions-buffer —
     N/A, we're M-RoPE.)
     2026-09-03: forcing the V2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`, exists
     in v0.28.1rc0; MTP + Mamba align pre-copy are V2-supported) was A/B'd as
     an alternative to this V1-proposer bug — full correctness battery passed,
     pp2048 +3.6–4% (consistent), but MTP acceptance parity (2.7–3.4) and
     decode flat, so the #54498 acceptance hypothesis did not materialize
     here; not adopted at the time (non-platform-default path; V1 is the
     validated baseline). **Superseded 2026-09-09**: v0.29.0 made MRV2 the
     platform default and the qwen3.8-27b profile adopted it (a54a74ca13).
     Revisit conditions in `benchmarks/2026-09-03_qwen3.8-27b_v1_vs_v2.md`.
  - `#54928` (2026-09-02, open, new): **Qwen3.8-27B** (our exact target)
    with DFlash2 + thinking is not greedy-equivalent to target-only —
    diverges at generated token 30, reproduced at K=1 and
    `--enforce-eager` (rules out draft-depth and CUDA-graph artifacts),
    text-only (no M-RoPE slot involvement), `enable_thinking:false`
    control matches; suspected spec-decode verify / hybrid GDN
    state-update path. No direct impact: we rejected DFlash2 on
    qwen3.8-27b (see archive/DEADENDS.md) and run MTP3; the DFlash2
    implementation itself is an in-flight PR (#52816), not in any release.
    **Monitor**: if the root cause is confirmed in the shared verify/
    GDN-state path, MTP3 + thinking is implicated too — run an MTP
     greedy-equivalence probe (target-only vs MTP3, temperature=0,
     thinking prompt) at that point.
   - `#55894` (2026-09-08, open, new): hybrid Mamba + MTP silently corrupts
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
     entry above (re-checked 2026-09-09 still `OPEN`, updated 2026-09-09;
     retitled `[Bugfix][MRv1]`).
  - `#54076` (2026-09-01, PR, open): `_mamba_block_aligned_split` must chunk
    on the **Mamba group's** block size, not `cache_config.block_size` (the
    min over all groups) — otherwise mandatory chunk ends land on a grid the
    worker can never materialize a Mamba state at. Repro is a Qwen3.8-27B
    hybrid + spec drafter with mismatched target/drafter attention blocks
      (1648/816); our MTP drafter group can create the same geometry.
      re-checked 2026-09-09: still open (new push 2026-09-09T18:30Z).
      Monitor for a merge.
  - `#53798` (2026-09-01, PR, open): align-mode `add_request` seeds the
    running-state block column by the scheduler block size instead of the
    (page-unification-scaled) Mamba block size, so a request admitted with
    `num_computed_tokens > 0` — explicitly under
     `--prefix-cache-retention-interval`, which we pin — points its precopy
    source into a neighbour's row (silent wrong-state read) or past the
    table (IMA in `precopy_mamba_align_fused_kernel`). re-checked
    2026-09-09: still open (2026-09-09T20:43Z commit is a merge of main
    into the fix branch — core fix unchanged); **carried as a local patch**
    (patches/vllm/53798-mamba-align-resume-seed.patch, rebased to v0.29.0
    on 2026-09-09 — v0.29.0's dense-retention default keeps the trigger
    reachable, prefix-cache resumes now happen on first repeats too).
    Drop when a release contains the fix. Sibling
    `#55600` (2026-09-06, open) shows the same line crashes with a
    small-block drafter (DFlash2 block 64/1024 vs `mamba_block_size` 7168) —
    still `OPEN`, no PR yet; reinforces the fix is incomplete on `main`.
  - `#50409` (2026-08-31, PR, open): when the prompt length is an exact
    multiple of the block size, align prefill runs as one chunk and the only
    cached Mamba state sits at `num_tokens`, which `get_computed_blocks`
    caps below — the Mamba group then reports a 0-token hit and the
    reconciled hybrid hit is 0. Adds the replay boundary as a mandatory
    chunk stop. Monitor for a merge.
  - `#54163` (2026-09-01, PR, open): removes the one-mamba-block back-off
    for DFlash/DSpark drafters (they never write target blocks, so the
     `#53388` `use_eagle_block_drop()` stand-in over-backs them). N/A for
     MTP (MTP *does* pollute the last target block, so its back-off stays) —
     2026-09-03: new commits, still open. Monitor as `#52817`-family signal
     only.
   - `#55196` (2026-09-03, RFC, open): fp8 KV yields only 1.00x–1.84x (not
     2x) cache capacity on Mamba/GDN hybrids — the never-quantized bf16
     Mamba page pins the unified hybrid block (`#37121`/`#40696` family), so
     `--kv-cache-dtype fp8` barely moves `num_blocks`. Independent second
     reason to keep **bf16 KV the default** (alongside the `#52793`
     calibration gap). Proposed levers (quantize Mamba state, decouple
     per-group page size) unimplemented anywhere in vLLM. Monitor; no action.
   - `#55600` (2026-09-06, open): hybrid mamba prefix-cache hit reads out of
     bounds — `add_request` seeds the state index with `cache_config.block_size`
     after it was lowered to the min prefix-cacheable group (small-block drafter
     64/1024 vs `mamba_block_size` 7168) → `precopy_mamba_align_fused_kernel`
     IMA (Xid 31) or silent wrong-state read. GLM-5.3-Flash/DFlash2 repro on
      `main`; same `mamba_hybrid.py` line as `#53798` — DFlash2 variant of that
      bug. N/A for this stack (MTP drafter not small-block; we are `block_size`
      1600) but sibling proof `#53798` fix still incomplete — monitor;
     no PR yet.
   - `#55533` (2026-09-06, open): Hybrid GDN (Qwen3.5/3.8 27B-class) + MTP
     scheduler caps at ~3 concurrent sequences at batch ≥ 4 — acceptance/
     throughput collapse (8-wide batch runs `[2,2,2]` only; `bs ≤ 3` healthy).
     Root: mamba cache budget shared between target states + MTP draft slots in
     scheduler accounting (scales with GDN layer count). **Relevant** — we have
     the `#35288` `max-num-seqs 2` cap so not hit today, but blocks any future
     cap raise. WIP fix `PR #55617` (2026-09-06). Monitor before raising
     `max-num-seqs`.
 
 Issues known **not** to apply (checked; re-check only if the stack changes):
NVIDIA-only (#52475, #52583 VL), non-Qwen models (#52833/#48568 GLM, #51530
DeepSeek, #53387 Qwen3.5-family compressed-tensors WNA16 MTP drafter load
crash — we use FP8, not WNA16), or paths not
reached here (PP ranks #51752, DP attention #51957, KV connectors #51805/
#51766/#40017/#53505/#53514, GPTQ #51971, gfx950 MLA #52312). #52897
(align-mode 0 hits with `--scheduling-policy priority` — variant of #45238;
we don't use priority). #52539/#53462 (Qwen GDN fused-MTP decode kernel
head-ratio support + SM110a crash — the kernel only builds for CUDA >= 13.0
sm80-120, absent from ROCm builds; our v/k ratio 48/16=3 is now in the
supported set but N/A on gfx1201. Listed again in the v0.29.0 release
notes as "fused GDN MTP for all Qwen head ratios" — still CUDA-kernel work,
N/A on ROCm). #50264 (RDNA hybrid-Mamba decode collapse
via Triton paged-attention fallback — head_dim 256/block_size != 16 misses
the custom-paged gate; we run AITER unified attention and never reach that
path; the #45916 fix was verified on gfx1201 but doesn't apply here).
#52688/#53397 (multi-layer MTP
spec_step_idx — all K draft steps re-execute layers[0]; both 27B models have
`mtp_num_hidden_layers=1`, so layer-0-only is correct and N/A). #53136 (ROCm
all-reduce 8–16 MiB dead zone → RCCL generic-kernel launch fault; requires
`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` and the fault was gfx942 TP=8-specific —
we never set QUICK_REDUCE and gfx1201 TP=2 is stable; re-check only if that
changes). #52793 (fp8 KV scale-1.0 on hybrids): no coherence failures
 (d258K probe passed), but the stock FP8 checkpoints ship no k/v/q scales, so
 fp8 KV serves at scale 1.0, which is genuinely miscalibrated (deep-layer V
 amax ~132 vs the ~1-24 range scale 1.0 assumes; calibrated vs scale-1.0
 outputs diverge ~20-27%). The 2026-08-22 quality A/B
 (`benchmarks/2026-08-22_kv_calibration_quality_ab.md`) found calibrated and
 scale-1.0 fp8 KV **indistinguishable** on PPL and long-context recall — so
 calibration is a correctness fix, not a measured quality win, and **bf16 KV
 is the default**. When fp8 KV is re-enabled, the default profiles still point
 `VLLM_MODEL` at the calibrated local copy that `just up` builds via
 `ensure-kvscales` (recalibrate with `just clear-kvscales`), whose coverage
 includes the **MTP prediction-head layer(s)** (`mtp.layers.*`, which cache
 fp8 KV separately and were silently at scale 1.0). The residual `prob_scale
 1.0` warning is the fp8-attention softmax-probability scale, separate from
 the KV cache scales, and caused no coherence issues at 258K. Re-visit
 calibration whenever KV precision matters (long-context recall). Also
 checked 2026-08-21: #53180 (turboquant_k8v4 + MTP degeneration on hybrid
GDN — NVIDIA Ada/AWQ, we use fp8 KV; same silent-corruption family, so
re-check if turboquant KV is ever tried), #52480 (qwen3_5_mtp TP≥2 load
failure — NVFP4/ModelOpt checkpoints on NVIDIA; our FP8 MTP head loads fine
at TP=2), #53142 (align pre-copy IMA on prefix-cache resume — requires
explicit `--block-size`, which we never pass; #54199 was retracted 2026-08-29
as a duplicate of this one — its "equal attn/mamba block sizes" premise was
wrong). #53387 (MTP drafter load crash
on compressed-tensors WNA16 checkpoints — unquantized `mtp.fc` vs packed
layout; we use FP8, not WNA16). #53887 (MTP drafter allocates a second full
vocab embedding, OOMing a 27B INT4 on a 24GB card — NVIDIA/INT4; our MTP3
loads fine on 2×32GB). #53983/#53982 (ROCm spec-decode attention-metadata
allowlist + `_compute_slot_mapping_kernel` OOB — both concern in-flight model
PRs, Qwen3.8-Flash-Next `QSAForwardMetadata` and GLM-5.3-Flash `KpoolTailSpec`,
one-block-per-request side caches not on main; our AITER unified-attention
metadata is already allowlisted and MTP works, so N/A unless a new backend is
added). #53066 (v1 detokenizer evaluates client `stop` strings against the
whole output stream, so a stop that a think-in-prompt CoT restates — Qwen3
family — truncates mid-think and the reasoning parser returns null; only
triggers when a client
actually sends `stop`; our clients don't — monitor). #40980 (R9700 TP2 deadlock — stale: v0.19-era, 16GB cards,
TRITON_ATTN + enforce-eager; AMD confirmed R9700 TP2 working on v0.25.1; our
TP2 stack is serving). #49851 (multimodal load failure on gfx1201 in the
`vit_torch_sdpa_wrapper` — v0.25.1/ROCm 7.15-specific, AMD states it loads on
v0.26.0+ with `--mm-encoder-tp-mode data`; we serve images on v0.28.0, stale
for this stack). #47194 (Qwen3.6/3.8 hybrid + prefix caching + MTP3 →
tool-call/needle-recall corruption on the cache-hit path — this stack's exact
config family): **resolved in our version** — the degradation is reported
 fixed in v0.28.0 by `#51113` (verified `c56f169` is in v0.28.1rc0; an
independent 3-arm A/B/C on a Qwen3.8-27B hybrid GDN/align/fp8-KV/TP2 setup
 shows no degradation with MTP on). We're on v0.29.0, so no action; the
 residual warm-rollback TTFT tax is the `#53479` performance item, not a
 correctness one. #54106 (KV cache group splitting assumes an n:1
 attention-type ratio — our 48 GDN : 16 full is 3:1, fine). #52682
 (Qwen3.8-27B-FP8 CUDA-graph capture hang at startup — NVIDIA Ampere
  A5000-specific). #54080 (TreeWY tree-spec-decode RFC for hybrid GDN) and
  #53786 (fine-grained prefix hits for sliding-window groups) — RFC/feature in
  the `#45238` family, monitor-only. Checked 2026-09-01: #54690 (draft-only
  fp8 KV dtype crashes hybrid GDN startup — NVIDIA/FlashInfer paths, we are
  all-bf16 KV), #54761 (DCP + non-FP8 KV unreachable on ROCm — no DCP here),
  #41862 (EP deadlock on hybrid GDN, Qwen3.5 — we run TP2 without expert
  parallelism), #54504 (nemotron_h prefix-cache no-op / CPU-backend crash),
  #54392 (PD-admitted Mamba spec-pad truncation — no P/D disaggregation),
  #54458 (GLM-5.3 page-alignment block inflation) and #54831 (GLM-5.3 DSA-
  indexer KV offload), #54728 (gfx1030 RDNA2) and #54438 (gfx1100 kernel
  ranking), #54698 (Qwen3.8-Flash-Next NVIDIA torch.compile RFC), #54547
   (Quark MXFP4 multimodal naming), #54281 (DeepEP v2 hybrid-mode flag),
   #53334 (sm121 turboquant KV observations — turboquant N/A per #53180).
   Checked 2026-09-05: #55291 (Qwen3.6-27B-FP8 "eventually collapses into
   repeated ! tokens, sticking across all subsequent requests" — v0.21.0 +
   NVIDIA L20, no spec decode, no root cause yet, repro on ≥0.28.0 requested;
   N/A for this stack, but tracked model + sticky state corruption: glance if
   a modern repro appears), #55322/#55323 (MTP `n_predict` auto-detection
   from multimodal-wrapper config / `num_speculative_tokens` auto-default —
   only triggers when `num_speculative_tokens` is unset; we pass it
   explicitly in the `--speculative-config` JSON), #55357 (Qwen3.8-
   Flash-Next/qwen4_exp MTP 0%-acceptance + thinking-block repetition
   collapse — different model/arch, SM120; #54928-family signal only),
   #55280 (GLM-5.3-Flash kpool 32-page split IMA on ROCm — kpool-indexer
   path we don't have, gfx942), #55416 (`--numa-bind` silent no-op in ROCm
   images — we don't use numa-bind), #55425 (XPU), #54926 (Gemma + NIXL PD
    disagg), #54906 (thinking_token_budget ignored by V2 runner — we now
    run MRV2 so the runner half is live territory, but we never send
    `thinking_token_budget`; protocol field only),
   #54165 (align-mode cache-hit restore under spec decode with a KV
   connector — no connectors here).
    Checked 2026-09-06: #55600 (hybrid mamba OOB read above — N/A for MTP but
    sibling to carried #53798) and #55533 (hybrid GDN+MTP 3-seq cap at batch ≥4
    — relevant, capped today by `max-num-seqs 2`; WIP #55617). Checked
    2026-09-08: #55766 (Qwen3.8/3.5 hybrid GDN align prefix-hit → NaN logits;
    our exact model — see watchlist) and v0.29.0rc5/rc6 (#55760/#55861 dense
    retention default, the #53504-family fix). Checked 2026-09-09: v0.29.0
    final released 08:54Z — we were already pinned to it (bumped this morning
    with ROCm 10.0/torch 2.13, 8c07ffea3d; 53877 patch dropped, 53798
    rebased, retention pins dropped, qwen3.8-27b on MRV2 a54a74ca13,
    qwen3.6 profiles confirmed fine on MRV2). New: #56021 (RDNA
    `VLLM_ROCM_USE_AITER=1` force-selects unified-attn at priority 0,
    ignoring `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`; stock kernel exceeds
    RDNA's LDS limit 65792>65536 → engine dies on split-KV — N/A: we pin
    `--attention-backend ROCM_AITER_UNIFIED_ATTN` explicitly and our local
    bf16-KV LDS-cap patch keeps us under the limit; confirms that patch is
    load-bearing), #55894 (hybrid Mamba+MTP reorder-threshold corruption —
    structurally mitigated by the pinned drafter backend; see watchlist),
    #55697 (application-directed prefix checkpoints RFC), #55916 (RDNA4
    FlyDSL all-reduce RFC — monitor), #55951 (PP>1 + spec decode — no PP),
    #56077 (ngram spec decode corrupts qwen3_coder tool args — we run MTP),
    #56088/#55922 (Qwen3.8-Flash-Next), #55775 (NVIDIA/FlashInfer/NVFP4),
    #55425 (XPU). #53798: today's commit = merge of main into the fix branch,
    core fix unchanged. #54076: new push today. #55766: actively
    investigated upstream (SM89 repro), no PR. AITER v0.1.21.post2 (09-09) =
    v0.1.21 + 30 FlyDSL/CI/gfx950/1250 commits — no bump. flash-attn HEAD
    a369df7 unchanged; template v22.5 unchanged. 2026-09-10: update check —
    all pins still current (v0.29.0 latest, AITER no-bump stands,
    flash-attn/ROCm-image/template unchanged), watchlist unchanged,
    #54716 force-pushed (e829a9c) with the flagged exceeds-condition defect
    addressed + boundary test (still open, monitor-only on MRV2). #55766
    probe run: CLEAN — masked by the MTP 2-block hit back-off (bad
    checkpoint unrestorable); see watchlist entry +
    benchmarks/2026-09-10_qwen3.8-27b_55766_nan_probe.md. Live geometry
     confirmed fp8 KV / block 1600.

### 4. Local patches vs upstream

`patches/vllm/*.patch` and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time. Before bumping any pin:

- The aiter patches (version-locked to `AITER_REF` v0.1.20.post1) are **RDNA4-local
  work**, not upstream cherry-picks: `unified-attention-bf16-kv.patch`
  (bf16-KV LDS caps, the fix for upstream ROCm/aiter#4329 / vllm#48723, still
  open), `unified-attention-gfx1201-tune.patch` (per-arch gfx1201 tuning:
  attn_warps 4 in 3D decode ~1.4-1.9x, num_warps 8 in 2D large-prefill ~7%),
  and `allowed-archs-gfx1201.patch` (build-path arch acceptance). When a newer
  `AITER_REF` merges #4329, the bf16-KV cap should be **dropped** (upstreamed)
  but re-verify the tuning still wins — re-run `tools/tune_ua_config.py` (with
  `just down` first) and re-check the LDS guard. See
  `benchmarks/2026-08-25_gfx1201_ua_tuning.md`.
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
