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
# vLLM — current pin VLLM_REF=v0.28.1rc0 (still the newest tag as of
# 2026-09-01; v0.28.1 final unreleased)
gh release list -R vllm-project/vllm --limit 8

# AITER — current pin AITER_REF=v0.1.20 (latest stable; v0.1.21.dev0 is a
# gfx1250 preview for a different arch, 2026-09-01)
gh release list -R ROCm/aiter --limit 8

# Flash Attention — pinned to a commit, so compare HEAD to FLASH_ATTN_REF
# (2026-09-01: HEAD a369df7 is 2 commits past our pin, both flash_attn/cute/
# SM100-Blackwell CuTe fixes — N/A on ROCm/gfx1201, no bump)
git ls-remote https://github.com/ROCm/flash-attention.git HEAD

# ROCm base image — current ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
# 7.14.x releases via TheRock (github.com/ROCm/TheRock releases), not the
# legacy repo. 7.14.1 (2026-08-31) is a 2-commit point release: rocm-systems
# net-ib fault-injection default-off (ROCm-27881/AIRDEL-40) + an sdist
# self-dependency packaging fix — N/A for this stack, no bump (re-checked
# 2026-09-01, `therock-7.14.1` unchanged).
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-ubuntu-24.04/tags?page_size=100&name=7.1" | jq -r '.results[].name' | sort -V | tail

# Froggeric chat template — current pin is the first line of chat-templates/qwen.jinja
# (template_version = "qwen3.8-froggeric-vXX.X"). Compare against upstream main:
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
for n in 35288 47087 48375 52872 47602 51250 52520 45238 51562 51812 51837 40707 52527 52789 48815 52817 52959 51198 49125 53479 51571 54039 54360 54498 53504 53488 51599 54076 53798 50409 54163; do
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

- **GPU**: `gfx1201` (RDNA4, 2× R9700), ROCm 7.14.0. ROCm-only issues and
  AITER unified-attention paths are in scope; NVIDIA/CUDA-only fixes are not.
- **Models**: Qwen3.6-27B (dense, MTP4), Qwen3.6-35B-A3B (MoE, MTP off),
  Qwen3.8-27B (hybrid GDN, MTP3, 256K context, bf16 KV). Anything touching:
  hybrid Mamba/GDN models, MTP/speculative decoding, prefix caching
  (align mamba cache mode), fp8 KV, or `ROCM_AITER_UNIFIED_ATTN` is in scope.
- **Chat template**: froggeric `chat-templates/qwen.jinja` (pinned, e.g.
  v22.4). A newer version matters when it changes prompt rendering in ways
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
     patch** (upstream PR still open; in merge conflict as of 2026-08-31)
  - `#52872` GDN/hybrid prefill peak under-predicted; `--max-num-batched-tokens`
    also sizes the CUDA-graph pool
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
    `block_size=832` on the bf16-KV default (was `1600` on fp8), so
    incremental multi-turn prefixes never hit — measured **0% on the 30-turn
    qwen3.8-27b probe (re-confirmed 2026-08-23)**. Note the cumulative
    `vllm:prefix_cache_hits_total` is non-zero: caching *does* hit on
    repeated-identical-prompt workloads — the failure is specific to the
    incremental shared-prefix pattern, not a global no-op. Fixes in flight:
    `#52527` (metrics), `#48815` (MTP align retention), **`#53479`
    (2026-08-25, the leading candidate — retention-aware boundary
    materialization + removal of the speculative one-block back-off; makes the
    store side consistent with the `#52216` retention-0 default; open, not
    merged; interacts with our `--prefix-cache-retention-interval` pin so
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
    CLI flag remains). Validate with the prefix-cache probe on any bump;
    re-check the flag's default on v0.29.
  - `#53041` RFC: tiered SWA/Mamba checkpointing (HBM tail + periodic store)
    + recompute backfill for divergent hybrid prefix hits (same family as
    `#52959`/`#52789`; monitor)
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
    prefix hit (832 tokens here on the bf16-KV default; was 1600 on fp8),
    bounding the prefix-cache win for MTP even after `#45238` is fixed. Monitor
    for a merged implementation.
  - `#51562` GDN metadata misclassifies stateless first chunk (open)
  - `#52959` RFC: internal state checkpoints for Mamba align mode (same
    family as `#52789`; in flight, not merged)
  - `#40707` hybrid Mamba scheduling deadlock with 2+ large images in one
    prompt (align block-split collapses to 0 → request hangs forever, engine
    never recovers). **Mitigated 2026-08-28**: all profiles now pass
    `--limit-mm-per-prompt image: 1`, so the 2+-image trigger is unreachable
    (multi-image prompts are rejected with a 400). Fix PR `#40709` is
    **not merged** (absent from v0.28.0rc2) — monitor it; re-raise the image
    cap if it lands.
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
    #51571; also cited as the root cause of `#35288`) — if it lands in a
    release we adopt, re-test before dropping `--no-async-scheduling`.
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
  - `#54498` (2026-08-27, open, checked 2026-09-01): V1 EAGLE/MTP drafter
    feeds the M-RoPE **temporal** dim (`positions[0]`) to the KV-slot
    computation on `SupportsMRoPE` targets — on any prompt with an image the
    temporal coord lags the absolute token index, so each draft step writes
    draft K/V into a **prompt** slot (overwriting real prompt K/V) while
    attention reads the full span: acceptance drops and the error compounds
    with K. **Affects this stack**: Qwen3.8-27B is M-RoPE (local config:
    `mrope_section [11,11,10]`, `mrope_interleaved`), we run MTP3 on the V1
    runner (hybrid GDN is excluded from `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`),
    and we serve images (capped 1). Buggy line verified in v0.28.1rc0
    (`llm_base_proposer.py:787`). Text-only prompts are unaffected (dims ==
    absolute index). Upstream measurements: Qwen3.8-27B K=6 ≈ -5.7% mean
    acceptance vs the V2 runner; the gap grows with K (K=3 ≈ -0.3%, K=15
    -24.8% on a VL-derived 27B). The prompt-K/V overwrite implies an
    unmeasured output-quality risk on image+MTP requests (upstream measured
    acceptance only). Fix PR `#54519` was **closed unmerged 2026-09-01**
    (superseded), leaving `#54716` as the sole fix (open, actively updated
    2026-09-01, scope confirmed M-RoPE-only after review; in no release).
    **Monitor**; no in-repo mitigation (text-only unaffected) — if
    image+MTP acceptance matters, backport the merged fix as a local patch
    or bump when it lands. (Companion `#54555`/`#54621` xDRoPE
    positions-buffer — N/A, we're M-RoPE.)
  - `#51599` (2026-09-01, PR, open): the `#51571` fix — see the `#51571`
    entry above.
  - `#54076` (2026-09-01, PR, open): `_mamba_block_aligned_split` must chunk
    on the **Mamba group's** block size, not `cache_config.block_size` (the
    min over all groups) — otherwise mandatory chunk ends land on a grid the
    worker can never materialize a Mamba state at. Repro is a Qwen3.8-27B
    hybrid + spec drafter with mismatched target/drafter attention blocks
    (1648/816); our MTP drafter group can create the same geometry. Monitor
    for a merge.
  - `#53798` (2026-09-01, PR, open): align-mode `add_request` seeds the
    running-state block column by the scheduler block size instead of the
    (page-unification-scaled) Mamba block size, so a request admitted with
    `num_computed_tokens > 0` — explicitly under
    `--prefix-cache-retention-interval`, which we pin — points its precopy
    source into a neighbour's row (silent wrong-state read) or past the
    table (IMA in `precopy_mamba_align_fused_kernel`). Monitor for a merge.
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
    monitor as `#52817`-family signal only.
 
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
supported set but N/A on gfx1201). #50264 (RDNA hybrid-Mamba decode collapse
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
 shows no degradation with MTP on). We're on v0.28.1rc0, so no action; the
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

### 4. Local patches vs upstream

`patches/vllm/*.patch` and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time. Before bumping any pin:

- The aiter patches (version-locked to `AITER_REF` v0.1.20) are **RDNA4-local
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
