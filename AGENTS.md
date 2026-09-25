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
# vLLM — pin VLLM_REF=v0.30.0 (final 2026-09-22; ROCm 10.0
# + torch 2.13). Carries over v0.29.0: #55760+#55861 (dense
# prefix_cache_retention_interval default — the #53504-family root-cause
# fix), #53877 (packed GDN decode beta FP32), #53821 (AITER unified-attn
# metadata across graph replay), #54994 + #52041 (multimodal prefix-cache
# worker paths), ROCm perf #52033 + #53712 + #53818. New in v0.30.0:
# #54713 + #55450 (prefix family), #54826 (draft attention_backend on
# MRV2), #53388 (trailing-block-drop opt-out — re-probe #55766 on any
# bump landing EAGLE-drop changes), #53945 (last-block replay impl),
# #55178, #54251 (GDN warmup), #54965 (ROCm W4A16 skinny GEMM), #47562
# (47137 content half — local hunk dropped). NOT in v0.30.0 (verified via
# compare API): #48606 (Quark W4A16, merged main 2026-09-18 post-cut —
# keep carrying, drop at v0.31+), #51565 (#51562 GDN fix, merged
# 2026-09-22 post-cut — rides v0.30.1/v0.31). No #54360 fix PR exists yet.
# Next target: v0.30.1/v0.31 (watch #51565, #53479, #54076, #50409,
# #54360-fix, #51599). MRV2 is the default for all our profiles (#53183).
gh release list -R vllm-project/vllm --limit 8

# AITER — pin AITER_REF=v0.1.20.post1 (TheRock 10.0 hipcub fix). v0.1.22.post1
# (2026-09-17, latest) = MoE/DSv4/gfx950 cherry-picks only — no bump (nothing
# touches unified-attention/gfx1201/LDS). UA refactor #5088 + perf #4761 mean
# any bump needs rebase + tools/tune_ua_config.py re-run. LDS fix #4868 is
# main-only (CLOSED aiter#4329; vllm#48723 still OPEN) — local LDS-cap patch
# stays load-bearing until a pinned ref contains #4868 (then verify
# equivalence: upstream shrinks stages-then-tile generically; ours is bf16
# caps + gfx1201 tuning; note aiter#5035 still open).
gh release list -R ROCm/aiter --limit 8

# Flash Attention — pinned to a commit; compare HEAD to FLASH_ATTN_REF
# (HEAD a369df7, 2 commits past pin, both Blackwell-only — no bump).
git ls-remote https://github.com/ROCm/flash-attention.git HEAD

# ROCm base image — ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.0.0-full (only 10.0
# tag so far). 10.0.x releases via TheRock releases page.
curl -s "https://hub.docker.com/v2/repositories/rocm/dev-ubuntu-24.04/tags?page_size=100&name=10.0" | jq -r '.results[].name' | sort -V

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
# Re-check watchlist status (open/closed/resolved) + any new labels:
for n in 35288 47087 48375 52872 47602 51250 52520 45238 51562 51812 51837 40707 52527 52789 48815 52817 52959 51198 49125 53479 51571 54039 54360 54498 53504 53488 51599 54076 53798 50409 54163 55600 55533 54713 55450 48007 57580 58020 50891; do
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
  - `#35288` MTP concurrency corruption — mitigated by `max-num-seqs 2`;
    precise root cause is the async+MTP align race `#51571` (fix `#51599`),
    and the corruption requires async scheduling, which we disable.
  - `#47087` MTP token loops on Qwen3-MoE — **resolved** by #51113 (in
    v0.27.1); MTP4 re-enabled on 35B-A3B after a clean re-test (~2x decode).
  - `#51812` (GDN gate/spec-token alignment) + `#51837` (KV-first blocks vs
    Mamba pages) — **both merged upstream 2026-08-11**, local patches dropped.
    #51837 is inert here anyway (AITER unified attn is blocks-first).
  - `#48375` MambaManager ignores `drop_eagle_block` (MTP + prefix caching
    corrupts hybrid recurrent state, #43559/#50188) — **carried as a local
    patch** (upstream PR open). Drop when a release contains the fix.
   - `#52872` GDN/hybrid prefill peak under-predicted; `--max-num-batched-tokens`
     also sizes the CUDA-graph pool. qwen3.8-27b pins 2048 (concurrent-ITL A/B:
     8192 default stalled the co-decoder 150–200x; 2048 → ~1 s ITL, −3.4%
     pp2048). Re-check headroom + re-run `benchmarks/conc_itl_probe.py` on any
     bump that changes chunk/graph-pool sizing.
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
    checkpoint vetoes every attention-group hit. Live geometry since
    2026-09-21: `block_size=832` on the bf16-KV profile (1600 on fp8), so
    incremental multi-turn prefixes never hit — measured **0% on the 30-turn
    qwen3.8-27b probe (re-confirmed 2026-08-23, fp8/1600 geometry)**. Note the cumulative
    `vllm:prefix_cache_hits_total` is non-zero: caching *does* hit on
    repeated-identical-prompt workloads — the failure is specific to the
    incremental shared-prefix pattern, not a global no-op. Fixes in flight:
    `#52527` (metrics), `#48815` (MTP align retention), **`#53479`
    (the leading candidate — retention-aware boundary materialization +
    removal of the speculative one-block back-off; open, conflicting, author
    confirms the head is a superseded branch since merged `#54713` touched
    the same path — correctly NOT carried as a local patch)**,
    `#52789` (internal prefill checkpoints — merged 2026-08-22, Kimi-K3/
    FlashKDA TTFT win, not a fix for this geometry). **When a real fix
    merges**: prefer the version bump; carry a local patch only if no
    available release contains it. Related: `#52897` (priority-scheduling
    variant — N/A, we don't use priority), `#53749` (same family on two more
    hybrids — reinforces the checkpoint fix is the binding constraint).
  - `#53504` first-repeat prefix miss on hybrid+MTP — **fixed in v0.29.0** by
    `#55760`+`#55861` (dense retention default; issue CLOSED as COMPLETED).
    Retention pins dropped from all env files. Re-run the prefix-cache probe
    after any bump to confirm the first repeat still hits.
   - `#54713` (merged — replay-boundary retention for EAGLE resends) +
     `#55450` (merged — Mamba retirement across null gaps): both verified IN
     the v0.30.0 tag, ride the v0.30.0 final (same for `#53388` + `#54826`);
     `#55450` is memory-efficiency, not correctness for our geometry. No action.
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
      Note: the live profile is **bf16 KV / block 832** since 2026-09-21 (the
      09-10 probe ran fp8/1600 — carrier mod-832 {4,6,8,10} now). **Re-probe**
      on the v0.30.0 bump (lands #53388 EAGLE-drop opt-out + #53945 replay
      impl — both touch the back-off geometry), if spec decode changes
      (MTP off → hit reaches the last aligned boundary → carrier live), or on
      any NaN / `"!"`-spam / empty reply in the field.
       **2026-09-19**: reproduced on `main` (ngram K=5, TP4, block 816):
       the trigger is a final prefill chunk of exactly `1 + num_speculative_tokens`
       tokens, dispatched to the captured FULL spec-verify decode graph while GDN
       metadata treats it as prefill (graph-dispatch family #49918/#47123, NOT
       align chunk-splitting #54076). r=8 NOT reproduced; failing-pair shape
       differs (A already corrupted vs A-fine here) — partial explanation only,
       no PR yet. For MTP3 the trigger chunk is 4 tokens = the documented
       mod-1600 {4,…} carrier; masking analysis unchanged.
   - `#53041` RFC: tiered SWA/Mamba checkpointing (HBM tail + periodic store)
     + recompute backfill for divergent hybrid prefix hits (same family as
     `#52959`/`#52789`; monitor)
    - `#55697` (2026-09-09, RFC, open): application-directed prefix
     checkpoints for Mamba/hybrid prefix caching — same family as `#53041`/
     `#52959`; monitor-only
   - `#58303` (2026-09-24, open): pushback on the dense-retention default we
     ride (`#55760`/`#55861`) — dense checkpoints share the pool with attention
     KV, so under interleaved long conversations the pool fills and prefix
     reuse returns to 0% (same symptom as `#53595`, other end of the knob).
     Proposes a sparse interval multiple of `scheduler_block_size` as the
     middle; no PR yet. Not our workload shape (2 concurrent seqs, far from
     pool ceiling) — monitor alongside `#53041`/`#55697`.
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
    prefix hit (832 tokens here on the bf16-KV profile; 1600 on fp8),
    bounding the prefix-cache win for MTP even after `#45238` is fixed.
    Implementation `#53945` landed 2026-09-16 and IS in v0.30.0 — re-run the
    prefix-cache + conc-ITL probes on the v0.30.0 bump to measure the bound;
    follow-up `#57329` adapts it for Mamba2 (open).
  - `#51562` GDN metadata misclassifies stateless first chunk — **CLOSED as
    COMPLETED 2026-09-22** via merged fix `#51565` (merged 2026-09-22 01:11
    UTC, hours after the v0.30.0 cut — verified NOT in the v0.30.0 final).
    Rides v0.30.1/v0.31. No action until then.
  - `#58020` (2026-09-21, open, new): engine-resolved prefix-cache match unit
    not propagated to workers — hybrid geometry with attn block 16 / mamba
    block 1600 (mirrors our fp8-KV geometry): the Mamba checkpoint consumer
    derives a 1600-token unit while the engine hashes at 16 → checkpoint/hash
    misalignment. Same `#45238` family, possible co-root-cause of the 0%-hit
    no-op. No PR yet — monitor.
  - `#52959` RFC: internal state checkpoints for Mamba align mode (same
    family as `#52789`; in flight, not merged)
  - `#40707` hybrid Mamba scheduling deadlock with 2+ large images in one
    prompt (align block-split collapses to 0 → request hangs forever, engine
    never recovers). **Carried as a local patch**
    (`patches/vllm/40707-mamba-block-aligned-split-deadlock.patch`, verbatim
    upstream fix `#40709`, open as of 2026-09-22 — applies cleanly on v0.30.0).
    Drop when a pinned `VLLM_REF` contains the fix. Live since the 2026-09-19
    rebuild; verified with `benchmarks/multi_image_probe.py` (PASS).
  - `#51571` async MTP align accepted-count race (open): async scheduling +
    MTP + hybrid GDN + `mamba-cache-mode align` → accepted-token D2H counts
    gathered from a mutated `InputBatch` after `condense()` (repeated/dropped/
    garbled tokens). Mitigation: `compose.yaml` passes `--no-async-scheduling`
    for all spec-decode profiles (tracks `VLLM_SPEC_DECODE`); re-check upstream
    before removing. Fix PR `#51599` (open, retargeted `[Bugfix][MRv1]` — all
    our profiles run MRV2, which shrinks the exposed surface, but
    `--no-async-scheduling` stays): if it lands in a release we adopt, re-test
    before dropping `--no-async-scheduling`.
  - `#54039` (question): vLLM's own ROCm CI disables async+MTP (#32275,
    unroot-caused shm-broadcast hang) while the default still enables that
    combination. Same combination we disable via `--no-async-scheduling`;
    monitor for a merged default change.
  - `#54360` (open): spec decode drives prefix-cache hits to **0** on
    Qwen3.8-27B hybrid GDN align — **nightly-only** (not in our pin; our
    v0.29.0 probe demonstrably hits). Root cause located 2026-09-18:
    `_annotate_eagle_groups()` can't identify a draft group for plain MTP →
    fallback flags *all* groups as eagle (insertion-side failure). Monitor for
    the fix PR ahead of the v0.30.0 final; re-run the prefix-cache probe if a
    bump lands that includes it.
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
     `#54716` as the sole fix (open, in no release — re-checked 2026-09-12,
     still `OPEN` + MERGEABLE, updated 2026-09-10T07:20Z).
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
     Revisit conditions in `archive/benchmarks/2026-09-03_qwen3.8-27b_v1_vs_v2.md`.
  - `#54928` (2026-09-02, open): **Qwen3.8-27B** (our exact target) with
    DFlash2 + thinking is not greedy-equivalent to target-only (diverges at
    generated token 30; text-only, K=1, `--enforce-eager` — rules out
    draft-depth and CUDA-graph artifacts). No direct impact: we rejected
    DFlash2 (see archive/DEADENDS.md) and run MTP3; the DFlash2
    implementation itself is an in-flight PR (#52816), not in any release.
    **Monitor**: if the root cause is confirmed in the shared verify/
    GDN-state path, MTP3 + thinking is implicated too — run an MTP
     greedy-equivalence probe (target-only vs MTP3, temperature=0,
     thinking prompt) at that point.
  - `#50891` (open, added 2026-09-24): AOT compile-cache key ignores
    `limit_mm_per_prompt` (`ModelConfig.compute_hash()` lists it in
    `ignored_factors`), so same-model runs with different image caps collide
    and crash in `profile_run` (`AttributeError: 'NoneType' object has no
    attribute 'size'`; can also flip attention-backend selection — wrong
    twice over). Surfaced via dup `#58203` (Qwen3.8-27B-FP8 + images + MTP,
    NVIDIA/B200). **Not exposed here today**: all qwen3.8-27b runs use
    image-cap 1 and profile model names differ, so hashes don't collide.
    Hygiene rule: never alternate image caps / `--language-model-only` for
    the same model on a shared compile cache without clearing it first.
    Monitor for a fix (key the cache on the multimodal config).
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
  - `#54076` (2026-09-01, PR, open): `_mamba_block_aligned_split` must chunk
    on the **Mamba group's** block size, not `cache_config.block_size` (the
    min over all groups) — otherwise mandatory chunk ends land on a grid the
    worker can never materialize a Mamba state at. Repro is a Qwen3.8-27B
    hybrid + spec drafter with mismatched target/drafter attention blocks
      (1648/816); our MTP drafter group can create the same geometry.
      Monitor for a merge.
  - `#53798` (2026-09-01, PR, open): align-mode `add_request` seeds the
    running-state block column by the scheduler block size instead of the
    (page-unification-scaled) Mamba block size, so a request admitted with
    `num_computed_tokens > 0` (reachable via prefix-cache resumes, now dense
    by default) points its precopy source into a neighbour's row (silent
    wrong-state read) or past the table (IMA in
    `precopy_mamba_align_fused_kernel`). **Carried as a local patch**
    (patches/vllm/53798-mamba-align-resume-seed.patch, rebased to v0.30.0:
    v0.30.0 builds ModelState in load_model, so the bind hook runs after
    `kv_cache_config` assignment in `initialize_kv_cache`).
    Drop when a release contains the fix. Sibling `#55600` (2026-09-06,
    open) shows the same line crashes with a small-block drafter (DFlash2
    block 64/1024 vs `mamba_block_size` 7168) — no PR yet; reinforces the
    fix is incomplete on `main`.
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
     `--kv-cache-dtype fp8` barely moves `num_blocks`. Caveat on fp8-KV
     capacity expectations (alongside the `#52793` calibration note).
     Proposed levers (quantize Mamba state, decouple per-group page size)
     unimplemented anywhere in vLLM. Monitor; no action.
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
     cap raise. WIP fix `PR #55617` (2026-09-06; 2026-09-12: still OPEN +
     CONFLICTING/WIP). Monitor before raising `max-num-seqs`.
  - `#48606` (PR — **carried as a local patch for the AWQ trial
    profile**, **MERGED to main 2026-09-18**, after the v0.30.0 cut):
    native Quark W4A16 INT4/UINT4 `real_quantized` (`reorder`)
    loading path (`QuarkW4A16Int4` dense + MoE, canonicalized to the `awq_*`
    kernel layout). Without it vLLM cannot load
    `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` (`quant_method: quark` matches no
    scheme in the stock Quark plugin; the AWQ loader expects `quant_config.
    json`). `patches/vllm/48606-quark-w4a16.patch` rebased to v0.30.0 on
    2026-09-22 to mirror the merged upstream (key-tuple dispatch,
    `init_scheme` weight-config path, keys-based MoE method,
    `super().__init__` restored, `has_g_idx` dropped per #54809).
    **Trial done 2026-09-10, kept as alternative profile**
    (`qwen3.8-27b-awq`, fp8 KV calibrated): kernel selects `RDNAHybridW4A16`
    on gfx1201 (the CUDA-only concern did NOT materialize on the MP-kernel
    path), decode +30–50% to d200K, prefill −26%, weights 10.3 GiB —
    full record `benchmarks/2026-09-10_qwen3.8-27b_awq_trial.md`. **Drop
     the patch when a pinned `VLLM_REF` contains the merge** (check merge
     commit vs tag, plus ROCm/gfx1201 kernel coverage — not just the merge;
     expected at v0.31+). **Verified NOT in the v0.30.0 final** (compare-API
     `diverged`, 2026-09-22) → keep carrying through any v0.30.0 pin.
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
 #50264 (Triton paged-attention fallback — we run AITER unified attn),
 #54906 (`thinking_token_budget` ignored — field we never send), #56419 (CPU
 backend), #56701 (KV offload+MTP — no offload), #56774 (hidden-state
  extraction), #55291 (Qwen3.6-27B-FP8 "!"-collapse — v0.21.0/L20 report, and
  a bounded 0.28.0 repro attempt found zero collapses; N/A, but tracked model
  + sticky corruption: glance if a modern repro appears), #56736 (hybrid
  Mamba/GDN + spec decode Xid 31 in the align precopy path — DFlash2 drafter
  on a v0.13-era NVIDIA fork with async on; same precopy family as carried
  #53798, signal only), #57838 (RowWise FP8 linear 5–24% slower than
  ChannelWise on gfx1201 — requires per-tensor/channel dynamic FP8 weights;
  our Qwen FP8 checkpoints are block-scaled → live log selects
  `TritonFp8BlockScaledMMKernel`, different path; re-check if a non-block
  FP8 checkpoint is ever served).

 Non-Qwen models (different arch/checkpoint format): #52833/#48568 (GLM),
 #51530 (DeepSeek), #56605 (GLM-5.3 word salad), #55280 (GLM kpool split IMA,
 gfx942), #54924/#54451/#56380 (GLM ROCm), #56506/#52911/#57149/#57230
 (DeepSeek/gfx950 perf), #54114 (GLM-5.1 reasoning), #55357 (Flash-Next MTP
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
  scope).

 Paths not reached here: PP ranks (#51752, #55951, #54709), DP attention
 (#51957, #48255), DCP (#54761, #57228), P/D disaggregation (#54392, #54926),
 KV connectors (#51805/#51766/#40017/#53505/#53514, #56972 Mooncake,
 #45407 LMCache, #54165), priority scheduling (#52897, #57580 —
 priority-preemption checkpoint visibility, same family), EP (#41862),
 GPTQ qzeros on gfx1201 (#51971 — we run FP8), gfx950 MLA (#52312),
 turboquant (#53180, #53334), numa-bind (#55416), XPU (#55425), prebuilt
 rocm/vllm image (#56945 — we build from source), GLM page-align/offload
 (#54458, #54831), gfx1030/gfx1100 (#54728, #54438), DeepEP (#54281),
 MTP `n_predict`/token auto-defaults (#55322/#55323 — we pass explicitly),
 client `stop` strings in think output (#53066 — our clients don't send
 stop), KV offloading (#52773), `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`
 (#53136 — never set; fault was gfx942 TP=8-specific), embedding/rerank-only
 kernels (#58060 — generative-only stack), multi-layer MTP
 layer-0 reuse (#52688/#53397 — both 27B models have
 `mtp_num_hidden_layers=1`, so N/A), in-flight model PR metadata
 (#53983/#53982 — QSA/Kpool side caches not on main), ngram spec decode
 (#56077 — we run MTP), ROCm 7.2.x libhsa (#56521 — we're on 10.0),
 gfx1151 (#57493/#57494 — different GPU), gfx1201 FP8 fall-through request
 (#28649 — OP retracted: already routes to W8A8).

 Requires an option we never pass: #53142 (align precopy IMA — needs
 explicit `--block-size`; #54199 retracted as its duplicate; a cluster of
 related fix variants #55507/#55601/#55688 is forming around our carried
 #53798 line — reinforces upstream is unsettled, keep carrying ours),
  #57032 (dflash drafter KV group — we run MTP; same annotation family as
  #54360), #58080 (MTP draft doesn't inherit `--hf-overrides`/YaRN — we pass
  no overrides; both 27B models run native context).

 Already resolved/stale for this stack: #40980 (R9700 TP2 deadlock —
 v0.19-era; AMD confirmed TP2 working, ours serves), #49851 (multimodal
  gfx1201 load failure — v0.25.1-era; we serve images), #47194 (hybrid+MTP
  tool/needle corruption — fixed in v0.28.0 by #51113, we're on v0.30.0),
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
   on any bump if startup crash-loops.

### 4. Local patches vs upstream

`patches/vllm/*.patch` and `patches/aiter/*.patch` are cherry-picks/overrides
applied at build time. Before bumping any pin:

- The two parser patches (`47137-tool-truncation-parity.patch`,
  `qwen3-thinkoff-kwarg-parity.patch`) are adaptations from third-party
  forks (radiance / GGZ14 lineage), not upstream PRs — each has a live probe
  (`benchmarks/tool_truncation_probe.py`, `benchmarks/thinkoff_probe.py`).
  Re-run its probe after any bump touching `vllm/parser/`.

- The aiter patches (version-locked to `AITER_REF` v0.1.20.post1) are **RDNA4-local
   work**, not upstream cherry-picks: `unified-attention-bf16-kv.patch`
   (bf16-KV LDS caps, the fix for upstream ROCm/aiter#4329 / vllm#48723 —
   #4329 CLOSED 2026-09-15 via main-only #4868, NOT in any release
   (verified 2026-09-21: #4868 merge commit not contained in v0.1.22.post1
   via compare API), and vllm#48723 still OPEN, so the patch stays
   load-bearing), `unified-attention-gfx1201-tune.patch` (per-arch gfx1201 tuning:
  attn_warps 4 in 3D decode ~1.4-1.9x, num_warps 8 in 2D large-prefill ~7%),
  and `allowed-archs-gfx1201.patch` (build-path arch acceptance). When a pinned
  `AITER_REF` contains #4868, the bf16-KV cap should be **dropped** (upstreamed)
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

