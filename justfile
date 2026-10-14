set positional-arguments

# Container runtime: "docker" (default) or "podman". Override with
# `just --set runtime podman <recipe>` or `RUNTIME=podman just <recipe>`.
runtime := env_var_or_default('RUNTIME', 'docker')

# Model profile: "qwen3.8-27b" (default), "qwen3.6-27b", or "qwen3.6-35b-a3b".
# Selects env/${model}.env for model-specific server arguments.
# Override with `just --set model qwen3.6-27b <recipe>` or
# `MODEL_PROFILE=qwen3.6-27b just <recipe>`.
model := env_var_or_default('MODEL_PROFILE', 'qwen3.8-27b')

# Exported so compose can resolve the `env_file:` path for the model profile.
export MODEL_PROFILE := model

# Every compose call must load `.env` (build pins) plus the env_file stack
# that compose.yaml declares (common → aiter → qwen3.6 → profile), because
# passing --env-file disables the implicit .env resolution, so each must be
# listed explicitly.
compose := runtime + " compose --env-file .env --env-file env/2xr9700.vllm.common --env-file env/aiter-unified-attention.env --env-file env/qwen3.6.env.common --env-file env/" + model + ".env"

# LMCache experiment overlay: same stack + overlay file + lmcache.env last
# (its values win). Only used by the lmcache-* recipes below.
lmcache_compose := runtime + " compose -f compose.yaml -f compose.lmcache.yaml --env-file .env --env-file env/2xr9700.vllm.common --env-file env/aiter-unified-attention.env --env-file env/qwen3.6.env.common --env-file env/" + model + ".env --env-file env/lmcache.env"

_default:
    @just --list

# Validate the compose config for the selected model profile.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -r .env ]; then
        printf 'error: no .env found. Copy the template: cp .env.example .env\n' >&2
        exit 1
    fi
    profile="env/{{model}}.env"
    if [ ! -r "$profile" ]; then
        printf 'error: no such model profile: %s\n' "$profile" >&2
        printf 'available profiles:\n' >&2
        for f in env/*.env; do
            case "$f" in env/aiter-*) continue ;; esac
            printf '  %s\n' "$(basename "$f" .env)" >&2
        done
        exit 1
    fi
    {{compose}} config --quiet
    printf 'Config OK (profile: {{model}}, model: %s).\n' \
        "$(grep -m1 '^VLLM_MODEL=' "$profile" | cut -d= -f2-)"

# Ensure the whole-home mount sources exist and are owned by the current user.
# Docker's daemon pre-creates missing bind-mount sources as root, which breaks
# non-root container writes and forces sudo for clear-vllm-caches; pre-creating
# them here prevents that. Individual ~/.cache/* dirs are created lazily by the
# container as the user, so only the top-level dirs need pre-creation. Safe to
# run repeatedly.
ensure-cache-dirs:
    #!/usr/bin/env bash
    set -euo pipefail
    cache_dirs=(
        "$HOME/.cache"
        "$HOME/.vllm-workspace"
    )
    uid="$(id -u)"
    gid="$(id -g)"
    mkdir -p "${cache_dirs[@]}"
    needs_sudo=0
    for dir in "${cache_dirs[@]}"; do
        if [ "$(stat -c '%u' "$dir")" != "$uid" ]; then
            needs_sudo=1
        fi
    done
    if [ "$needs_sudo" -eq 1 ]; then
        printf 'Some cache dirs are root-owned; fixing ownership (one-time sudo required).\n' >&2
        sudo chown -R "$uid:$gid" "$HOME/.cache" "$HOME/.vllm-workspace"
    fi
    printf 'Cache dirs ready (owner: %s).\n' "$(id -un)"

# Remove host cache directories written by the container (under ~/.cache).
clear-vllm-caches:
    #!/usr/bin/env bash
    set -euo pipefail

    cache_dirs=(
        # "$HOME/.cache/huggingface"
        "$HOME/.cache/vllm"
        "$HOME/.cache/triton"
        "$HOME/.cache/torchinductor"
        "$HOME/.cache/aiter"
        "$HOME/.cache/comgr"
        "$HOME/.cache/tvm-ffi"
        "$HOME/.cache/tilelang"
    )

    printf 'Removing vLLM host cache directories:\n'
    for dir in "${cache_dirs[@]}"; do
        if [ ! -e "$dir" ]; then
            printf '  %s (absent, skipped)\n' "$dir"
        elif [ ! -w "$dir" ]; then
            printf '  %s (not writable, using sudo)\n' "$dir"
            sudo rm -rf -- "$dir"
        else
            printf '  %s\n' "$dir"
            rm -rf -- "$dir"
        fi
    done

build: check
    @{{compose}} build

rebuild: check
    @{{compose}} build --no-cache

# Build the shared aiter JIT infrastructure (module_aiter_core, the
# unified-attention triton kernels) in a single throwaway process BEFORE the
# server starts. On a fresh cache the first `vllm serve` run would otherwise
# have the model-inspection subprocess and both TP workers racing to build
# these kernels: losers get `ModuleNotFoundError: aiter.ops.triton.unified_attention`
# and a subprocess that dies at exit can leave a stale aiter baton lock that
# deadlocks startup. Warm them first so every later build is a 0s cache hit.
prewarm: check ensure-cache-dirs
    #!/usr/bin/env bash
    set -euo pipefail
    # A process that dies mid-build can leave a stale aiter baton lock that
    # `file_baton` never detects as dead, hanging the next build forever. No
    # aiter process runs between container stops, so clearing any leftover lock
    # before building is always safe here.
    rm -f "$HOME/.cache/aiter/jit/build/lock_"* 2>/dev/null || true
    printf 'Pre-warming aiter JIT kernels ...\n'
    {{compose}} run -T --rm --no-deps --entrypoint python vllm \
        -c "import aiter; from aiter.ops.triton.unified_attention import unified_attention"
    printf 'aiter pre-warm complete.\n'

# Ensure the selected profile's calibrated KV-scale model copy exists. Profiles
# that want calibrated fp8 KV declare VLLM_MODEL_ID (HF source id) + VLLM_MODEL
# (local copy path) in their env file. If the local copy's sidecar is missing,
# create the copy from the HF cache and calibrate it in one throwaway container.
# Idempotent: skips once the sidecar exists. Runs before the server starts, so
# the GPUs are free for calibration.
ensure-kvscales:
    #!/usr/bin/env bash
    set -euo pipefail
    profile="env/{{model}}.env"
    model_id="$(grep -m1 '^VLLM_MODEL_ID=' "$profile" | cut -d= -f2- || true)"
    local_model="$(grep -m1 '^VLLM_MODEL=' "$profile" | cut -d= -f2- || true)"
    if [ -z "$model_id" ]; then
        printf 'profile %s has no VLLM_MODEL_ID; skipping kv-scale setup.\n' "$profile"
        exit 0
    fi
    sidecar="$local_model/model-kvscales.safetensors"
    if [ -f "$sidecar" ]; then
        printf 'kv-scales up to date: %s\n' "$local_model"
        exit 0
    fi
    printf 'setting up calibrated KV-scale copy: %s -> %s\n' "$model_id" "$local_model"
    python3 tools/setup_kvscales.py "$model_id" "$local_model"
    printf 'calibrating fp8 KV scales ...\n'
    {{compose}} run -T --rm --no-deps -e CALIB_KV_LOG=/workspace/kvscale.log \
        --entrypoint python vllm \
        "$PWD/tools/calibrate_kv_scales.py" "$local_model"
    printf 'kv-scales calibrated: %s\n' "$local_model"

# Remove the calibrated model copy so the next `just up` recalibrates it.
clear-kvscales:
    #!/usr/bin/env bash
    set -euo pipefail
    local_model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2- || true)"
    if [ -n "$local_model" ] && [ -d "$local_model" ]; then
        rm -rf -- "$local_model"
        printf 'removed %s\n' "$local_model"
    else
        printf 'no local model copy for profile %s.\n' "{{model}}"
    fi

up: check ensure-cache-dirs prewarm ensure-kvscales
    #!/usr/bin/env bash
    set -euo pipefail
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    {{compose}} up -d
    printf 'Waiting for server to be ready ...\n'
    ready=0
    # Wait window matches the compose healthcheck start_period (600s): cold
    # caches (fresh triton/compile state) can exceed 300s to become healthy.
    for _ in $(seq 1 300); do
        if curl -sf --max-time 3 http://localhost:8180/health > /dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        printf 'error: server did not become ready within 600s. Run `just logs`.\n' >&2
        exit 1
    fi
    printf 'Warming up Triton kernels ...\n'
    if curl -sf --max-time 300 http://localhost:8180/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${served_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":100,\"temperature\":0}" \
        > /dev/null; then
        printf 'Warmup complete. Serving %s at http://localhost:8180/v1\n' "$served_name"
    else
        printf 'error: warmup request failed. Run `just logs`.\n' >&2
        exit 1
    fi
    printf 'Starting dashboard ...\n'
    {{compose}} --profile dashboard up -d dashboard
    printf 'Dashboard at http://localhost:8083\n'

# Run a command inside the running vLLM container (e.g. `just exec bash`).
exec *args:
    @{{compose}} exec vllm {{args}}

logs:
    @{{compose}} logs -f

bench: check
    #!/usr/bin/env bash
    set -euo pipefail
    model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)"
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    tokenizer="$(grep -m1 '^VLLM_TOKENIZER=' "env/{{model}}.env" | cut -d= -f2-)"
    printf 'Benchmarking %s (pp2048, tg32+128) ...\n\n' "$model"
    uvx llama-benchy@0.4.0 --base-url http://localhost:8180/v1 \
        --model "$served_name" \
        --tokenizer "$tokenizer" \
        --pp 2048 \
        --tg 32 128 \
        --runs 3 \
        --enable-prefix-caching \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --format md

# Bench + log JSON to observability/history.jsonl for the dashboard history chart.
bench-json: check
    #!/usr/bin/env bash
    set -euo pipefail
    model="$(grep -m1 '^VLLM_MODEL=' "env/{{model}}.env" | cut -d= -f2-)"
    served_name="$(grep -m1 '^VLLM_SERVED_NAME=' "env/{{model}}.env" | cut -d= -f2-)"
    tokenizer="$(grep -m1 '^VLLM_TOKENIZER=' "env/{{model}}.env" | cut -d= -f2-)"
    # Single run (json): llama-benchy has no combined json+md format, so a
    # second `--format md` invocation would re-bench (double the ~2-4 min).
    # Use `just bench` for the md report.
    printf 'Benchmarking %s (pp2048, tg32+128) json ...\n\n' "$model"
    out="$(mktemp)"
    uvx llama-benchy@0.4.0 --base-url http://localhost:8180/v1 \
        --model "$served_name" \
        --tokenizer "$tokenizer" \
        --pp 2048 \
        --tg 32 128 \
        --runs 3 \
        --enable-prefix-caching \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --format json | tee "$out"
    ts="$(date +%s)"
    python3 tools/log_bench.py "$out" "$ts" "$served_name" >> observability/history.jsonl
    echo "Logged to observability/history.jsonl (use \`just bench\` for the md report)"

# Depth sweep sized to the live profile (rungs fit under VLLM_MAX_MODEL_LEN;
# logs to observability/depth_history.jsonl). Pass through extra args, e.g.
# `just bench-depth --depth 0 65536 --dry-run`.
bench-depth *args: check
    @python3 benchmarks/depth_sweep.py {{args}}

# Concurrency sweep sized to the live profile (levels cover 1..max_num_seqs
# over the depth ladder; logs to observability/conc_history.jsonl), e.g.
# `just bench-conc --depth 0 --dry-run`.
bench-conc *args: check
    @python3 benchmarks/conc_sweep.py {{args}}

logs-dashboard:
    @{{compose}} --profile dashboard logs -f dashboard

# Dashboard rides along with `up` (and is removed by `down`); these recipes
# control it standalone.
# `just up` starts vLLM + dashboard; use `just dashboard-down` to stop only it.
dashboard-up: check
    @{{compose}} --profile dashboard up -d dashboard
    @echo "Dashboard at http://localhost:8083 (profile: dashboard)"

dashboard-down:
    -@{{compose}} --profile dashboard stop dashboard 2>/dev/null || true
    -@{{compose}} --profile dashboard rm -f dashboard 2>/dev/null || true
    @echo "Dashboard stopped."

down:
    @{{compose}} down
    -@{{compose}} --profile dashboard stop dashboard 2>/dev/null || true
    -@{{compose}} --profile dashboard rm -f dashboard 2>/dev/null || true

# --- LMCache MP-mode experiment (qwen3.8-27b hybrid) ---
# Gate 0: read the unified block size N from the LIVE server logs and reconcile
# env/lmcache.env (CHUNK_SIZE=N, MAX_BATCHED_TOKENS=2N-1) before building.
lmcache-probe-N:
    # No --tail: the N line is printed once at engine startup, i.e. at the
    # OLDEST end of the logs; a tail window misses it on long-lived servers.
    # The trailing `grep .` fails closed (tail exits 0 even on empty input).
    @{{compose}} logs --no-log-prefix vllm 2>/dev/null \
        | grep -i "Setting attention block size" | tail -1 | grep . || \
        (echo "N line not found — is the live server up? (just up, wait, retry)" >&2; exit 1)

# Validate the overlay config (does not start anything).
lmcache-check:
    @{{lmcache_compose}} --profile lmcache config --quiet
    @echo "LMCache overlay config OK."

# Build the experiment engine image (live fullbuild + LMCache from source).
# Highest-risk step: fails here (HIP compile) => stop, nothing else is valid.
lmcache-build: lmcache-check
    #!/usr/bin/env bash
    set -euo pipefail
    ref="$(grep -m1 '^LMCACHE_REF=' env/lmcache.env | cut -d= -f2-)"
    printf 'Building test image with LMCACHE_REF=%s ...\n' "$ref"
    {{runtime}} build -f Dockerfile.lmcache-test \
        --build-arg "LMCACHE_REF=$ref" \
        -t localhost/vllm-lmcache-test:latest .

# Start sidecar + test engine on host 8180. LIVE MUST BE DOWN (GPUs + port).
lmcache-up: lmcache-check
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "$({{runtime}} ps -q --filter 'name=^vllm$')" ]; then
        printf 'error: live vllm container is running. Run `just down` first.\n' >&2
        exit 1
    fi
    # Bare `up -d` would ALSO start the base `vllm` service (it has no profile,
    # and profiles only ADD services) — a rogue live engine on the same GPUs +
    # port. Always name the two experiment services explicitly.
    {{lmcache_compose}} --profile lmcache up -d lmcache vllm-lmcache-test
    printf 'Waiting for test engine on :8180 ...\n'
    ready=0
    for _ in $(seq 1 300); do
        if curl -sf --max-time 3 http://localhost:8180/health > /dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        printf 'error: test engine not ready in 600s. Run `just lmcache-logs`.\n' >&2
        exit 1
    fi
    printf 'Test engine ready. Unified block size (must equal LMCACHE_CHUNK_SIZE):\n'
    {{lmcache_compose}} logs --no-log-prefix --tail 20000 vllm-lmcache-test 2>/dev/null \
        | grep -i "Setting attention block size" | tail -1 || true

# Correctness + hit-rate gates on the test engine (:8180).
# 1) store run 2) local-only reset (LMCache external preserved) 3) retrieve run
# 4) NaN/tool/thinkoff no-regression probes. Reject on zero cross-restart
# hits or any probe FAIL (see plan: score-level equivalence is not enough
# for tool-calling traffic).
lmcache-verify:
    #!/usr/bin/env bash
    set -euo pipefail
    curl -sf --max-time 3 http://localhost:8180/health > /dev/null
    printf '=== 1. store run (cold) ===\n'
    python3 benchmarks/prefix_cache_probe.py qwen3.8-27b 4
    printf '=== 2. local-only prefix-cache reset (external LMCache kept) ===\n'
    curl -sf -X POST http://localhost:8180/reset_prefix_cache
    printf '\n=== 3. retrieve run (must show external hits) ===\n'
    python3 benchmarks/prefix_cache_probe.py qwen3.8-27b 4
    printf '=== 4. no-regression probes ===\n'
    python3 benchmarks/nan_checkpoint_probe.py qwen3.8-27b
    python3 benchmarks/tool_truncation_probe.py
    python3 benchmarks/thinkoff_probe.py
    printf 'VERIFY COMPLETE. Remaining manual gate: `just bench` parity\n'
    printf '(pp2048/tg + MTP1 acceptance in the 79-93%% band) + cross-restart\n'
    printf 'retrieve (lmcache-down, lmcache-up, re-run step 3 only).\n'

lmcache-logs:
    @{{lmcache_compose}} --profile lmcache logs -f lmcache vllm-lmcache-test

lmcache-down:
    -@{{lmcache_compose}} --profile lmcache stop lmcache vllm-lmcache-test 2>/dev/null || true
    -@{{lmcache_compose}} --profile lmcache rm -f lmcache vllm-lmcache-test 2>/dev/null || true
    @echo "LMCache experiment stopped (live untouched; restart with \`just up\`)."
