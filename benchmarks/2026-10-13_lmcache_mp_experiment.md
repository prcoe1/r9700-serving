# 2026-10-13 LMCache MP-mode experiment (qwen3.8-27b hybrid) — REJECTED

Contained experiment: LMCache multiprocess sidecar + test engine to see if
external KV cache restores cross-restart prefix reuse while upstream
vllm#45238 holds in-memory multi-turn hits at 0%.

## Verdict: REJECT — board-level incompatibility (root-caused 2026-10-13)

The test engine never becomes ready. Root cause is structural to this board,
not a config error:

- `lmcache_driven` (the only transfer mode supporting multi-group hybrids)
  requires the server to open the engine's GPU IPC handles
  (`hipIpcOpenMemHandle`), which needs GPU↔GPU peer mapping. **This board has
  no P2P**: `torch.cuda.can_device_access_peer(0,1) == False` both directions
  (measured 2026-09-24,
  `benchmarks/2026-09-24_qwen3.8-27b_rdna4_optimizations.md`; the two R9700s
  sit on separate PCIe root ports, `NCCL_P2P_DISABLE=1` is mandatory).
  Server-side open fails with `hipErrorInvalidDevicePointer` — the exact
  failure shape of upstream ROCm/clr#46 (two processes, separate GPUs, fails
  at `hipIpcOpenMemHandle`). Registration then times out engine-side ("did
  not respond to register_kv_caches within 300s").
- `engine_driven` (worker-side staging through host, no P2P needed)
  hard-fails at worker init by upstream design: "engine-driven transfer does
  not support multiple KV transfer groups now".

Ruled out along the way (same signature throughout): sidecar GPU invisibility
(fixed UUID resolution, exposed the pointer error), missing in-process HIP
context (H1: `torch.cuda.init()` pre-init, 2 devices visible — same error),
transport/host addressing (ZMQ control plane healthy), PCIe ACS/IOMMU
blocking (neither present; the lack is genuine no-P2P silicon/topology).

## Gate log

| Gate | Result |
|---|---|
| N probe (live TP2/bf16) | **832** (`interface.py:989`), matches committed defaults (chunk 832, batched 1663 = 2N−1) |
| Source build vs torch 2.13/ROCm 10.1 | PASS after 3 fixes: `LMCACHE_REF=dev` (no `main` branch), pre-install `grpcio==1.78.0` + `grpcio-tools==1.78.0` (build-system deps for `--no-build-isolation`), `libibverbs1` (mooncake dlopen). Image `localhost/vllm-lmcache-test:latest`, `lmcache ok` import check green |
| Sidecar boot | PASS (chunk_size=832, `Supported transfer mode: lmcache_driven`, HTTP + ZMQ healthy) |
| Connector load | PASS (`LMCacheMPConnector` via `kv_connector_module_path`, `Registering kv caches!`, `mode=lmcache_driven`) |
| `register_kv_cache` (server opens GPU IPC) | **FAIL** `hipErrorInvalidDevicePointer`, both with and without GPU devices on the sidecar / in-process CUDA init |
| `engine_driven` fallback | **FAIL** by design on hybrids (multi KV groups) |
| Correctness / hit-rate / perf gates | never reached (engine never ready) |

## Collateral fixes kept (real bugs in the harness)

- `just lmcache-probe-N`: dropped `--tail` (N prints once at startup = oldest
  end of logs; tail misses it on long-lived servers) + trailing `grep .` so a
  miss fails closed instead of passing silently.
- `just lmcache-up`: names services explicitly (`up -d lmcache
  vllm-lmcache-test`). Bare `up -d` also starts the base `vllm` service (no
  profile = always on) → rogue live engine on the same GPUs/port. Bit us once.
- `LMCACHE_HTTP_PORT=8088`: host 8080 belongs to nextcloud-aio-mastercontainer.

## Revisit when (watchlist)

- `engine_driven` gains multi-group support (removes the GPU-visibility
  requirement entirely — the cleaner architecture for this box, and the only
  path that avoids P2P).
- A TP=1 / single-GPU profile is ever served here: with one GPU the server
  opens same-device handles only, so the missing peer mapping may not bite.
  (Not applicable to any current 27B profile — TP=2 is mandatory.)
- Any bump touching the UA/vision path or `VLLM_REF`: the whole overlay
  (`Dockerfile.lmcache-test`, `compose.lmcache.yaml`, `env/lmcache.env`,
  `just lmcache-*`) is kept as-is and re-runnable; only `LMCACHE_REF` may
  need a re-pin (grpcio pins mirror upstream `requirements/build.txt`).

## Files (kept, inert for live)

`env/lmcache.env`, `Dockerfile.lmcache-test`, `compose.lmcache.yaml`,
`justfile` `lmcache-*` recipes. Live `just up` / `compose.yaml` untouched —
overlay is only loaded by `lmcache-*` recipes.
