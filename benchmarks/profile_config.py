#!/usr/bin/env python3
"""Shared live-profile reader for benchmark tooling.

Reads the same env-file stack `compose.yaml` hands the server (last wins):

    env/2xr9700.vllm.common
    env/aiter-unified-attention.env
    env/qwen3.6.env.common
    env/${MODEL_PROFILE}.env

so probes and sweep wrappers automatically follow the in-use config:
`VLLM_MAX_NUM_SEQS` sizes concurrency tests, `VLLM_MAX_MODEL_LEN` sizes
depth ladders and prompt caps. MODEL_PROFILE resolves like the justfile:
`$MODEL_PROFILE` env var wins, then `.env`, then `qwen3.8-27b`.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_STACK = [
    "env/2xr9700.vllm.common",
    "env/aiter-unified-attention.env",
    "env/qwen3.6.env.common",
]

DEFAULT_PROFILE = "qwen3.8-27b"


def _parse_env_file(path: Path) -> dict:
    out = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def active_profile() -> str:
    if os.environ.get("MODEL_PROFILE"):
        return os.environ["MODEL_PROFILE"]
    env = _parse_env_file(REPO_ROOT / ".env")
    return env.get("MODEL_PROFILE", DEFAULT_PROFILE)


def load_profile(profile: str | None = None) -> dict:
    """Effective env vars for `profile` (last file in the stack wins)."""
    prof = profile or active_profile()
    merged: dict = {}
    for rel in ENV_STACK + [f"env/{prof}.env"]:
        merged.update(_parse_env_file(REPO_ROOT / rel))
    merged["MODEL_PROFILE"] = prof
    return merged


def _as_int(cfg: dict, key: str, default: int) -> int:
    try:
        return max(1, int(cfg.get(key, default)))
    except (TypeError, ValueError):
        return default


def max_num_seqs(cfg: dict | None = None) -> int:
    return _as_int(cfg or load_profile(), "VLLM_MAX_NUM_SEQS", 2)


def max_model_len(cfg: dict | None = None) -> int | None:
    try:
        return max(1, int((cfg or load_profile())["VLLM_MAX_MODEL_LEN"]))
    except (KeyError, TypeError, ValueError):
        return None


def served_name(cfg: dict | None = None) -> str:
    cfg = cfg or load_profile()
    return cfg.get("VLLM_SERVED_NAME", cfg.get("MODEL_PROFILE", DEFAULT_PROFILE))


def live_max_model_len(base_url: str, timeout: float = 5.0) -> int | None:
    """Live `--max-model-len` from the running server (`GET /v1/models`
    exposes it as `data[0].max_model_len`). Returns None when the server
    is unreachable or the field is absent — callers fall back to the
    env-file value. Stdlib only (no extra deps for sweep wrappers)."""
    import json
    import urllib.request
    try:
        url = base_url.rstrip("/") + "/v1/models"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8", "replace"))
        data = payload.get("data") or []
        if not data:
            return None
        v = data[0].get("max_model_len")
        v = int(v)
        return v if v > 0 else None
    except Exception:
        return None


def resolve_max_len(cfg: dict | None = None, base_url: str | None = None,
                    timeout: float = 5.0) -> tuple[int | None, str]:
    """(max_len, source): live server first, env-file stack second.

    The depth ladder must fit the *launched* context window, which can
    differ from the env files (profile switch, CLI override, stale
    checkout) — so the live `GET /v1/models` value wins when reachable.
    Source is "live", "env", or "none"."""
    if base_url:
        live = live_max_model_len(base_url, timeout=timeout)
        if live is not None:
            return live, "live"
    env_len = max_model_len(cfg)
    if env_len is not None:
        return env_len, "env"
    return None, "none"


def depth_ladder(max_len: int, pp: int = 2048, tg: int = 32,
                 margin: int = 2048, rung: int = 1024) -> list[int]:
    """Depth rungs that fit in `max_len` tokens: powers of two plus a top
    rung at the largest `rung`-aligned depth below
    max_len - pp - tg - margin. Always includes 0."""
    cap = max_len - pp - tg - margin
    if cap < 4096:
        return [0]
    cap = (cap // rung) * rung
    ladder = [0]
    d = 4096
    while d <= cap:
        ladder.append(d)
        d *= 2
    if cap - ladder[-1] >= 4096:
        ladder.append(cap)
    return ladder


def conc_ladder(max_seqs: int) -> list[int]:
    """Concurrency rungs covering 1..max_seqs: powers of two plus the max."""
    if max_seqs <= 1:
        return [1]
    levels = [1]
    c = 2
    while c < max_seqs:
        levels.append(c)
        c *= 2
    levels.append(max_seqs)
    return levels


def fit_prompt(request_tokens: int, max_len: int | None, margin: int = 2048) -> int:
    """Largest prompt size <= request_tokens that leaves `margin` headroom
    under max_len (for the completion + template overhead)."""
    if max_len is None:
        return request_tokens
    return max(1024, min(request_tokens, max_len - margin))
