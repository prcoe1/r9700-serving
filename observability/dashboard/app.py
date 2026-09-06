#!/usr/bin/env python3
"""Lightweight vLLM performance dashboard backend.

Proxies vLLM Prometheus metrics, serves history from *.jsonl files, and
runs llama-benchy sweeps on demand:
  POST /api/bench  — pp2048 + tg32/128 ×3 (~1-4 min)
  POST /api/depth  — depth sweep 0-200K, tg1024, --no-cache (~20 min)
  POST /api/conc   — same depths at --concurrency max_num_seqs (~40 min)
Each is cancellable (POST /api/{kind}/cancel) with a live streaming log.
"""
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8180")
HISTORY_PATH = Path(os.environ.get("HISTORY_PATH", "/app/history.jsonl"))
DEPTH_HISTORY_PATH = Path(os.environ.get("DEPTH_HISTORY_PATH", "/app/depth_history.jsonl"))
CONC_HISTORY_PATH = Path(os.environ.get("CONC_HISTORY_PATH", "/app/conc_history.jsonl"))
# Host path when running outside Docker (repo/observability/*.jsonl)
HOST_HISTORY = Path(__file__).resolve().parents[1] / "history.jsonl"
HOST_DEPTH_HISTORY = Path(__file__).resolve().parents[1] / "depth_history.jsonl"
HOST_CONC_HISTORY = Path(__file__).resolve().parents[1] / "conc_history.jsonl"
STATIC_DIR = Path(__file__).parent / "static"

HISTORY_LIMIT = 20
# In-memory live-log cap (chars); the client renders the tail.
LOG_CAP = 16000
# Tails kept for the history record: last N chars of stdout / all stderr tail.
RAW_CAP = 8000
STDERR_CAP = 4000
# Depth sweep (full 0-200K corpus; 256K exceeds 262144 with overhead)
DEPTHS = [0, 4096, 8192, 16384, 32768, 65536, 128000, 200000]

app = FastAPI(title="r9700 dashboard")

# One shared async client (the dashboard is a single small process).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10)
    return _client


@app.middleware("http")
async def no_store(request: Request, call_next):
    # Live-poll endpoints must never be heuristically cached by browsers.
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# History files (JSONL, one record per line, capped at HISTORY_LIMIT)
# ---------------------------------------------------------------------------

SWEEP_HISTORY: dict[str, tuple[Path, Path]] = {
    "bench": (HISTORY_PATH, HOST_HISTORY),
    "depth": (DEPTH_HISTORY_PATH, HOST_DEPTH_HISTORY),
    "conc": (CONC_HISTORY_PATH, HOST_CONC_HISTORY),
}


def _resolve_history_path(container: Path, host: Path) -> Path:
    # Prefer the bind-mounted path inside Docker; fall back to the repo path
    # when running the app on the host.
    if container.exists() or str(container).startswith("/app"):
        try:
            container.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return container
    return host


def _append_history_limited(path: Path, record: dict[str, Any], limit: int = HISTORY_LIMIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    # truncate to last `limit` lines
    try:
        with open(path) as f:
            lines = f.read().strip().split("\n")
        if len(lines) > limit:
            with open(path, "w") as f:
                f.write("\n".join(lines[-limit:]) + "\n")
    except Exception:
        pass


def _history_candidates(path: Path, host_path: Path) -> list[Path]:
    return [path, host_path] if path != host_path else [path]


def _read_history(path: Path, host_path: Path) -> list[dict[str, Any]]:
    # List view: strip heavy raw/stderr payloads (full record available via
    # the per-section download endpoint) to keep the UI responsive.
    for p in _history_candidates(path, host_path):
        if not p.exists():
            continue
        items: list[dict[str, Any]] = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = rec.get("raw")
                if isinstance(raw, str) and len(raw) > 500:
                    rec = {**rec, "raw_len": len(raw), "raw": raw[:500] + "…"}
                err = rec.get("stderr")
                if isinstance(err, str) and len(err) > 500:
                    rec = {**rec, "stderr_len": len(err), "stderr": err[:500] + "…"}
                items.append(rec)
        return items
    return []


def _find_record(path: Path, host_path: Path, ts_f: float):
    for p in _history_candidates(path, host_path):
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("ts") is not None and abs(float(rec["ts"]) - ts_f) < 0.001:
                    return rec
    return None


def _serve_download(rec: dict[str, Any], prefix: str) -> Response:
    # Serve the full record as an attachment without touching disk (a fixed
    # /tmp filename would race between concurrent downloads).
    fname = f"{prefix}-{rec.get('ts')}.json"
    return Response(
        content=json.dumps(rec, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _delete_history_by_ts(path: Path, host_path: Path, ts: float) -> bool:
    # Delete single row by ts (float compare with a 1ms tolerance)
    for p in _history_candidates(path, host_path):
        if not p.exists():
            continue
        try:
            with open(p) as f:
                lines = f.read().strip().split("\n")
            new_lines = []
            deleted = False
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec_ts = rec.get("ts")
                    if rec_ts is not None and abs(float(rec_ts) - float(ts)) < 0.001:
                        deleted = True
                        continue
                except Exception:
                    pass
                new_lines.append(line)
            if deleted:
                with open(p, "w") as f:
                    f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
                return True
        except Exception:
            pass
    return False


def _history_etag(path: Path) -> str:
    try:
        st = path.stat()
    except FileNotFoundError:
        return '""'
    return f'"{st.st_size}:{st.st_mtime_ns}"'


def _history_response(request: Request, kind: str) -> Response:
    path, host = SWEEP_HISTORY[kind]
    p = _resolve_history_path(path, host)
    etag = _history_etag(p)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    return JSONResponse(_read_history(p, host), headers={"ETag": etag})


def _clear_sweep_history(kind: str) -> JSONResponse:
    path, host = SWEEP_HISTORY[kind]
    p = _resolve_history_path(path, host)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w"):
            pass
        if host != p and host.exists():
            with open(host, "w"):
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear failed: {e}")
    state = SWEEP_STATES[kind]
    state["last"] = None
    state["log"] = f"[{kind} history cleared]\n"
    return JSONResponse({"cleared": True})


def _delete_sweep_record(kind: str, payload: dict) -> JSONResponse:
    ts = payload.get("ts")
    if ts is None:
        raise HTTPException(status_code=400, detail="missing ts")
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    path, host = SWEEP_HISTORY[kind]
    if not _delete_history_by_ts(_resolve_history_path(path, host), host, ts_f):
        raise HTTPException(status_code=404, detail="not found")
    state = SWEEP_STATES[kind]
    if state.get("last") and abs(float(state["last"].get("ts", 0)) - ts_f) < 0.001:
        state["last"] = None
    return JSONResponse({"deleted": True})


# ---------------------------------------------------------------------------
# Metrics proxy
# ---------------------------------------------------------------------------

METRIC_NAMES = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    # cumulative token counters — the UI derives t/s rates from their deltas
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def _parse_metrics(text: str) -> dict[str, Any]:
    """Extract key vLLM metrics from Prometheus text exposition (one pass)."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("vllm:cache_config_info{"):
            def _lbl(key: str):
                m = re.search(rf'{key}="([^"]+)"', line)
                return m.group(1) if m else None
            for k in ("block_size", "mamba_cache_mode", "cache_dtype", "kv_cache_size_tokens", "num_gpu_blocks", "gpu_memory_utilization"):
                v = _lbl(k)
                if v is not None:
                    # keep numeric where possible
                    try:
                        out[k] = int(v) if v.isdigit() else float(v) if re.match(r'^-?\d+\.\d+$', v) else v
                    except ValueError:
                        out[k] = v
            continue
        for name in METRIC_NAMES:
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    out[name] = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    pass
                break
    return out


@app.get("/api/metrics")
async def api_metrics():
    try:
        r = await _get_client().get(f"{VLLM_URL}/metrics", timeout=5)
        r.raise_for_status()
        text = r.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vllm metrics fetch failed: {e}")

    parsed = _parse_metrics(text)
    # derived
    q = parsed.get("vllm:prefix_cache_queries_total")
    h = parsed.get("vllm:prefix_cache_hits_total")
    if q is not None and h is not None:
        parsed["prefix_hit_pct"] = (h / q) * 100.0 if q > 0 else 0.0
    parsed["ts"] = time.time()
    # a successful /metrics fetch doubles as the liveness signal
    parsed["vllm_up"] = True
    return JSONResponse(parsed)


@app.get("/api/health")
async def api_health():
    # Kept for the compose healthcheck; the UI uses /api/metrics instead.
    try:
        r = await _get_client().get(f"{VLLM_URL}/health", timeout=5)
        return JSONResponse({"vllm_up": r.status_code == 200, "status": r.status_code, "body": r.text[:500]})
    except Exception as e:
        return JSONResponse({"vllm_up": False, "error": str(e)}, status_code=200)


@app.get("/api/info")
async def api_info():
    # Exposed for top-bar badges: model + kv dtype + profile
    model = os.environ.get("VLLM_SERVED_NAME") or os.environ.get("VLLM_MODEL") or ""
    kv = os.environ.get("VLLM_KV_CACHE_DTYPE") or ""
    profile = os.environ.get("MODEL_PROFILE") or ""
    # Try live model from vLLM if env not set
    if not model:
        try:
            r = await _get_client().get(f"{VLLM_URL}/v1/models")
            if r.status_code == 200:
                data = r.json()
                if data.get("data"):
                    model = data["data"][0].get("id", model)
        except Exception:
            pass
    return JSONResponse({"model": model, "kv_dtype": kv, "profile": profile, "vllm_url": VLLM_URL})


# ---------------------------------------------------------------------------
# Bench history (URL kept as /api/history for compatibility)
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def api_history(request: Request):
    return _history_response(request, "bench")


@app.get("/api/history/download/{ts}")
async def api_history_download(ts: str):
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    path, host = SWEEP_HISTORY["bench"]
    rec = _find_record(_resolve_history_path(path, host), host, ts_f)
    if rec is None:
        raise HTTPException(status_code=404, detail="not found")
    return _serve_download(rec, "bench")


@app.post("/api/history/clear")
async def api_history_clear():
    return _clear_sweep_history("bench")


@app.post("/api/history/delete")
async def api_history_delete(payload: dict):
    return _delete_sweep_record("bench", payload)


# ---------------------------------------------------------------------------
# Sweeps (bench / depth / conc) — one shared runner
# ---------------------------------------------------------------------------

SWEEP_STATES: dict[str, dict[str, Any]] = {
    "bench": {"running": False, "last": None, "log": "", "cancelled": False},
    "depth": {"running": False, "last": None, "log": "", "progress": "", "cancelled": False},
    "conc": {"running": False, "last": None, "log": "", "progress": "", "cancelled": False},
}
SWEEP_LOCKS: dict[str, asyncio.Lock] = {k: asyncio.Lock() for k in SWEEP_STATES}
sweep_procs: dict[str, subprocess.Popen | None] = {k: None for k in SWEEP_STATES}


def _get_max_concurrency() -> int:
    # From env (dashboard inherits the same env_file stack as vllm,
    # compose.yaml env_file), so this is authoritative when set.
    for key in ("VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_SEQS_PER_REQUEST"):
        v = os.environ.get(key)
        if v and v.isdigit():
            try:
                return max(1, int(v))
            except Exception:
                pass
    # No env set — match the compose.yaml --max-num-seqs default.
    print("[warn] VLLM_MAX_NUM_SEQS not set; assuming max concurrency 2", file=sys.stderr)
    return 2


def _bench_cmd(model: str, tokenizer: str) -> list[str]:
    return [
        "uvx", "llama-benchy@0.4.0",
        "--base-url", f"{VLLM_URL}/v1",
        "--model", model,
        "--tokenizer", tokenizer,
        "--pp", "2048",
        "--tg", "32", "128",
        "--runs", "3",
        "--enable-prefix-caching",
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--format", "json",
    ]


def _depth_cmd(model: str, tokenizer: str) -> list[str]:
    return [
        "uvx", "llama-benchy@0.4.0",
        "--base-url", f"{VLLM_URL}/v1",
        "--model", model,
        "--tokenizer", tokenizer,
        "--pp", "2048",
        "--tg", "1024",
        "--depth", *[str(d) for d in DEPTHS],
        "--runs", "2",
        "--no-cache",
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--format", "json",
    ]


def _conc_cmd(model: str, tokenizer: str) -> list[str]:
    return [
        "uvx", "llama-benchy@0.4.0",
        "--base-url", f"{VLLM_URL}/v1",
        "--model", model,
        "--tokenizer", tokenizer,
        "--pp", "2048",
        "--tg", "1024",
        "--depth", *[str(d) for d in DEPTHS],
        "--concurrency", str(_get_max_concurrency()),
        "--runs", "2",
        "--no-cache",
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--format", "json",
    ]


def _normalize_bench(record: dict[str, Any], result: dict[str, Any]) -> None:
    # Bench history chart/table expects pp2048/tg32/tg128 top-level.
    for b in (result.get("benchmarks") or []):
        ps = b.get("prompt_size")
        rs = b.get("response_size")
        pp_m = (b.get("pp_throughput") or {}).get("mean")
        tg_m = (b.get("tg_throughput") or {}).get("mean")
        if ps == 2048 and rs == 32:
            if pp_m is not None:
                record["pp2048"] = pp_m
            if tg_m is not None:
                record["tg32"] = tg_m
        elif ps == 2048 and rs == 128:
            # pp for the tg128 run is also pp2048; keep the first as canonical
            if "pp2048" not in record and pp_m is not None:
                record["pp2048"] = pp_m
            if tg_m is not None:
                record["tg128"] = tg_m
        if rs == 32 and b.get("e2e_ttft"):
            record["ttft32"] = b["e2e_ttft"].get("mean")
        if rs == 128 and b.get("e2e_ttft"):
            record["ttft128"] = b["e2e_ttft"].get("mean")


def _normalize_depth(record: dict[str, Any], result: dict[str, Any]) -> None:
    norm = [{
        "depth": b.get("depth", b.get("context_size", 0)),
        "pp": (b.get("pp_throughput") or {}).get("mean"),
        "tg": (b.get("tg_throughput") or {}).get("mean"),
        "ttft": (b.get("e2e_ttft") or {}).get("mean"),
        "raw": b,
    } for b in (result.get("benchmarks") or [])]
    if norm:
        record["depth_results"] = norm


def _normalize_conc(record: dict[str, Any], result: dict[str, Any]) -> None:
    fallback = record.get("concurrency")
    norm = [{
        "depth": b.get("depth", b.get("context_size", 0)),
        "concurrency": b.get("concurrency", fallback),
        "pp": (b.get("pp_throughput") or {}).get("mean"),
        "tg": (b.get("tg_throughput") or {}).get("mean"),
        "ttft": (b.get("e2e_ttft") or {}).get("mean"),
        "raw": b,
    } for b in (result.get("benchmarks") or [])]
    if norm:
        record["conc_results"] = norm


SWEEPS: dict[str, dict[str, Any]] = {
    "bench": {
        "timeout": 600,
        "cmd": _bench_cmd,
        "normalize": _normalize_bench,
        "record_extra": lambda model, tokenizer: {},
    },
    "depth": {
        "timeout": 3600,
        "cmd": _depth_cmd,
        "normalize": _normalize_depth,
        "record_extra": lambda model, tokenizer: {"depths": DEPTHS},
    },
    "conc": {
        "timeout": 3600,
        "cmd": _conc_cmd,
        "normalize": _normalize_conc,
        "record_extra": lambda model, tokenizer: {"depths": DEPTHS, "concurrency": _get_max_concurrency()},
    },
}


async def _discover_model_tokenizer() -> tuple[str, str]:
    # llama-benchy needs the served model id plus a real tokenizer. Served
    # names like qwen3.8-27b are not HF repo ids; the Qwen3 tokenizer is
    # compatible across sizes, gpt2 is the count-accurate fallback.
    model = None
    try:
        r = await _get_client().get(f"{VLLM_URL}/v1/models")
        if r.status_code == 200:
            data = r.json()
            if data.get("data"):
                model = data["data"][0].get("id")
    except Exception:
        pass
    if not model:
        model = os.environ.get("VLLM_SERVED_NAME") or "qwen3.8-27b"
    tokenizer = "Qwen/Qwen3-8B" if "qwen" in model.lower() else "gpt2"
    return model, tokenizer


def _popen_sweep(cmd: list[str]) -> subprocess.Popen:
    # start_new_session: the child leads its own process group so cancel/
    # timeout can kill uvx AND its grandchildren (HF downloads, workers) —
    # terminate() alone would leave orphans hammering vLLM after "cancel".
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )


def _kill_sweep_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except Exception:
            pass


def _stream_lines(proc: subprocess.Popen, state: dict[str, Any], sink: dict, key: str, cap: int | None) -> None:
    # Reader thread: mirror the stream into the live log (always capped to
    # LOG_CAP) and a rolling tail for the history record (cap=None: keep all).
    readline = proc.stdout.readline if key == "out" else proc.stderr.readline
    for line in iter(readline, ""):
        if cap is None:
            sink[key] += line
        else:
            sink[key] = (sink[key] + line)[-cap:]
        state["log"] = (state["log"] + line)[-LOG_CAP:]


async def _stream_and_wait(proc: subprocess.Popen, state: dict[str, Any], timeout_s: int):
    """Run proc, streaming output into state['log'] live; return
    (returncode, stdout_tail, stderr_tail, timed_out)."""
    tails: dict[str, str] = {"out": "", "err": ""}
    readers = [
        threading.Thread(target=_stream_lines, args=(proc, state, tails, "out", None), daemon=True),
        threading.Thread(target=_stream_lines, args=(proc, state, tails, "err", STDERR_CAP), daemon=True),
    ]
    for t in readers:
        t.start()
    timed_out = False
    try:
        await asyncio.to_thread(proc.wait, timeout_s)
    except subprocess.TimeoutExpired:
        _kill_sweep_group(proc, signal.SIGKILL)
        await asyncio.to_thread(proc.wait)
        timed_out = True
    for t in readers:
        t.join(timeout=5)
    return proc.returncode, tails["out"], tails["err"], timed_out


async def _run_sweep(kind: str) -> None:
    cfg = SWEEPS[kind]
    state = SWEEP_STATES[kind]
    async with SWEEP_LOCKS[kind]:
        if state["running"]:
            return
        state["running"] = True
        state["log"] = ""
        state["last"] = None
        state["cancelled"] = False
        if "progress" in state:
            state["progress"] = ""
        start = time.time()
        try:
            model, tokenizer = await _discover_model_tokenizer()
            cmd = cfg["cmd"](model, tokenizer)
            state["log"] += f"$ {' '.join(cmd)}\n"
            if "progress" in state:
                state["progress"] = "starting…"
            proc = await asyncio.to_thread(_popen_sweep, cmd)
            sweep_procs[kind] = proc
            try:
                returncode, out, err, timed_out = await _stream_and_wait(proc, state, cfg["timeout"])
            finally:
                sweep_procs[kind] = None
            if timed_out:
                state["log"] += f"\n[TIMEOUT after {cfg['timeout']}s]\n"
            if state["cancelled"]:
                state["log"] += f"\n[CANCELLED] returncode {returncode}\n"
                state["last"] = {"ts": time.time(), "cancelled": True, "elapsed": time.time() - start, "model": model}
                return
            # Parse the result JSON (llama-benchy prints it to stdout last)
            result = None
            try:
                stripped = out.strip()
                if stripped.startswith("{"):
                    result = json.loads(stripped)
                else:
                    m = re.search(r"\{.*\}", out, re.DOTALL)
                    if m:
                        result = json.loads(m.group(0))
            except Exception as e:
                state["log"] += f"\n[parse failed: {e}]\n"
            record: dict[str, Any] = {
                "ts": time.time(),
                "elapsed": time.time() - start,
                "model": model,
                "returncode": returncode,
                **cfg["record_extra"](model, tokenizer),
                "raw": out[-RAW_CAP:],
                "stderr": err,
            }
            if isinstance(result, dict):
                record.update(result)
                try:
                    cfg["normalize"](record, result)
                except Exception as e:
                    state["log"] += f"\n[normalize failed: {e}]\n"
            try:
                path, host = SWEEP_HISTORY[kind]
                _append_history_limited(_resolve_history_path(path, host), record)
            except Exception as e:
                state["log"] += f"\n[history write failed: {e}]\n"
            state["last"] = record
            if returncode not in (0, None):
                state["log"] += f"\n[exit {returncode}]\n"
        except Exception as e:
            state["log"] += f"\n[{kind} failed: {e}]\n"
        finally:
            state["running"] = False


async def _cancel_sweep(kind: str) -> JSONResponse:
    state = SWEEP_STATES[kind]
    if not state["running"]:
        raise HTTPException(status_code=400, detail="not running")
    state["cancelled"] = True
    state["log"] += "\n[CANCEL requested]\n"
    proc = sweep_procs.get(kind)
    if proc and proc.poll() is None:
        _kill_sweep_group(proc, signal.SIGTERM)
        await asyncio.sleep(1)
        if proc.poll() is None:
            _kill_sweep_group(proc, signal.SIGKILL)
    return JSONResponse({"cancelled": True})


@app.post("/api/{kind}")
async def api_start(kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    if any(s["running"] for s in SWEEP_STATES.values()):
        raise HTTPException(status_code=409, detail="another bench running")
    asyncio.create_task(_run_sweep(kind))
    return JSONResponse({"started": True})


@app.get("/api/{kind}/status")
async def api_status(kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    return JSONResponse(SWEEP_STATES[kind])


@app.post("/api/{kind}/cancel")
async def api_cancel(kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    return await _cancel_sweep(kind)


@app.get("/api/{kind}/history")
async def api_sweep_history(request: Request, kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    return _history_response(request, kind)


@app.get("/api/{kind}/download/{ts}")
async def api_sweep_download(ts: str, kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    path, host = SWEEP_HISTORY[kind]
    rec = _find_record(_resolve_history_path(path, host), host, ts_f)
    if rec is None:
        raise HTTPException(status_code=404, detail="not found")
    return _serve_download(rec, kind)


@app.post("/api/{kind}/clear")
async def api_sweep_clear(kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    return _clear_sweep_history(kind)


@app.post("/api/{kind}/delete")
async def api_sweep_delete(payload: dict, kind: str):
    if kind not in SWEEPS:
        raise HTTPException(status_code=404, detail="unknown sweep")
    return _delete_sweep_record(kind, payload)


# Static — mount last so /api/* routes (registered earlier) take precedence.
# StaticFiles(html=True) serves index.html for GET /.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
