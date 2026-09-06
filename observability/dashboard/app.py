#!/usr/bin/env python3
"""Lightweight vLLM performance dashboard backend.

Proxies vLLM Prometheus metrics, serves history from history.jsonl, and
optionally runs llama-benchy on demand (POST /api/bench).
"""
import asyncio
import json
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8180")
HISTORY_PATH = Path(os.environ.get("HISTORY_PATH", "/app/history.jsonl"))
DEPTH_HISTORY_PATH = Path(os.environ.get("DEPTH_HISTORY_PATH", "/app/depth_history.jsonl"))
CONC_HISTORY_PATH = Path(os.environ.get("CONC_HISTORY_PATH", "/app/conc_history.jsonl"))
# Host path when running outside Docker (repo/observability/history.jsonl)
HOST_HISTORY = Path(__file__).resolve().parents[1] / "history.jsonl"
HOST_DEPTH_HISTORY = Path(__file__).resolve().parents[1] / "depth_history.jsonl"
HOST_CONC_HISTORY = Path(__file__).resolve().parents[1] / "conc_history.jsonl"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="r9700 dashboard")

# In-memory ring for sparklines (1s poll -> 600 points = 10 min)
RING_SIZE = 600
ring: deque[dict[str, Any]] = deque(maxlen=RING_SIZE)

# Bench state
bench_lock = asyncio.Lock()
bench_state: dict[str, Any] = {"running": False, "last": None, "log": ""}
bench_proc: subprocess.Popen | None = None

# Depth sweep state (full 0-200K corpus, tg1024; 256K exceeds 262144 with overhead)
depth_lock = asyncio.Lock()
depth_state: dict[str, Any] = {"running": False, "last": None, "log": "", "progress": ""}
depth_proc: subprocess.Popen | None = None
DEPTHS = [0, 4096, 8192, 16384, 32768, 65536, 128000, 200000]

# Concurrency sweep state (corpus up to max_num_seqs)
conc_lock = asyncio.Lock()
conc_state: dict[str, Any] = {"running": False, "last": None, "log": "", "progress": ""}
conc_proc: subprocess.Popen | None = None
HISTORY_LIMIT = 20


def _resolve_history() -> Path:
    # Prefer mounted path if exists, else fallback
    if HISTORY_PATH.exists() or str(HISTORY_PATH).startswith("/app"):
        # Ensure parent exists
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return HISTORY_PATH
    return HOST_HISTORY


def _resolve_depth_history() -> Path:
    if DEPTH_HISTORY_PATH.exists() or str(DEPTH_HISTORY_PATH).startswith("/app"):
        try:
            DEPTH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return DEPTH_HISTORY_PATH
    return HOST_DEPTH_HISTORY


def _resolve_conc_history() -> Path:
    if CONC_HISTORY_PATH.exists() or str(CONC_HISTORY_PATH).startswith("/app"):
        try:
            CONC_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return CONC_HISTORY_PATH
    return HOST_CONC_HISTORY


def _append_history_limited(path: Path, record: dict[str, Any], limit: int = HISTORY_LIMIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    # truncate to last `limit` lines
    try:
        lines = open(path).read().strip().split("\n")
        if len(lines) > limit:
            open(path, "w").write("\n".join(lines[-limit:]) + "\n")
    except Exception:
        pass


def _delete_history_by_ts(path: Path, host_path: Path, ts: float) -> bool:
    # Delete single row by ts (exact float match, also string compare for safety)
    candidates = [path, host_path] if path != host_path else [path]
    deleted = False
    for p in candidates:
        if not p.exists():
            continue
        try:
            lines = open(p).read().strip().split("\n")
            new_lines = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    # match ts within 1ms or exact
                    rec_ts = rec.get("ts")
                    if rec_ts is not None and abs(float(rec_ts) - float(ts)) < 0.001:
                        deleted = True
                        continue
                except Exception:
                    pass
                new_lines.append(line)
            if deleted or len(new_lines) != len(lines):
                open(p, "w").write("\n".join(new_lines) + ("\n" if new_lines else ""))
                deleted = True
        except Exception:
            pass
    return deleted


def _get_max_concurrency() -> int:
    # From env (dashboard inherits same env_file stack as vllm)
    for key in ("VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_SEQS_PER_REQUEST"):
        v = os.environ.get(key)
        if v and v.isdigit():
            try:
                return max(1, int(v))
            except Exception:
                pass
    # Fallback: try last metrics ring or fetch live
    for src in (ring,):
        if src:
            for item in reversed(src):
                if "num_gpu_blocks" in item:
                    # not concurrency, skip
                    pass
    return 2


def _parse_metrics(text: str) -> dict[str, Any]:
    """Extract key vLLM metrics from Prometheus text exposition."""
    out: dict[str, Any] = {}
    # Simple line parser: metric{labels} value
    # We collect last value per metric name (labels ignored except cache_config_info)
    def last_value(name: str) -> float | None:
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    val = float(line.rsplit(" ", 1)[-1])
                    out[name] = val
                except ValueError:
                    pass
        return out.get(name)

    last_value("vllm:kv_cache_usage_perc")
    last_value("vllm:num_requests_running")
    last_value("vllm:num_requests_waiting")
    last_value("vllm:prefix_cache_queries_total")
    last_value("vllm:prefix_cache_hits_total")
    last_value("vllm:num_requests_swapped")
    last_value("vllm:gpu_cache_usage_perc")

    # cache_config_info is a gauge with labels block_size, mamba_cache_mode etc.
    for line in text.splitlines():
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
            # aliases for frontend convenience
            if "kv_cache_size_tokens" in out:
                try:
                    out["kv_cache_size_tokens"] = int(str(out["kv_cache_size_tokens"]))
                except Exception:
                    pass
            if "num_gpu_blocks" in out:
                try:
                    out["num_gpu_blocks"] = int(str(out["num_gpu_blocks"]))
                except Exception:
                    pass
            break
    return out


@app.get("/api/metrics")
async def api_metrics():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{VLLM_URL}/metrics")
            r.raise_for_status()
            text = r.text
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"vllm metrics fetch failed: {e}")

    parsed = _parse_metrics(text)
    # derived
    q = parsed.get("vllm:prefix_cache_queries_total")
    h = parsed.get("vllm:prefix_cache_hits_total")
    if q is not None and h is not None and q > 0:
        parsed["prefix_hit_pct"] = (h / q) * 100.0
    else:
        # may be zero at start
        if q is not None and h is not None:
            parsed["prefix_hit_pct"] = 0.0

    parsed["ts"] = time.time()
    # also probe health quickly
    parsed["vllm_up"] = True
    ring.append(parsed)
    return JSONResponse(parsed)


@app.get("/api/history")
async def api_history():
    p = _resolve_history()
    # Also check host fallback if primary empty
    candidates = [p, HOST_HISTORY] if p != HOST_HISTORY else [p]
    for cand in candidates:
        if cand.exists():
            p = cand
            break
    if not p.exists():
        return JSONResponse([])
    items: list[dict[str, Any]] = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # strip heavy raw/stderr from list view to keep payload small
                    # full record available via /api/history/download/{ts}
                    if "raw" in rec:
                        rec = {**rec, "raw_len": len(rec.get("raw") or ""), "raw": (rec["raw"][:500] + "…") if len(rec.get("raw") or "") > 500 else rec.get("raw")}
                    if "stderr" in rec and rec["stderr"] and len(rec["stderr"]) > 500:
                        rec = {**rec, "stderr_len": len(rec["stderr"]), "stderr": rec["stderr"][:500] + "…"}
                    items.append(rec)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return JSONResponse([])
    return JSONResponse(items)


@app.get("/api/history/download/{ts}")
async def api_history_download(ts: str):
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p = _resolve_history()
    candidates = [p, HOST_HISTORY] if p != HOST_HISTORY else [p]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            with open(cand) as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        rec=json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts") is not None and abs(float(rec["ts"])-ts_f) < 0.001:
                        fname = f"bench-{ts}.json"
                        tmp = Path(f"/tmp/{fname}")
                        tmp.write_text(json.dumps(rec, indent=2))
                        return FileResponse(str(tmp), filename=fname, media_type="application/json")
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="not found")


@app.post("/api/history/clear")
async def api_history_clear():
    p = _resolve_history()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        open(p, "w").close()
        # also clear host fallback if different
        if HOST_HISTORY != p and HOST_HISTORY.exists():
            try:
                open(HOST_HISTORY, "w").close()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear failed: {e}")
    bench_state["last"] = None
    # keep log but note clear
    bench_state["log"] = "[history cleared]\n"
    return JSONResponse({"cleared": True})


@app.post("/api/history/delete")
async def api_history_delete(payload: dict):
    ts = payload.get("ts")
    if ts is None:
        raise HTTPException(status_code=400, detail="missing ts")
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p = _resolve_history()
    ok = _delete_history_by_ts(p, HOST_HISTORY, ts_f)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    # clear last if it was deleted
    if bench_state.get("last") and abs(float(bench_state["last"].get("ts", 0)) - ts_f) < 0.001:
        bench_state["last"] = None
    return JSONResponse({"deleted": True})


@app.get("/api/ring")
async def api_ring():
    return JSONResponse(list(ring))


@app.get("/api/health")
async def api_health():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{VLLM_URL}/health")
            # vLLM health returns 200 when ready
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
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{VLLM_URL}/v1/models")
                if r.status_code == 200:
                    data = r.json()
                    if data.get("data"):
                        model = data["data"][0].get("id", model)
        except Exception:
            pass
    return JSONResponse({"model": model, "kv_dtype": kv, "profile": profile, "vllm_url": VLLM_URL})


@app.get("/api/bench/status")
async def api_bench_status():
    return JSONResponse(bench_state)


@app.post("/api/bench")
async def api_bench():
    if bench_state["running"] or depth_state["running"] or conc_state["running"]:
        raise HTTPException(status_code=409, detail="another bench running")
    # Fire and forget background task
    asyncio.create_task(_run_bench())
    return JSONResponse({"started": True})


# Depth sweep (full 0-256K corpus)
@app.get("/api/depth/history")
async def api_depth_history():
    p = _resolve_depth_history()
    candidates = [p, HOST_DEPTH_HISTORY] if p != HOST_DEPTH_HISTORY else [p]
    for cand in candidates:
        if cand.exists():
            p = cand
            break
    if not p.exists():
        return JSONResponse([])
    items: list[dict[str, Any]] = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec=json.loads(line)
                    if "raw" in rec and rec["raw"] and len(rec["raw"])>500:
                        rec={**rec, "raw_len":len(rec["raw"]), "raw":rec["raw"][:500]+"…"}
                    if "stderr" in rec and rec["stderr"] and len(rec["stderr"])>500:
                        rec={**rec, "stderr_len":len(rec["stderr"]), "stderr":rec["stderr"][:500]+"…"}
                    items.append(rec)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return JSONResponse([])
    return JSONResponse(items)


@app.get("/api/depth/download/{ts}")
async def api_depth_download(ts: str):
    try:
        ts_f=float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p=_resolve_depth_history()
    candidates=[p, HOST_DEPTH_HISTORY] if p!=HOST_DEPTH_HISTORY else [p]
    for cand in candidates:
        if not cand.exists(): continue
        try:
            with open(cand) as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        rec=json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts") is not None and abs(float(rec["ts"])-ts_f)<0.001:
                        fname=f"depth-{ts}.json"
                        tmp=Path(f"/tmp/{fname}")
                        tmp.write_text(json.dumps(rec, indent=2))
                        return FileResponse(str(tmp), filename=fname, media_type="application/json")
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="not found")


@app.post("/api/depth/clear")
async def api_depth_clear():
    p = _resolve_depth_history()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        open(p, "w").close()
        if HOST_DEPTH_HISTORY != p and HOST_DEPTH_HISTORY.exists():
            try:
                open(HOST_DEPTH_HISTORY, "w").close()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear failed: {e}")
    depth_state["last"] = None
    depth_state["log"] = "[depth history cleared]\n"
    return JSONResponse({"cleared": True})


@app.post("/api/depth/delete")
async def api_depth_delete(payload: dict):
    ts = payload.get("ts")
    if ts is None:
        raise HTTPException(status_code=400, detail="missing ts")
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p = _resolve_depth_history()
    ok = _delete_history_by_ts(p, HOST_DEPTH_HISTORY, ts_f)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    if depth_state.get("last") and abs(float(depth_state["last"].get("ts", 0)) - ts_f) < 0.001:
        depth_state["last"] = None
    return JSONResponse({"deleted": True})


@app.get("/api/depth/status")
async def api_depth_status():
    return JSONResponse(depth_state)


@app.post("/api/depth")
async def api_depth():
    if bench_state["running"] or depth_state["running"] or conc_state["running"]:
        raise HTTPException(status_code=409, detail="another bench running")
    asyncio.create_task(_run_depth())
    return JSONResponse({"started": True})


@app.post("/api/depth/cancel")
async def api_depth_cancel():
    global depth_proc
    if not depth_state["running"]:
        raise HTTPException(status_code=400, detail="not running")
    depth_state["log"] += "\n[CANCEL requested]\n"
    proc = depth_proc
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            await asyncio.sleep(1)
            if proc.poll() is None:
                proc.kill()
        except Exception as e:
            depth_state["log"] += f"\n[cancel failed: {e}]"
    return JSONResponse({"cancelled": True})


# Concurrency sweep (corpus up to max_num_seqs)
@app.get("/api/conc/history")
async def api_conc_history():
    p = _resolve_conc_history()
    candidates = [p, HOST_CONC_HISTORY] if p != HOST_CONC_HISTORY else [p]
    for cand in candidates:
        if cand.exists():
            p = cand
            break
    if not p.exists():
        return JSONResponse([])
    items: list[dict[str, Any]] = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec=json.loads(line)
                    if "raw" in rec and rec["raw"] and len(rec["raw"])>500:
                        rec={**rec, "raw_len":len(rec["raw"]), "raw":rec["raw"][:500]+"…"}
                    if "stderr" in rec and rec["stderr"] and len(rec["stderr"])>500:
                        rec={**rec, "stderr_len":len(rec["stderr"]), "stderr":rec["stderr"][:500]+"…"}
                    items.append(rec)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return JSONResponse([])
    return JSONResponse(items)


@app.get("/api/conc/download/{ts}")
async def api_conc_download(ts: str):
    try:
        ts_f=float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p=_resolve_conc_history()
    candidates=[p, HOST_CONC_HISTORY] if p!=HOST_CONC_HISTORY else [p]
    for cand in candidates:
        if not cand.exists(): continue
        try:
            with open(cand) as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        rec=json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts") is not None and abs(float(rec["ts"])-ts_f)<0.001:
                        fname=f"conc-{ts}.json"
                        tmp=Path(f"/tmp/{fname}")
                        tmp.write_text(json.dumps(rec, indent=2))
                        return FileResponse(str(tmp), filename=fname, media_type="application/json")
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="not found")


@app.post("/api/conc/clear")
async def api_conc_clear():
    p = _resolve_conc_history()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        open(p, "w").close()
        if HOST_CONC_HISTORY != p and HOST_CONC_HISTORY.exists():
            try:
                open(HOST_CONC_HISTORY, "w").close()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clear failed: {e}")
    conc_state["last"] = None
    conc_state["log"] = "[conc history cleared]\n"
    return JSONResponse({"cleared": True})


@app.post("/api/conc/delete")
async def api_conc_delete(payload: dict):
    ts = payload.get("ts")
    if ts is None:
        raise HTTPException(status_code=400, detail="missing ts")
    try:
        ts_f = float(ts)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid ts")
    p = _resolve_conc_history()
    ok = _delete_history_by_ts(p, HOST_CONC_HISTORY, ts_f)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    if conc_state.get("last") and abs(float(conc_state["last"].get("ts", 0)) - ts_f) < 0.001:
        conc_state["last"] = None
    return JSONResponse({"deleted": True})


@app.get("/api/conc/status")
async def api_conc_status():
    return JSONResponse(conc_state)


@app.post("/api/conc")
async def api_conc():
    if bench_state["running"] or depth_state["running"] or conc_state["running"]:
        raise HTTPException(status_code=409, detail="another bench running")
    asyncio.create_task(_run_conc())
    return JSONResponse({"started": True})


@app.post("/api/conc/cancel")
async def api_conc_cancel():
    global conc_proc
    if not conc_state["running"]:
        raise HTTPException(status_code=400, detail="not running")
    conc_state["log"] += "\n[CANCEL requested]\n"
    proc = conc_proc
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            await asyncio.sleep(1)
            if proc.poll() is None:
                proc.kill()
        except Exception as e:
            conc_state["log"] += f"\n[cancel failed: {e}]"
    return JSONResponse({"cancelled": True})


async def _run_bench():
    async with bench_lock:
        if bench_state["running"]:
            return
        bench_state["running"] = True
        bench_state["log"] = ""
        bench_state["last"] = None
        start = time.time()
        try:
            # Discover model/tokenizer from vLLM /v1/models
            model = None
            tokenizer = None
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    r = await client.get(f"{VLLM_URL}/v1/models")
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("data"):
                            model = data["data"][0].get("id")
                            tokenizer = model
                except Exception:
                    pass
            if not model:
                model = os.environ.get("VLLM_SERVED_NAME", "qwen3.8-27b")
                tokenizer = model

            # Resolve tokenizer: served name like qwen3.8-27b is not a HF id.
            # Use a real Qwen tokenizer when model is qwen-family, else gpt2 fallback.
            # This avoids the 401 "Repository Not Found" warning that looks like a failure.
            if tokenizer in (None, model) and model and "qwen" in model.lower():
                # Qwen3 family tokenizer is compatible across sizes
                tokenizer = "Qwen/Qwen3-8B"
            elif tokenizer in (None, model):
                tokenizer = "gpt2"

            cmd = [
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
            bench_state["log"] += f"$ {' '.join(cmd)}\n"
            # Run synchronously in thread pool (subprocess is blocking)
            def _run():
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                return proc

            proc = await asyncio.to_thread(_run)
            bench_state["log"] += proc.stdout[-8000:] + "\n" + proc.stderr[-4000:]
            elapsed = time.time() - start
            try:
                result = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else None
                # llama-benchy json shape may vary; keep raw
                if result is None:
                    # try to find json blob in output
                    m = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
                    if m:
                        try:
                            result = json.loads(m.group(0))
                        except Exception:
                            result = None
                record: dict[str, Any] = {
                    "ts": time.time(),
                    "elapsed": elapsed,
                    "model": model,
                    "returncode": proc.returncode,
                    "raw": proc.stdout[:8000],
                    "stderr": proc.stderr[:3000],
                }
                if isinstance(result, dict):
                    record.update(result)
                    # Normalize for frontend: benchmarks is the canonical shape
                    # Bench history chart expects pp2048/tg32/tg128 top-level.
                    try:
                        bms = result.get("benchmarks") or []
                        for b in bms:
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
                                # pp for tg128 run is also pp2048 but keep first pp as canonical
                                if "pp2048" not in record and pp_m is not None:
                                    record["pp2048"] = pp_m
                                if tg_m is not None:
                                    record["tg128"] = tg_m
                            # also stash ttft for display if needed
                            if rs == 32 and b.get("e2e_ttft"):
                                record["ttft32"] = b["e2e_ttft"].get("mean")
                            if rs == 128 and b.get("e2e_ttft"):
                                record["ttft128"] = b["e2e_ttft"].get("mean")
                    except Exception as e:
                        bench_state["log"] += f"\n[normalize failed: {e}]"
                # Append to history (bind-mounted to host) limited to 20
                p = _resolve_history()
                try:
                    _append_history_limited(p, record, HISTORY_LIMIT)
                except Exception as e:
                    bench_state["log"] += f"\n[history write failed {p}: {e}]"
                bench_state["last"] = record
            except Exception as e:
                bench_state["log"] += f"\n[bench parse failed: {e}]"
                bench_state["last"] = {"ts": time.time(), "error": str(e), "raw": proc.stdout[:4000]}
            if proc.returncode != 0:
                bench_state["log"] += f"\n[exit {proc.returncode}]"
        except subprocess.TimeoutExpired:
            bench_state["log"] += "\n[TIMEOUT after 600s]"
        except Exception as e:
            bench_state["log"] += f"\n[bench failed: {e}]"
        finally:
            bench_state["running"] = False


async def _run_depth():
    global depth_proc
    async with depth_lock:
        if depth_state["running"]:
            return
        depth_state["running"] = True
        depth_state["log"] = ""
        depth_state["last"] = None
        depth_state["progress"] = ""
        start = time.time()
        try:
            model = None
            tokenizer = None
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    r = await client.get(f"{VLLM_URL}/v1/models")
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("data"):
                            model = data["data"][0].get("id")
                            tokenizer = model
                except Exception:
                    pass
            if not model:
                model = os.environ.get("VLLM_SERVED_NAME", "qwen3.8-27b")
                tokenizer = model
            if tokenizer in (None, model) and model and "qwen" in model.lower():
                tokenizer = "Qwen/Qwen3-8B"
            elif tokenizer in (None, model):
                tokenizer = "gpt2"
            # Full depth sweep: 0-200K book corpus, tg1024, --no-cache per depth md
            depths_str = " ".join(str(d) for d in DEPTHS)
            cmd = [
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
            depth_state["log"] += f"$ {' '.join(cmd)}\n"
            depth_state["progress"] = f"depths {depths_str} starting..."

            def _run():
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return proc

            # Use Popen to allow cancellation
            proc = await asyncio.to_thread(_run)
            depth_proc = proc
            # Wait with polling for cancel
            try:
                # Block in thread
                def _wait():
                    out, err = proc.communicate(timeout=3600)
                    return proc.returncode, out, err
                returncode, stdout, stderr = await asyncio.to_thread(_wait)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                returncode = proc.returncode
                depth_state["log"] += "\n[TIMEOUT after 3600s]"
                stdout = stdout or ""
                stderr = stderr or ""
            else:
                stdout = stdout or ""
                stderr = stderr or ""

            depth_proc = None
            # Check if cancelled
            if "CANCEL requested" in depth_state["log"]:
                depth_state["log"] += f"\n[CANCELLED] returncode {returncode}\n"
                depth_state["log"] += stdout[-8000:] + "\n" + stderr[-4000:]
                depth_state["last"] = {"ts": time.time(), "cancelled": True, "elapsed": time.time() - start, "model": model}
                depth_state["running"] = False
                return

            depth_state["log"] += stdout[-8000:] + "\n" + stderr[-3000:]
            elapsed = time.time() - start
            # Parse JSON
            result = None
            try:
                result = json.loads(stdout) if stdout.strip().startswith("{") else None
                if result is None:
                    m = re.search(r"\{.*\}", stdout, re.DOTALL)
                    if m:
                        result = json.loads(m.group(0))
            except Exception as e:
                depth_state["log"] += f"\n[parse failed: {e}]"

            record: dict[str, Any] = {
                "ts": time.time(),
                "elapsed": elapsed,
                "model": model,
                "depths": DEPTHS,
                "returncode": returncode,
                "raw": stdout[:8000],
                "stderr": stderr[:3000],
            }
            if isinstance(result, dict):
                record.update(result)
                # Normalize per-depth pp/tg/ttft for table
                try:
                    bms = result.get("benchmarks") or []
                    norm = []
                    for b in bms:
                        norm.append({
                            "depth": b.get("depth", b.get("context_size", 0)),
                            "pp": (b.get("pp_throughput") or {}).get("mean"),
                            "tg": (b.get("tg_throughput") or {}).get("mean"),
                            "ttft": (b.get("e2e_ttft") or {}).get("mean"),
                            "raw": b,
                        })
                    if norm:
                        record["depth_results"] = norm
                except Exception as e:
                    depth_state["log"] += f"\n[normalize failed: {e}]"
            p = _resolve_depth_history()
            try:
                _append_history_limited(p, record, HISTORY_LIMIT)
            except Exception as e:
                depth_state["log"] += f"\n[history write failed {p}: {e}]"
            depth_state["last"] = record
            if returncode not in (0, None):
                depth_state["log"] += f"\n[exit {returncode}]"
        except Exception as e:
            depth_state["log"] += f"\n[depth failed: {e}]"
        finally:
            depth_state["running"] = False
            depth_proc = None


async def _run_conc():
    global conc_proc
    async with conc_lock:
        if conc_state["running"]:
            return
        conc_state["running"] = True
        conc_state["log"] = ""
        conc_state["last"] = None
        conc_state["progress"] = ""
        start = time.time()
        try:
            model = None
            tokenizer = None
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    r = await client.get(f"{VLLM_URL}/v1/models")
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("data"):
                            model = data["data"][0].get("id")
                            tokenizer = model
                except Exception:
                    pass
            if not model:
                model = os.environ.get("VLLM_SERVED_NAME", "qwen3.8-27b")
                tokenizer = model
            if tokenizer in (None, model) and model and "qwen" in model.lower():
                tokenizer = "Qwen/Qwen3-8B"
            elif tokenizer in (None, model):
                tokenizer = "gpt2"
            max_conc = _get_max_concurrency()
            # Concurrency sweep = same depth sweep (0-200K) but with x parallel runs (x = max_num_seqs)
            conc_str = str(max_conc)
            cmd = [
                "uvx", "llama-benchy@0.4.0",
                "--base-url", f"{VLLM_URL}/v1",
                "--model", model,
                "--tokenizer", tokenizer,
                "--pp", "2048",
                "--tg", "1024",
                "--depth", *[str(d) for d in DEPTHS],
                "--concurrency", str(max_conc),
                "--runs", "2",
                "--no-cache",
                "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
                "--format", "json",
            ]
            conc_state["log"] += f"$ {' '.join(cmd)}\n"
            conc_state["progress"] = f"depths { ' '.join(str(d) for d in DEPTHS)} @ concurrency {max_conc} starting..."

            def _run():
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                return proc

            proc = await asyncio.to_thread(_run)
            conc_proc = proc
            try:
                def _wait():
                    out, err = proc.communicate(timeout=3600)
                    return proc.returncode, out, err
                returncode, stdout, stderr = await asyncio.to_thread(_wait)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                returncode = proc.returncode
                conc_state["log"] += "\n[TIMEOUT after 3600s]"
                stdout = stdout or ""
                stderr = stderr or ""
            else:
                stdout = stdout or ""
                stderr = stderr or ""

            conc_proc = None
            if "CANCEL requested" in conc_state["log"]:
                conc_state["log"] += f"\n[CANCELLED] returncode {returncode}\n"
                conc_state["log"] += stdout[-8000:] + "\n" + stderr[-4000:]
                conc_state["last"] = {"ts": time.time(), "cancelled": True, "elapsed": time.time() - start, "model": model}
                conc_state["running"] = False
                return

            conc_state["log"] += stdout[-8000:] + "\n" + stderr[-3000:]
            elapsed = time.time() - start
            result = None
            try:
                result = json.loads(stdout) if stdout.strip().startswith("{") else None
                if result is None:
                    m = re.search(r"\{.*\}", stdout, re.DOTALL)
                    if m:
                        result = json.loads(m.group(0))
            except Exception as e:
                conc_state["log"] += f"\n[parse failed: {e}]"

            record: dict[str, Any] = {
                "ts": time.time(),
                "elapsed": elapsed,
                "model": model,
                "depths": DEPTHS,
                "concurrency": max_conc,
                "returncode": returncode,
                "raw": stdout[:8000],
                "stderr": stderr[:3000],
            }
            if isinstance(result, dict):
                record.update(result)
                try:
                    bms = result.get("benchmarks") or []
                    norm = []
                    for b in bms:
                        norm.append({
                            "depth": b.get("depth", b.get("context_size", 0)),
                            "concurrency": b.get("concurrency", max_conc),
                            "pp": (b.get("pp_throughput") or {}).get("mean"),
                            "tg": (b.get("tg_throughput") or {}).get("mean"),
                            "ttft": (b.get("e2e_ttft") or {}).get("mean"),
                            "raw": b,
                        })
                    if norm:
                        record["conc_results"] = norm
                except Exception as e:
                    conc_state["log"] += f"\n[normalize failed: {e}]"
            p = _resolve_conc_history()
            try:
                _append_history_limited(p, record, HISTORY_LIMIT)
            except Exception as e:
                conc_state["log"] += f"\n[history write failed {p}: {e}]"
            conc_state["last"] = record
            if returncode not in (0, None):
                conc_state["log"] += f"\n[exit {returncode}]"
        except Exception as e:
            conc_state["log"] += f"\n[conc failed: {e}]"
        finally:
            conc_state["running"] = False
            conc_proc = None


# Static — mount last so /api/* takes precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/")
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"status": "dashboard up", "vllm": VLLM_URL})
