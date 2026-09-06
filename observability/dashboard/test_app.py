"""Unit tests for the dashboard's pure helpers (no network, no container).

Run: uvx pytest observability/dashboard/test_app.py -q
"""
import json
from pathlib import Path

import pytest

from app import (
    _append_history_limited,
    _delete_history_by_ts,
    _find_record,
    _history_etag,
    _parse_metrics,
    _read_history,
    _resolve_history_path,
    _serve_download,
)

METRICS = """# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.42
vllm:num_requests_running 2
vllm:num_requests_waiting 0
vllm:num_requests_swapped 0
vllm:gpu_cache_usage_perc 0.42
vllm:prefix_cache_queries_total 1000.0
vllm:prefix_cache_hits_total 560.0
vllm:prompt_tokens_total{engine="0",model_name="qwen3.8-27b"} 19275232.0
vllm:generation_tokens_total{engine="0",model_name="qwen3.8-27b"} 161412.0
vllm:cache_config_info{block_size="832",mamba_cache_mode="align",cache_dtype="bfloat16",kv_cache_size_tokens="619479",num_gpu_blocks="745",gpu_memory_utilization="0.95"} 1.0
"""


def test_parse_metrics_scalars():
    m = _parse_metrics(METRICS)
    assert m["vllm:kv_cache_usage_perc"] == pytest.approx(0.42)
    assert m["vllm:num_requests_running"] == 2
    assert m["vllm:num_requests_waiting"] == 0
    assert m["vllm:prefix_cache_queries_total"] == 1000.0
    assert m["vllm:prefix_cache_hits_total"] == 560.0


def test_parse_metrics_token_counters():
    # labeled counters (engine/model_name) parse to the last value per name
    m = _parse_metrics(METRICS)
    assert m["vllm:prompt_tokens_total"] == 19275232.0
    assert m["vllm:generation_tokens_total"] == 161412.0
    # bucket lines must not shadow the plain counters
    assert "vllm:request_prompt_tokens" not in m


def test_parse_metrics_prefix_counters():
    # prefix_hit_pct is derived in the /api/metrics endpoint, not the parser
    m = _parse_metrics(METRICS)
    assert m["vllm:prefix_cache_queries_total"] == 1000.0
    assert m["vllm:prefix_cache_hits_total"] == 560.0
    # same derivation as api_metrics: h/q*100 when q>0, else 0.0
    q, h = m["vllm:prefix_cache_queries_total"], m["vllm:prefix_cache_hits_total"]
    pct = (h / q) * 100.0 if q > 0 else 0.0
    assert pct == pytest.approx(56.0)


def test_parse_metrics_cache_config_info():
    m = _parse_metrics(METRICS)
    assert m["block_size"] == 832
    assert m["mamba_cache_mode"] == "align"
    assert m["cache_dtype"] == "bfloat16"
    assert m["kv_cache_size_tokens"] == 619479
    assert m["num_gpu_blocks"] == 745
    assert m["gpu_memory_utilization"] == 0.95


def test_append_history_limited_truncates(tmp_path):
    p = tmp_path / "h.jsonl"
    for i in range(25):
        _append_history_limited(p, {"ts": float(i), "v": i}, limit=20)
    lines = [json.loads(l) for l in p.read_text().strip().split("\n")]
    assert len(lines) == 20
    # newest kept: ts 5..24
    assert [r["ts"] for r in lines] == [float(i) for i in range(5, 25)]


def test_delete_history_by_ts(tmp_path):
    p = tmp_path / "h.jsonl"
    for i in range(5):
        _append_history_limited(p, {"ts": float(i)})
    assert _delete_history_by_ts(p, tmp_path / "nope.jsonl", 3.0) is True
    lines = [json.loads(l) for l in p.read_text().strip().split("\n")]
    assert [r["ts"] for r in lines] == [0.0, 1.0, 2.0, 4.0]
    assert _delete_history_by_ts(p, tmp_path / "nope.jsonl", 99.0) is False


def test_read_history_truncates_raw(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({"ts": 1.0, "model": "m", "raw": "x" * 1000}) + "\n")
    items = _read_history(p, tmp_path / "nope.jsonl")
    assert len(items) == 1
    assert items[0]["raw_len"] == 1000
    assert items[0]["raw"].endswith("…")
    assert len(items[0]["raw"]) == 501
    # record stays findable by ts for downloads
    assert _find_record(p, tmp_path / "nope.jsonl", 1.0)["model"] == "m"
    assert _find_record(p, tmp_path / "nope.jsonl", 2.0) is None


def test_serve_download_returns_attachment():
    # served in-memory as an attachment (no /tmp file — a fixed filename
    # would race between concurrent downloads)
    rec = {"ts": 1.5, "model": "m", "pp2048": 3000.0}
    resp = _serve_download(rec, "bench")
    assert resp.status_code == 200
    assert resp.media_type == "application/json"
    assert resp.headers["content-disposition"] == 'attachment; filename="bench-1.5.json"'
    assert json.loads(resp.body)["model"] == "m"


def test_resolve_history_path_prefers_mounted(tmp_path):
    # /app/* paths are assumed to be the bind mount inside the container
    mounted = tmp_path / "app" / "history.jsonl"
    assert _resolve_history_path(Path("/app/history.jsonl"), tmp_path / "host.jsonl") == Path("/app/history.jsonl")
    # an existing container path wins
    mounted.parent.mkdir(parents=True, exist_ok=True)
    mounted.write_text("{}\n")
    assert _resolve_history_path(mounted, tmp_path / "host.jsonl") == mounted
    # a non-/app path that does not exist falls back to the host file
    assert _resolve_history_path(tmp_path / "missing" / "h.jsonl", tmp_path / "host.jsonl") == tmp_path / "host.jsonl"


def test_history_etag(tmp_path):
    p = tmp_path / "h.jsonl"
    assert _history_etag(p) == '""'
    p.write_text("{}\n")
    first = _history_etag(p)
    assert first != '""'
    p.write_text("{}\n{}\n")
    assert _history_etag(p) != first  # size/mtime changed
