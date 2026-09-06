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
    _parse_metrics,
    _read_history,
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
vllm:cache_config_info{block_size="832",mamba_cache_mode="align",cache_dtype="bfloat16",kv_cache_size_tokens="619479",num_gpu_blocks="745",gpu_memory_utilization="0.95"} 1.0
"""


def test_parse_metrics_scalars():
    m = _parse_metrics(METRICS)
    assert m["vllm:kv_cache_usage_perc"] == pytest.approx(0.42)
    assert m["vllm:num_requests_running"] == 2
    assert m["vllm:num_requests_waiting"] == 0
    assert m["vllm:prefix_cache_queries_total"] == 1000.0
    assert m["vllm:prefix_cache_hits_total"] == 560.0


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


def test_serve_download_writes_file():
    # _serve_download writes /tmp/bench-{ts}.json synchronously (the
    # FileResponse BackgroundTask unlinks it after the response).
    rec = {"ts": 1.5, "model": "m", "pp2048": 3000.0}
    resp = _serve_download(rec, "bench")
    assert resp.filename == "bench-1.5.json"
    try:
        body = Path(str(resp.path)).read_text()
        assert json.loads(body)["model"] == "m"
    finally:
        Path(str(resp.path)).unlink(missing_ok=True)
