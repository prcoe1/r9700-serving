"""Unit tests for the dashboard's pure helpers (no network, no container).

Run: uvx pytest observability/dashboard/test_app.py -q
"""
import json
from pathlib import Path

import pytest

from app import (
    _METRICS_RING,
    _append_history_limited,
    _clamp_spike,
    _delete_history_by_ts,
    _derive_metrics,
    _ema,
    _find_record,
    _get_max_concurrency,
    _history_etag,
    _parse_metrics,
    _read_history,
    _record_ring_point,
    _reset_rate_state,
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
vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen3.8-27b",source="local_compute"} 12000000.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen3.8-27b",source="local_cache_hit"} 7275232.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="qwen3.8-27b",source="external_kv_transfer"} 0.0
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
    # per-source breakdown parses per source label
    assert m["vllm:prompt_tokens_by_source:local_compute"] == 12000000.0
    assert m["vllm:prompt_tokens_by_source:local_cache_hit"] == 7275232.0
    assert m["vllm:prompt_tokens_by_source:external_kv_transfer"] == 0.0
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


LATENCY_METRICS = METRICS + """vllm:time_to_first_token_seconds_sum 12.5
vllm:time_to_first_token_seconds_count 10.0
vllm:time_per_output_token_seconds_sum 2.0
vllm:time_per_output_token_seconds_count 100.0
vllm:spec_decode_draft_tokens_total 1000.0
vllm:spec_decode_accepted_tokens_total 750.0
"""


def test_parse_metrics_latency_and_spec():
    m = _parse_metrics(LATENCY_METRICS)
    assert m["vllm:time_to_first_token_seconds_sum"] == 12.5
    assert m["vllm:time_to_first_token_seconds_count"] == 10.0
    assert m["vllm:spec_decode_accepted_tokens_total"] == 750.0
    # labeled duplicates keep last value, bucket lines never shadow
    m2 = _parse_metrics(LATENCY_METRICS + 'vllm:time_to_first_token_seconds_count{engine="0"} 11.0\n')
    assert m2["vllm:time_to_first_token_seconds_count"] == 11.0


def test_parse_metrics_extra_family_fallback():
    # renamed/new counters in a known family are still captured
    m = _parse_metrics(METRICS + "vllm:request_success_total 42.0\n")
    assert m["vllm:request_success_total"] == 42.0
    # unknown families stay out (payload stays small)
    m2 = _parse_metrics(METRICS + "vllm:some_unrelated_gauge 7.0\n")
    assert "vllm:some_unrelated_gauge" not in m2


def test_derive_metrics_ema_and_window():
    _reset_rate_state()
    p1 = {"vllm:prompt_tokens_total": 1000.0, "vllm:generation_tokens_total": 100.0,
          "vllm:prefix_cache_queries_total": 100.0, "vllm:prefix_cache_hits_total": 50.0,
          "vllm:time_to_first_token_seconds_sum": 10.0, "vllm:time_to_first_token_seconds_count": 10.0}
    _derive_metrics(p1, 1000.0)  # seeds baselines, no EMA yet
    assert "pp_ema" not in p1
    p2 = {"vllm:prompt_tokens_total": 4000.0, "vllm:generation_tokens_total": 160.0,
          "vllm:prefix_cache_queries_total": 200.0, "vllm:prefix_cache_hits_total": 150.0,
          "vllm:time_to_first_token_seconds_sum": 22.0, "vllm:time_to_first_token_seconds_count": 20.0}
    _derive_metrics(p2, 1001.0)
    assert p2["pp_ema"] == pytest.approx(3000.0)
    assert p2["tg_ema"] == pytest.approx(60.0)
    assert p2["hit_window_pct"] == pytest.approx(100.0)
    assert p2["ttft_window_mean"] == pytest.approx(1.2)
    assert p2["prefix_hit_pct"] == pytest.approx(75.0)
    _reset_rate_state()


def test_derive_prompt_rate_uses_compute_not_cache_hits():
    # prompt_tokens_total includes cache-hit tokens; the prompt t/s rate must
    # track local_compute (real prefill work) so cache hits don't read as
    # prefill throughput.
    _reset_rate_state()
    _derive_metrics({
        "vllm:prompt_tokens_total": 5000.0,
        "vllm:prompt_tokens_by_source:local_compute": 1000.0,
        "vllm:generation_tokens_total": 100.0,
    }, 3000.0)  # seeds baselines, no EMA yet
    p2 = {
        # 3000 tokens served from cache: total jumps, compute is flat
        "vllm:prompt_tokens_total": 8000.0,
        "vllm:prompt_tokens_by_source:local_compute": 1000.0,
        "vllm:generation_tokens_total": 160.0,
    }
    _derive_metrics(p2, 3001.0)
    assert p2["pp_raw"] == pytest.approx(0.0)
    assert p2["pp_ema"] == pytest.approx(0.0)
    assert p2["tg_ema"] == pytest.approx(60.0)
    _reset_rate_state()


def test_derive_prompt_rate_falls_back_to_total():
    # vLLM builds without the by-source breakdown: total counter still drives
    # the rate (old behavior preserved).
    _reset_rate_state()
    _derive_metrics({"vllm:prompt_tokens_total": 1000.0}, 4000.0)
    p2 = {"vllm:prompt_tokens_total": 4000.0}
    _derive_metrics(p2, 4001.0)
    assert p2["pp_ema"] == pytest.approx(3000.0)
    _reset_rate_state()


def test_derive_metrics_spike_guard_and_reset():
    _reset_rate_state()
    _derive_metrics({"vllm:prompt_tokens_total": 0.0, "vllm:generation_tokens_total": 0.0}, 2000.0)
    _derive_metrics({"vllm:prompt_tokens_total": 3000.0, "vllm:generation_tokens_total": 60.0}, 2001.0)
    assert _ema(None, 5.0) == 5.0
    # 1M-token jump in 1s is a chunked-prefill completion spike → EMA holds
    assert _clamp_spike(1_000_000.0, 3000.0) == 3000.0
    assert _clamp_spike(3100.0, 3000.0) == 3100.0
    # counter reset (e.g. server restart) must not poison the EMA
    before = dict(__import__("app")._RATE)
    _derive_metrics({"vllm:prompt_tokens_total": 10.0, "vllm:generation_tokens_total": 1.0}, 2002.0)
    assert __import__("app")._RATE["pp_ema"] == pytest.approx(before["pp_ema"])
    _reset_rate_state()


def test_ring_record_and_history_shape():
    _reset_rate_state()
    pt = {"ts": 1.0, "vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_running": 2.0,
          "prefix_hit_pct": 50.0, "pp_ema": 3000.0}
    _record_ring_point(pt)
    assert len(_METRICS_RING) == 1
    assert _METRICS_RING[0]["kv"] == 0.5
    assert _METRICS_RING[0]["pp"] == 3000.0
    assert "ttft" not in _METRICS_RING[0]  # unsupported: vLLM exposes no TTFT/ITL histograms
    _reset_rate_state()
    assert len(_METRICS_RING) == 0


def test_get_max_concurrency_prefers_env(monkeypatch):
    import os
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "2")
    assert _get_max_concurrency() == 2
    monkeypatch.delenv("VLLM_MAX_NUM_SEQS")
    monkeypatch.delenv("VLLM_MAX_NUM_SEQS_PER_REQUEST", raising=False)
    assert _get_max_concurrency() == 2  # compose.yaml --max-num-seqs default
    assert os.environ.get("VLLM_MAX_NUM_SEQS") is None
