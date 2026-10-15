# 2026-10-15 — MTP3 re-verification, spec-off baseline, c2-at-depth isolation

Live profile: qwen3.8-27b, v0.32.0rc0 + local patches, bf16 KV (block 832),
131K ctx, cap 2, chunk 1024, drafter pinned UA, `--no-async-scheduling`.

## d0 ladder (bf16, same build, same day)

| spec | pp2048 | tg32 | tg128 |
|:-----|-------:|-----:|------:|
| off | ~3443 | 34.0 | 34.3 |
| MTP1 (10-14 row) | ~3297 | ~53 | ~52 |
| MTP3 | ~3265 | 68.3 | 69.3 |

MTP3 ≈ 2× decode for ~−5% prefill. Acceptance 66% (470/708 d0;
1001/1512 incl. depth traffic — no decay signal). Coherence PASS (bench) +
100K needle recall exact (`BLUEBIRD-7429`, temp 0).

Depth under MTP3 (tg128): d0 69.3 / d65K 61.3 / d100K 59.4. No #47602-style
collapse (old parity point was ~110K).

## c2-at-depth collapse — isolated, cause still open

c2 per-request tg falls off a cliff from ~4K up, while c1 holds flat and
prefill holds at all conc/depth. Preemptions = 0 throughout.

| depth | c1 off | c1 MTP3 | c2 off/bf16 | c2 MTP3/bf16 | c2 MTP3/fp8 |
|------:|-------:|--------:|------------:|-------------:|------------:|
| 0 | 34.2 | 67.6 | 40.2 | 52.7 | — |
| 4K | 33.3 | 56.5 | 22.1 | 24.9 | — |
| 8K | 33.1 | 66.0 | 14.8 | 15.9 | — |
| 16K | 32.5 | 59.5 | 8.8 | 8.7 | — |
| 32K | 32.0 | 57.8 | 4.1 | 4.2 | 5.7 |
| 65K | 30.2 | 43.4 | 1.8 | 1.8 | 2.7 |
| 111K | 27.7 | 54.9 | 0.84 | 0.84 | 1.44 |

Ruled out: MTP/draft machinery (identical 0.84 spec-off), KV dtype/capacity
(fp8 only ~1.5× less bad, same shape; block 1600 vs 832), pool exhaustion
(no preemptions). Remaining suspects: #55533-family Mamba-budget sharing at
concurrency, align per-step overhead scaling with context × sequences.

Rule of thumb: c2 wins aggregate shallow (2×52.7 ≈ 105 vs 67.6 solo at d0),
breaks even ~8K, loses hard beyond — serialize deep concurrent work.

## Incidental: empty-override config bug (fixed same day)

`VLLM_SPEC_DECODE=` (empty) in the last `--env-file` does NOT clear the
value from `env/qwen3.6.env.common` — the "spec-off" bench first ran as
silent MTP4 (tg ~67–72 gave it away; server log confirmed
`num_speculative_tokens: 4`). Fix: no shared spec-decode default; each
profile pins its own (`qwen3.6-27b.env`, `qwen3.6-35b-a3b.env` carry MTP4);
spec-off profiles omit the var. The awq profile had the same latent bug
(intended off, got MTP4). Verified per-profile via `compose config`:
qwen3.8-27b + awq carry no `--speculative-config`, both 3.6 profiles keep
MTP4. The discarded MTP4 run is NOT recorded above.

## Config after today

MTP3 live, bf16 KV (fp8 trial reverted — no decisive win, keep the 10-08
operator baseline), cap 2. Raw sweeps in `observability/conc_history.jsonl`
(2026-10-15 rows) and `observability/depth_history.jsonl`.
