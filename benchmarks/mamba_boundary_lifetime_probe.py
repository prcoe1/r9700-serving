#!/usr/bin/env python3
"""GPU-free regression guard: Mamba align prompt-boundary lifetime (#61281).

Drives KVCacheManager directly on the live qwen3.8-27b geometry (attn block
832, mamba block 16 GDN, hash GCD 16, sched 832, MTP eagle drop, scheduler-
faithful chunking) through pressured multi-turn conversations and asserts
the reusable boundary is hit on follow-up turns.

Why this exists: benchmarks/prefix_cache_probe.py is structurally insensitive
on this geometry (its prompts stay under ~500 tokens: no full 832 attention
block exists and the Mamba boundary is 0 below 1664 tokens under MTP, so it
reports 0% fixed or not). This probe uses long prompts (P1 >= 2000) where a
non-zero reconciled hit is mechanically possible, plus churn between turns.

Covers upstream #61282 (carried as patches/vllm/61282-*.patch): retention
hold through decode+churn and last-evicted free order. Exits 0 iff every
long-prompt case reconciles exactly its theory boundary with no phantom hits
on short prompts.

Usage (needs the built image for torch+vllm; host python lacks them):
    docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \\
      -v benchmarks/mamba_boundary_lifetime_probe.py:/tmp/probe.py:ro \\
      --entrypoint python3 localhost/vllm-fullbuild:latest /tmp/probe.py
    CASES=3000  # optional: run only P1=3000 (comma-separated P1 list)
"""
import itertools
import os
import sys

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.request import Request

ATTN_BLOCK = 832
MAMBA_BLOCK = 16
HASH_BLOCK = 16
SCHED_BLOCK = 832
CHUNK_CAP = 1024
DECODE_STEPS = 64
CHURN_REQUESTS = int(os.environ.get("CHURN_N_OVERRIDE", "40"))
POST_N = int(os.environ.get("POST_N_OVERRIDE", "") or 0)
CHURN_LEN = int(os.environ.get("CHURN_LEN_OVERRIDE", "2000"))
FOLLOW_NEW = 32
NB_OVERRIDE = os.environ.get("NB_OVERRIDE", "")

_token_ids = itertools.count(1000)
init_none_hash(sha256)


def theory_boundary(n):
    return get_mamba_prefill_checkpoint_position(n, SCHED_BLOCK, True)


def split_chunks(P):
    """Faithful replica of _mamba_block_aligned_split stops for our config.

    GDN: no internal prefill checkpoints, partial-hash-hits off, no junction.
    last_cache_position and replay_boundary are per-request constants.
    """
    lcp = theory_boundary(P)
    resend = (P - 1) // SCHED_BLOCK * SCHED_BLOCK
    ext = P // SCHED_BLOCK * SCHED_BLOCK
    rb = min(resend, ext) - SCHED_BLOCK
    rb = max(rb, 0)
    stops_const = {s for s in (lcp, rb) if s > 0}
    chunks, start = [], 0
    while start < P:
        end = min(start + CHUNK_CAP, P)
        if end < P:
            ae = end // SCHED_BLOCK * SCHED_BLOCK
            if ae > start:
                end = ae
        for s in sorted(stops_const):
            if start < s < end:
                end = s
                break
        assert end > start, (P, start)
        chunks.append(end - start)
        start = end
    return chunks


def make_manager(nblocks, retention):
    attn = KVCacheGroupSpec(
        ["attn"],
        FullAttentionSpec(block_size=ATTN_BLOCK, num_kv_heads=1, head_size=1,
                           dtype=torch.float32),
        # Live MTP: draft attention layers join this group (same spec merge),
        # MTP heads have no mamba layers, so the mamba group stays unflagged.
        is_eagle_group=True)
    mamba = KVCacheGroupSpec(
        ["mamba"],
        MambaSpec(block_size=MAMBA_BLOCK, shapes=((1, 1),),
                  dtypes=(torch.float32,), mamba_cache_mode="align",
                  mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                  num_speculative_blocks=3))
    config = KVCacheConfig(num_blocks=nblocks, kv_cache_tensors=[],
                           kv_cache_groups=[attn, mamba],
                           prefix_cache_retention_interval=retention)
    manager = KVCacheManager(config, max_model_len=16384,
                             scheduler_block_size=SCHED_BLOCK,
                             hash_block_size=HASH_BLOCK, enable_caching=True,
                             use_eagle=True)  # MTP: eagle groups + dropped block
    for m in manager.coordinator.single_type_managers:
        if type(m).__name__ == "MambaManager":
            m.drop_eagle_checkpoint_block = True
    return manager


def make_request(rid, prompt):
    return Request(request_id=rid, prompt_token_ids=prompt, mm_features=None,
                   sampling_params=SamplingParams(max_tokens=64),
                   pooling_params=None,
                   block_hasher=get_request_block_hasher(HASH_BLOCK, sha256))


def release_cow(manager):
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)


def prefill(manager, request, chunks):
    computed, num_computed, _ = manager.get_computed_blocks(request)
    for i, chunk in enumerate(chunks):
        blocks = manager.allocate_slots(request, chunk,
                                        num_computed if i == 0 else 0,
                                        computed if i == 0 else None)
        assert blocks is not None, f"allocate failed chunk {chunk}"
        request.num_computed_tokens += chunk + (num_computed if i == 0 else 0)
        manager.coordinator.new_step_starts()
        release_cow(manager)


def decode(manager, request, steps):
    for _ in range(steps):
        request.append_output_token_ids(next(_token_ids))
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        manager.coordinator.new_step_starts()
        release_cow(manager)


def churn(manager, n, length):
    for i in range(n):
        o = make_request(f"other-{next(_token_ids)}",
                         [next(_token_ids) for _ in range(length)])
        prefill(manager, o, split_chunks(length))
        manager.free(o)


def group_id(manager, spec_type):
    for gid, group in enumerate(manager.kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, spec_type):
            return gid
    raise AssertionError(spec_type)


def mamba_hashes(manager, mamba_group):
    return {b.block_hash_num_tokens: b.block_hash
            for b in manager.block_pool.blocks
            if b.block_hash is not None
            and get_group_id(b.block_hash) == mamba_group}


def queue_pos(manager, block):
    q = manager.block_pool.free_block_queue.get_all_free_blocks()
    ids = [b.block_id for b in q]
    return ids.index(block.block_id) if block.block_id in ids else None


def run_conversation(p1, nturns, retention, nblocks):
    from vllm.v1.kv_cache_interface import FullAttentionSpec as FA
    from vllm.v1.kv_cache_interface import MambaSpec as MS
    manager = make_manager(nblocks, retention)
    ag, mg = group_id(manager, FA), group_id(manager, MS)
    print(f"  chunks({p1})={split_chunks(p1)} "
          f"theory_boundary={theory_boundary(p1)}", flush=True)
    tokens = [next(_token_ids) for _ in range(p1)]
    results = []
    for t in range(1, nturns + 1):
        req = make_request(f"t{t}-{p1}", tokens)
        _, num_computed, _ = manager.get_computed_blocks(req)
        prefill(manager, req, split_chunks(len(tokens)))
        b0 = theory_boundary(len(tokens))
        reg = mamba_hashes(manager, mg)
        reg_key = reg.get(b0)
        snap = next((b for b in manager.block_pool.blocks
                     if b.block_hash == reg_key), None) if reg_key else None
        trace = []
        if snap is not None:
            trace.append(f"prefill:ref={snap.ref_cnt},q={queue_pos(manager, snap)}")
        decode(manager, req, DECODE_STEPS // 2)
        if snap is not None:
            trace.append(f"dec32:ref={snap.ref_cnt},q={queue_pos(manager, snap)}")
        churn(manager, CHURN_REQUESTS // 2, CHURN_LEN)
        decode(manager, req, DECODE_STEPS // 2)
        churn(manager, CHURN_REQUESTS // 2, CHURN_LEN)
        if snap is not None:
            alive = {b.block_hash for b in manager.block_pool.blocks
                     if b.block_hash is not None}
            trace.append(f"churn:ref={snap.ref_cnt},q={queue_pos(manager, snap)},"
                         f"hashkept={reg_key in alive}")
        if t < nturns:
            manager.free(req)
            if snap is not None:
                trace.append(f"free:ref={snap.ref_cnt},q={queue_pos(manager, snap)}")
            churn(manager, POST_N, CHURN_LEN)
            if snap is not None:
                alive = {b.block_hash for b in manager.block_pool.blocks
                         if b.block_hash is not None}
                trace.append(f"postfree-churn:q={queue_pos(manager, snap)},"
                             f"hashkept={reg_key in alive}")
            print(f"  turn{t} boundary-block trace: {' | '.join(trace)}",
                  flush=True)
            tokens = list(req.all_token_ids) + [next(_token_ids)
                                                for _ in range(FOLLOW_NEW)]
            nxt = make_request(f"t{t + 1}-{p1}-probe", tokens)
            _, hits = manager.coordinator.find_longest_cache_hit_per_group(
                nxt.block_hashes, nxt.num_tokens - 1)
            _, nc, _ = manager.get_computed_blocks(nxt)
            mh = mamba_hashes(manager, mg)
            b = theory_boundary(len(tokens))
            results.append((len(tokens), b, hits[ag], hits[mg], nc,
                            len(mh), reg_key is not None,
                            reg_key in alive if snap is not None else False))
    return results


def main():
    only = os.environ.get("CASES", "")
    cases = []
    for p1, nb in ((400, 160), (2000, 160), (3000, 160), (5000, 160)):
        if NB_OVERRIDE:
            nb = int(NB_OVERRIDE)
        for ret in (0, None):
            for nt in ((2,) if p1 != 2000 else (2, 3)):
                cases.append((p1, ret, nt, nb))
    if only:
        cases = [c for c in cases if str(c[0]) in only]
    failed = 0
    for p1, ret, nt, nb in cases:
        print(f"P1={p1} retention={ret} turns={nt}:", flush=True)
        for plen, b, ah, mh, nc, nhashes, regged, surv in run_conversation(
                p1, nt, ret, nb):
            if b > 0:
                ok = (nc == b and mh == b)
            else:
                ok = (nc == 0 and mh == 0)  # structural miss, no phantoms
            failed += not ok
            print(f"  turnP={plen} boundary={b} attn={ah} mamba={mh} "
                  f"reconciled={nc} mhashes={nhashes} regged={regged} "
                  f"survived={surv} => {'PASS' if ok else 'FAIL'}",
                  flush=True)
    print("OVERALL: %s" % ("PASS" if not failed else "FAIL"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
