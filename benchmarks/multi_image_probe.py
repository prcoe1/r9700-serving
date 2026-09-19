#!/usr/bin/env python3
"""Multi-image probe for the #40707 scheduler-deadlock fix
(patches/vllm/40707-mamba-block-aligned-split-deadlock.patch).

Unpatched, 2+ large images in one prompt hang the engine forever: the
encoder cache holds one image while the second waits, and
_mamba_block_aligned_split floors the sub-block inter-image gap to 0 new
tokens, so the scheduler skips the request and Image 1's cache entry is
never freed.

Sends one prompt with 4x 3024x4032 photos (picsum seeds, ~11.8k vision
tokens each, ~47k prompt tokens total) and requires a completed, non-empty
answer. Exit 0 on PASS, 1 otherwise.
"""
import base64
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8180/v1"
MODEL = "qwen3.8-27b"
SEEDS = ["alpha", "bravo", "charlie", "delta"]
IMG_W, IMG_H = 3024, 4032


def fetch_images(d: Path) -> list[Path]:
    paths = []
    for s in SEEDS:
        p = d / f"{s}.jpg"
        req = urllib.request.Request(
            f"https://picsum.photos/seed/{s}/{IMG_W}/{IMG_H}",
            headers={"User-Agent": "r9700-probe"},
        )
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
            f.write(r.read())
        paths.append(p)
    return paths


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mmprobe-") as tmp:
        try:
            imgs = fetch_images(Path(tmp))
        except Exception as e:
            print(f"FAIL: image download: {type(e).__name__}: {e}")
            return 1
        content: list[dict] = [
            {"type": "text", "text": "You are shown several photos. "}
        ]
        for i, p in enumerate(imgs, 1):
            b64 = base64.b64encode(p.read_bytes()).decode()
            content.append({"type": "text", "text": f"Photo {i}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "Reply in one short sentence: "
                    f"what do the {len(imgs)} photos have in common?"
                ),
            }
        )
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{BASE}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                body = json.load(resp)
        except Exception as e:
            print(f"FAIL: request hung/errored after {time.time() - t0:.1f}s: "
                  f"{type(e).__name__}: {e}")
            return 1
    dt = time.time() - t0
    msg = body["choices"][0]["message"]
    usage = body.get("usage", {})
    text = msg.get("content") or ""
    print(f"completed in {dt:.1f}s, usage={usage}")
    print(f"content: {text!r}")
    if not text.strip():
        print("FAIL: empty content")
        return 1
    print("PASS: multi-image request completed with non-empty answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
