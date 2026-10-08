#!/usr/bin/env python3
"""Image-correctness probe for the UA backend + image-input warning (vLLM #56021).

An AMD engineer warned that ROCM_AITER_UNIFIED_ATTN with multimodal image
input "silently passes and produces wrong attention output". Our target runs
UA globally with image-cap 99, and multi_image_probe.py only checks liveness
(non-empty completion), so silent corruption would pass unnoticed.

This probe renders images with KNOWN text (embedded 5x7 bitmap font, pure
stdlib) and requires the live server to read them back exactly. Correct reads
under the live config (UA + MTP) mean the warning does not reproduce on our
path; garbled reads mean image support should be disabled (image cap 0).

Exit 0 on PASS, 1 otherwise.
"""
import base64
import json
import struct
import sys
import urllib.request
import zlib

BASE = "http://localhost:8180/v1"
MODEL = "qwen3.8-27b"

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000"] * 7,
}

SCALE = 12
MARGIN = 30


def render_text_png(text: str) -> bytes:
    rows = [""] * 7
    for ch in text:
        for i, row in enumerate(FONT[ch]):
            rows[i] += row + "0"
    w, h = len(rows[0]) * SCALE + 2 * MARGIN, 7 * SCALE + 2 * MARGIN
    px = bytearray([255]) * (w * h)
    for gy, row in enumerate(rows):
        for gx, bit in enumerate(row):
            if bit == "1":
                for dy in range(SCALE):
                    base = (MARGIN + gy * SCALE + dy) * w + MARGIN + gx * SCALE
                    for dx in range(SCALE):
                        px[base + dx] = 0
    raw = b"".join(b"\x00" + bytes(px[y * w:(y + 1) * w]) for y in range(h))

    def chunk(typ: bytes, data: bytes) -> bytes:
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def ask(images_b64: list[bytes], question: str) -> str:
    content: list[dict] = [{"type": "text", "text": question}]
    for img in images_b64:
        content.append(
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}}
        )
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps({
            "model": MODEL, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 16000,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def norm(s: str) -> str:
    return " ".join(s.upper().split())


def main() -> int:
    cases = [
        "QUARTZ 7391", "MANGO 2048",
        "ZEBRA 1307", "FALCON 8842", "WIDGET 5519", "PIXEL 9021",
        "JUMBO 3146", "KAYAK 7720", "VORTEX 6654", "YACHT 2083",
        "WALNUT 4917", "SPHINX 1230",
    ]
    imgs = {t: render_text_png(t) for t in cases}
    if "--write-only" in sys.argv:
        for t, data in imgs.items():
            p = f"/tmp/opencode/ua_probe_{t.split()[0].lower()}.png"
            with open(p, "wb") as f:
                f.write(data)
            print(f"wrote {p}")
        return 0
    ok = True
    for t in cases:
        reply = ask([imgs[t]], "Read the text shown in the image. Reply with only the exact text, nothing else.")
        good = norm(t) in norm(reply)
        print(f"[{'PASS' if good else 'FAIL'}] expected={t!r} reply={reply.strip()!r}")
        ok &= good
    reply = ask(
        [imgs[cases[0]], imgs[cases[1]]],
        "Two images are shown, each with one line of text. Reply with both lines, first image first, separated by ' / '. Nothing else.",
    )
    good = norm(cases[0]) in norm(reply) and norm(cases[1]) in norm(reply)
    print(f"[{'PASS' if good else 'FAIL'}] two-image reply={reply.strip()!r}")
    ok &= good
    print("OVERALL: PASS" if ok else "OVERALL: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
