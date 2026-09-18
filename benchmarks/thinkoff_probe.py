#!/usr/bin/env python3
"""Probe for the qwen3 thinkoff fix
(patches/vllm/qwen3-thinkoff-kwarg-parity.patch).

With chat_template_kwargs={"reasoning_effort": "off"} the template
pre-closes <think> in the prompt. Unpatched, Qwen3Parser (which only reads
enable_thinking) files the whole output as reasoning: content=None with the
answer hiding in reasoning. Patched, content carries the answer.

Checks, all temperature=0:
  1. default request -> thinking on, content present (control).
  2. reasoning_effort=off, non-streaming -> content holds the answer.
  3. reasoning_effort=off, streaming -> same as (2).
Exit 0 when all hold, 1 otherwise.
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8180/v1"
MODEL = "qwen3.8-27b"
PROMPT = "What is 17*23? Answer with just the number."
EXPECTED = "391"


def post(payload, stream=False):
    payload = dict(payload)
    payload["stream"] = stream
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        if not stream:
            return json.loads(resp.read())["choices"][0]
        content_parts, reasoning_parts, finish = [], [], None
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choice = json.loads(data)["choices"][0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
        return {
            "finish_reason": finish,
            "message": {
                "content": "".join(content_parts) or None,
                "reasoning_content": "".join(reasoning_parts) or None,
            },
        }


def base(extra=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 200,
    }
    if extra:
        payload.update(extra)
    return payload


def show(label, choice):
    msg = choice["message"]
    print("%s: finish=%s content=%r reasoning=%r"
          % (label, choice["finish_reason"],
             msg.get("content"), msg.get("reasoning_content")))


def main():
    ok = True
    ctl = post(base())
    show("control (default)      ", ctl)
    if not (ctl["message"].get("content") and
            EXPECTED in ctl["message"]["content"]):
        print("FAIL: control response has no content answer")
        ok = False

    off = post(base({"chat_template_kwargs": {"reasoning_effort": "off"}}))
    show("effort=off non-stream  ", off)
    if not (off["message"].get("content") and
            EXPECTED in off["message"]["content"]):
        print("FAIL: effort=off non-stream content missing answer "
              "(answer stranded in reasoning?)")
        ok = False

    offs = post(base({"chat_template_kwargs": {"reasoning_effort": "off"}}),
                stream=True)
    show("effort=off stream      ", offs)
    if ((offs["message"].get("content") or None) !=
            (off["message"].get("content") or None)):
        print("FAIL: effort=off stream/non-stream content diverge")
        ok = False

    print("THINKOFF: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
