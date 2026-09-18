#!/usr/bin/env python3
"""Probe for vLLM #47137: streaming vs non-streaming tool-parser parity on
truncated tool calls (qwen3_coder engine parser).

1. Gets a full tool-call response (temperature=0 for determinism).
2. Backs max_tokens off until the tool call is truncated mid-arguments
   (finish_reason == 'length').
3. Requests the same truncated completion non-streaming and streaming,
   assembles the streamed deltas, and compares content + arguments.

Parity (plus no raw <tool_call> markup in content) = the
patches/vllm/47137-tool-truncation-parity.patch halves hold.
Exit 0 on parity, 1 otherwise.
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8180/v1"
MODEL = "qwen3.8-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        },
    }
]
MESSAGES = [
    {
        "role": "user",
        "content": "What is the weather like in San Francisco? "
        "Use the weather tool with celsius.",
    }
]


def post(payload):
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status, resp.read()


def non_stream(max_tokens):
    payload = {
        "model": MODEL,
        "messages": MESSAGES,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    _, body = post(payload)
    data = json.loads(body)
    choice = data["choices"][0]
    msg = choice["message"]
    tcs = msg.get("tool_calls") or []
    return {
        "finish": choice["finish_reason"],
        "content": msg.get("content"),
        "tools": [
            (t["function"]["name"], t["function"].get("arguments")) for t in tcs
        ],
        "usage": data.get("usage", {}),
    }


def stream(max_tokens):
    payload = {
        "model": MODEL,
        "messages": MESSAGES,
        "tools": TOOLS,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    content_parts = []
    arg_parts = []
    names = []
    finish = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choice = chunk["choices"][0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name"):
                    names.append(fn["name"])
                if fn.get("arguments"):
                    arg_parts.append(fn["arguments"])
    return {
        "finish": finish,
        "content": "".join(content_parts) or None,
        "tools": [(names[0] if names else None, "".join(arg_parts))],
    }


def main():
    full = non_stream(1024)
    print("full: finish=%s tools=%s" % (full["finish"], full["tools"]))
    if not full["tools"] or full["finish"] not in ("stop", "tool_calls"):
        print("FAIL: baseline did not produce a complete tool call")
        return 1
    full_args = full["tools"][0][1]
    total = (full["usage"].get("completion_tokens") or 0)
    print("full completion tokens: %d" % total)

    # Back max_tokens off to find two truncation signatures:
    #  (a) mid-arguments cut: tool call promoted with partial args;
    #  (b) early cut: finish=length with no tool calls (content-leak check).
    mid = None
    early = None
    for mt in range(total - 1, max(total - 70, 1), -1):
        r = non_stream(mt)
        if r["tools"] and r["tools"][0][1] != full_args and mid is None:
            mid = (mt, r)
        if r["finish"] == "length" and not r["tools"] and early is None:
            early = (mt, r)
        if mid is not None and early is not None:
            break
    if mid is None:
        print("FAIL: no mid-arguments truncation found")
        return 1
    mt, ns = mid
    print("truncating max_tokens=%d: finish=%s" % (mt, ns["finish"]))
    print("  non-stream content=%r" % (ns["content"],))
    print("  non-stream tools=%s" % (ns["tools"],))

    s = stream(mt)
    print("  stream     content=%r" % (s["content"],))
    print("  stream     tools=%s" % (s["tools"],))

    ok = True
    if ns["finish"] != s["finish"]:
        print("MISMATCH finish: %r vs %r" % (ns["finish"], s["finish"]))
        ok = False
    if (ns["content"] or None) != (s["content"] or None):
        print("MISMATCH content")
        ok = False
    ns_args = ns["tools"][0][1] if ns["tools"] else None
    s_args = s["tools"][0][1] if s["tools"] else None
    if ns_args != s_args:
        print("MISMATCH arguments:\n  ns=%r\n  s =%r" % (ns_args, s_args))
        ok = False
    for label, c in (("non-stream", ns["content"]), ("stream", s["content"])):
        if c and "<tool_call>" in c:
            print("LEAK: raw tool markup in %s content" % label)
            ok = False
    print("PARITY: %s" % ("PASS" if ok else "FAIL"))
    if early is not None:
        emt, er = early
        es = stream(emt)
        print("early cut max_tokens=%d: ns=%r/%s stream=%r/%s"
              % (emt, er["content"], er["tools"], es["content"], es["tools"]))
        if (er["content"] or None) != (es["content"] or None):
            print("MISMATCH early-cut content")
            ok = False
        for label, c in (("non-stream", er["content"]),
                         ("stream", es["content"])):
            if c and "<tool_call>" in c:
                print("LEAK: raw tool markup in %s content" % label)
                ok = False
    print("OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
