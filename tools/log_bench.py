#!/usr/bin/env python3
"""Wrap llama-benchy json output for dashboard history."""
import json
import sys

path, ts, model = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(path).read()
try:
    j = json.loads(raw)
except Exception:
    j = {"raw": raw}
j["ts"] = float(ts)
j["model"] = model
if "raw" not in j or not isinstance(j["raw"], str):
    j["raw"] = raw[:12000]
else:
    j["raw"] = j["raw"][:12000]
print(json.dumps(j))
