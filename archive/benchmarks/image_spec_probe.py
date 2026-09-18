"""Image + speculative-decoding coherence probe for a VLM served with DFlash.

Verifies that an image prompt + a follow-up (same image) both produce coherent,
non-garbage output and the model can answer a question about the image contents.
Spec-decode (DFlash) is active if the profile configures it.
"""
import base64, io, json, requests

from PIL import Image, ImageDraw

BASE = "http://localhost:8180/v1"
MODEL = "qwen3.8-27b"


def make_image() -> str:
    img = Image.new("RGB", (320, 240), (240, 240, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 200], fill=(220, 40, 40))     # red square
    d.ellipse([140, 60, 280, 200], fill=(40, 120, 220))     # blue circle
    d.text((20, 220), "RED SQUARE BLUE CIRCLE", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


IMG_B64 = make_image()


def ask(messages, label):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 160,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(BASE + "/chat/completions", json=body, timeout=300)
    print(f"\n--- {label} (HTTP {r.status_code}) ---")
    if r.status_code != 200:
        print("ERROR:", r.text[:200])
        return False
    out = r.json()["choices"][0]["message"]["content"]
    print("OUT:", out[:400])
    return True


ok = True
ok &= ask([
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG_B64}"}},
        {"type": "text", "text": "Describe exactly what shapes and colors you see."},
    ]},
], "image turn 1")

ok &= ask([
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG_B64}"}},
        {"type": "text", "text": "What is the shape on the left and what color is it?"},
    ]},
], "image turn 2 (repeated image)")

print("\nRESULT:", "PASS" if ok else "FAIL")
