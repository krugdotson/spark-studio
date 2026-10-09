#!/usr/bin/env python3
"""Render one of each thing Spark Studio can make, through the app's own API, and print a timing table.

Usage:
  ./spark-studio selftest           every mode and video engine whose models are downloaded
  ./spark-studio selftest --quick   just one image

Only tests whose models are downloaded run; the rest are listed as skipped. Results also appear in the
web page's "Your work" area. Exit code is 1 if any test fails.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import httpx

APP = f"http://127.0.0.1:{os.environ.get('APP_PORT', '7860')}"
COMFY_DIR = os.environ.get("COMFY_DIR", "")
COMFY_PY = os.environ.get("COMFY_PY", "")
TIMEOUT = 30 * 60  # per test

# (label, request, needs a photo from the image test)
TESTS = [
    ("Image", {"mode": "image", "style": "photo", "prompt": "a red apple on a wooden table"}, False),
    ("Edit a photo", {"mode": "edit", "style": "free", "prompt": "make the apple green"}, True),
    ("Video: WAN 2.2, 3 s", {"mode": "t2v", "engine": "wan", "style": "none", "seconds": 3,
                            "prompt": "waves rolling onto a beach"}, False),
    ("Video: MiniMax H3, 5 s + sound", {"mode": "t2v", "engine": "minimax", "style": "none", "seconds": 5,
                                       "prompt": "Waves rolling onto a beach at sunset. Sound: surf, gulls, light wind."}, False),
    ("Video: LTX 2.3, 5 s 720p + sound", {"mode": "t2v", "engine": "ltx", "style": "none", "seconds": 5, "resolution": "720p",
                                         "prompt": "Waves rolling onto a beach at sunset. Sound: surf, gulls, light wind."}, False),
    ("Photo to video: WAN 2.2, 3 s", {"mode": "i2v", "engine": "wan", "style": "free", "seconds": 3,
                                     "prompt": "The apple slowly rotates on the table."}, True),
    ("Photo to video: MiniMax H3, 5 s", {"mode": "i2v", "engine": "minimax", "style": "free", "seconds": 5,
                                        "prompt": "The apple slowly rotates on the table. Sound: quiet kitchen ambience."}, True),
    ("Photo to video: LTX 2.3, 3 s", {"mode": "i2v", "engine": "ltx", "style": "free", "seconds": 3, "resolution": "480p",
                                     "prompt": "The apple slowly rotates on the table. Sound: quiet kitchen ambience."}, True),
]


def status_key(req):
    engine = req.get("engine", "wan")
    return f"{req['mode']}:{engine}" if engine != "wan" else req["mode"]


def media_info(path):
    """Return 'video+audio' / 'video' / 'image' for an output file, using PyAV from ComfyUI's environment."""
    if not path.endswith((".mp4", ".webm", ".mkv")):
        return "image"
    if not COMFY_PY:
        return "video"
    code = "import av,sys; c=av.open(sys.argv[1]); print(','.join(sorted({s.type for s in c.streams})))"
    try:
        out = subprocess.run([COMFY_PY, "-c", code, path], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return "video"
    return "video+audio" if "audio" in out else "video"


def run(client, label, req):
    body = {"fast": True, **req}
    r = client.post(f"{APP}/api/generate", json=body)
    if r.status_code != 200:
        return "FAIL", 0, r.json().get("detail", r.text)[:120], None
    jid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < TIMEOUT:
        time.sleep(2)
        try:
            jobs = client.get(f"{APP}/api/jobs").json()["jobs"]
        except Exception:
            continue
        j = next((x for x in jobs if x["id"] == jid), None)
        if not j:
            return "FAIL", time.time() - t0, "job disappeared", None
        print(f"\r  {label}: {j.get('phase') or j['status']} · {int(time.time() - t0)}s      ", end="", flush=True)
        if j["status"] == "done":
            took = (j.get("finished") or time.time()) - (j.get("started") or j["created"])
            out = (j.get("outputs") or [None])[0]
            if not out:
                return "FAIL", took, "finished without an output file", None
            path = os.path.join(COMFY_DIR, "output", out.get("subfolder", ""), out["filename"])
            kind = media_info(path) if COMFY_DIR else "?"
            if req.get("engine") in ("minimax", "ltx") and kind == "video":
                return "FAIL", took, f"{out['filename']} has no audio track", out
            return "ok", took, f"{out['filename']} ({kind})", out
        if j["status"] in ("error", "cancelled"):
            return "FAIL", time.time() - t0, (j.get("error") or j["status"])[:120], None
    return "FAIL", TIMEOUT, "timed out", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only render one image")
    args = ap.parse_args()

    client = httpx.Client(timeout=60)
    try:
        cfg = client.get(f"{APP}/api/config").json()
    except Exception as e:
        sys.exit(f"Spark Studio isn't answering at {APP} ({e}). Try: ./spark-studio status")
    if not cfg.get("online"):
        sys.exit("ComfyUI isn't running. Try: ./spark-studio restart")
    print(f"  GPU: {cfg.get('gpu', '?')} · ComfyUI {cfg.get('comfy_version', '?')}\n")

    tests = TESTS[:1] if args.quick else TESTS
    photo = None
    rows = []
    for label, req, needs_photo in tests:
        st = cfg["status"].get(status_key(req), {})
        if not st.get("ready"):
            rows.append((label, "skipped", None, "models not downloaded"))
            continue
        if needs_photo and not photo:
            rows.append((label, "skipped", None, "needs the image test's picture"))
            continue
        if needs_photo:
            req = {**req, "image": photo}
        result, took, note, out = run(client, label, req)
        print("\r" + " " * 90 + "\r", end="")
        print(f"  {'✓' if result == 'ok' else '✗'} {label}: {note} ({took:.1f} s)")
        rows.append((label, result, took, note))
        if req["mode"] == "image" and out:
            r = client.post(f"{APP}/api/reuse", json=out)
            if r.status_code == 200:
                photo = r.json()["name"]

    print(f"\n  {'Test':36} {'Result':8} {'Time':>8}")
    print(f"  {'-' * 36} {'-' * 8} {'-' * 8}")
    for label, result, took, note in rows:
        t = f"{took:.1f} s" if took is not None else ""
        print(f"  {label:36} {result:8} {t:>8}" + (f"   {note}" if result != "ok" else ""))
    failed = [r for r in rows if r[1] == "FAIL"]
    ran = [r for r in rows if r[1] != "skipped"]
    print(f"\n  {len(ran) - len(failed)} of {len(ran)} passed" + (f", {len(rows) - len(ran)} skipped" if len(ran) < len(rows) else ""))
    if not ran:
        print("  Nothing was tested: download some model packs first (./spark-studio models all).")
    sys.exit(1 if failed or not ran else 0)


if __name__ == "__main__":
    main()
