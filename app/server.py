"""Spark Studio: a three-step wizard on top of ComfyUI.

Run:  uvicorn server:app --host 0.0.0.0 --port 7860   (from the app/ folder)
Env:  COMFY_URL (default http://127.0.0.1:8188), COMFY_DIR (ComfyUI folder),
      APP_PASSWORD (optional; turns on a simple browser login).
"""
import asyncio
import base64
import io
import json
import os
import random
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

import workflows as wf

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("APP_DATA", HERE.parent / "data"))
DATA.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = DATA / "history.json"
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ["COMFY_DIR"]).expanduser() if os.environ.get("COMFY_DIR") else None
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
CLIENT_ID = uuid.uuid4().hex
PRESETS = json.loads((HERE / "presets.json").read_text())
STYLES = {m: {s["id"]: s for s in lst} for m, lst in PRESETS["styles"].items()}
MODES = {m["id"]: m for m in PRESETS["modes"]}
VIDEO_MODES = ("t2v", "i2v")


def status_key(mode, engine="wan"):
    return f"{mode}:minimax" if engine == "minimax" else mode

jobs: dict[str, dict] = {}
by_prompt: dict[str, str] = {}
ws_state = {"connected": False}
http: httpx.AsyncClient


# ---------------------------------------------------------------- persistence
def load_history():
    if HISTORY_FILE.exists():
        try:
            for j in json.loads(HISTORY_FILE.read_text()):
                if j.get("status") in ("queued", "running"):
                    j["status"] = "checking"  # reconciled against ComfyUI on first poll
                jobs[j["id"]] = j
                by_prompt[j["prompt_id"]] = j["id"]
        except Exception as e:  # corrupt file: keep a copy, start fresh
            HISTORY_FILE.rename(HISTORY_FILE.with_suffix(f".bad-{int(time.time())}"))
            print("history unreadable, started fresh:", e)


def save_history():
    items = sorted(jobs.values(), key=lambda j: j["created"])[-500:]
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    tmp.replace(HISTORY_FILE)


def public(j):
    return {k: v for k, v in j.items() if k not in ("graph",)}


# ------------------------------------------------------------ ComfyUI events
def set_progress(job, sampler_node, value, maximum):
    samplers = job.get("samplers") or []
    idx = samplers.index(sampler_node) if sampler_node in samplers else 0
    n = max(1, len(samplers))
    job["progress"] = round((idx + value / max(1, maximum)) / n, 4)
    job["phase"] = "Rendering" if n == 1 else f"Rendering (pass {idx + 1} of {n})"


async def finish(job):
    """Pull final status and outputs from ComfyUI's history."""
    try:
        r = await http.get(f"{COMFY_URL}/history/{job['prompt_id']}")
        h = r.json().get(job["prompt_id"])
    except Exception:
        return
    if not h:
        return
    status = h.get("status", {})
    outs = []
    for node_out in h.get("outputs", {}).values():
        for item in node_out.get("images", []) + node_out.get("videos", []) + node_out.get("gifs", []):
            if item.get("type") != "output":
                continue
            fn = item["filename"]
            kind = "video" if fn.lower().endswith((".mp4", ".webm", ".mov", ".mkv")) else "image"
            outs.append({"filename": fn, "subfolder": item.get("subfolder", ""), "kind": kind})
    if status.get("status_str") == "success" or (status.get("completed") and outs):
        job.update(status="done", progress=1.0, phase="Done", outputs=outs, finished=time.time())
    else:
        msg = "Generation failed."
        for kind, data in status.get("messages", []):
            if kind == "execution_error":
                msg = friendly_error(data.get("exception_message", ""), data.get("node_type", ""))
            elif kind == "execution_interrupted":
                msg = "Cancelled."
        job.update(status="cancelled" if msg == "Cancelled." else "error", error=msg, finished=time.time())
    save_history()


def friendly_error(msg, node_type=""):
    m = (msg or "").strip()
    low = m.lower()
    if "out of memory" in low or "oom" in low:
        return ("Ran out of memory. Try Fast mode, a shorter clip or a lower resolution. If it keeps happening, "
                "free cached memory on the Spark with: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    if "safetensors" in low or "header" in low or "no such file" in low:
        return f"A model file looks missing or incomplete. Re-run ./spark-studio models for this mode. ({m[:200]})"
    return f"{node_type + ': ' if node_type else ''}{m[:400] or 'Generation failed.'}"


async def ws_listener():
    url = COMFY_URL.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}"
    while True:
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                ws_state["connected"] = True
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue  # live previews; not used
                    try:
                        ev = json.loads(raw)
                    except ValueError:
                        continue
                    await handle_event(ev.get("type"), ev.get("data") or {})
        except Exception:
            ws_state["connected"] = False
            await asyncio.sleep(3)


async def handle_event(kind, data):
    jid = by_prompt.get(data.get("prompt_id", ""))
    job = jobs.get(jid) if jid else None
    if not job or job["status"] in ("done", "error", "cancelled"):
        return
    if kind == "execution_start":
        job.update(status="running", phase="Loading models", started=time.time())
    elif kind == "executing":
        node = data.get("node")
        if node is None:
            await finish(job)
        elif node in (job.get("samplers") or []):
            set_progress(job, node, 0, 1)
        elif node in ("decode",):
            job["phase"] = "Decoding"
        elif node in ("save", "video"):
            job["phase"] = "Saving"
    elif kind == "progress":
        if data.get("node") in (job.get("samplers") or []):
            set_progress(job, data["node"], data.get("value", 0), data.get("max", 1))
    elif kind in ("execution_success", "execution_error", "execution_interrupted"):
        await finish(job)


# ------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_):
    global http
    http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))
    load_history()
    task = asyncio.create_task(ws_listener())
    yield
    task.cancel()
    await http.aclose()


app = FastAPI(title="Spark Studio", lifespan=lifespan)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if APP_PASSWORD:
        auth = request.headers.get("authorization", "")
        ok = False
        if auth.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Spark Studio"'})
    return await call_next(request)


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


async def comfy_models(folder):
    r = await http.get(f"{COMFY_URL}/models/{folder}")
    r.raise_for_status()
    return set(r.json())


@app.get("/api/config")
async def config():
    status = {}
    online = True
    try:
        have = {f: await comfy_models(f) for f in ("diffusion_models", "text_encoders", "vae", "loras")}
        stats = (await http.get(f"{COMFY_URL}/system_stats")).json()
    except Exception:
        online, have, stats = False, {}, {}
    for mode in MODES:
        for engine in (wf.ENGINES if mode in VIDEO_MODES else ("wan",)):
            base = wf.required_files(mode, False, engine)
            full = wf.required_files(mode, True, engine)
            missing = [n for f, names in base.items() for n in names if n not in have.get(f, set())]
            fast_missing = [n for n in full.get("loras", []) if n not in have.get("loras", set()) and n not in missing]
            status[status_key(mode, engine)] = {"ready": online and not missing,
                                                "fast_ready": online and not missing and not fast_missing,
                                                "missing": missing + fast_missing}
    dev = (stats.get("devices") or [{}])[0]
    return {
        "online": online,
        "gpu": dev.get("name", ""),
        "comfy_version": stats.get("system", {}).get("comfyui_version", ""),
        "modes": PRESETS["modes"],
        "styles": PRESETS["styles"],
        "status": status,
        "image_aspects": list(wf.IMAGE_SIZES),
        "video_aspects": list(wf.VIDEO_SIZES["480p"]),
        "engines": {
            "wan": {"name": "WAN 2.2", "blurb": "Silent video", "resolutions": list(wf.VIDEO_SIZES), "seconds": [3, 5, 8], "pack": None},
            "minimax": {"name": "MiniMax H3", "blurb": "Video with sound", "resolutions": list(wf.MINIMAX_SIZES), "seconds": [5, 8, 10, 15], "pack": "minimax"},
        },
    }


async def upload_bytes(data: bytes, filename: str):
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data))
        w, h = im.size
    except Exception:
        raise HTTPException(400, "That file isn't an image I can read.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename or "image.png").name)[-80:]
    name = f"{uuid.uuid4().hex[:8]}_{safe}"
    files = {"image": (name, data, "application/octet-stream")}
    r = await http.post(f"{COMFY_URL}/upload/image", files=files, data={"subfolder": "spark-studio", "type": "input"})
    if r.status_code != 200:
        raise HTTPException(502, f"ComfyUI rejected the upload: {r.text[:200]}")
    info = r.json()
    ref = f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]
    return {"name": ref, "width": w, "height": h}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 50 MB.")
    return await upload_bytes(data, file.filename)


@app.post("/api/reuse")
async def reuse(body: dict):
    """Copy an output image into ComfyUI's input folder so it can be edited or animated."""
    params = {"filename": body["filename"], "subfolder": body.get("subfolder", ""), "type": "output"}
    r = await http.get(f"{COMFY_URL}/view", params=params)
    if r.status_code != 200:
        raise HTTPException(404, "Couldn't find that image.")
    return await upload_bytes(r.content, body["filename"])


async def input_size(name):
    sub, _, fn = name.rpartition("/")
    r = await http.get(f"{COMFY_URL}/view", params={"filename": fn, "subfolder": sub, "type": "input"})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).size


@app.post("/api/generate")
async def generate(body: dict):
    mode = body.get("mode")
    if mode not in MODES:
        raise HTTPException(400, "Pick a media type.")
    style = STYLES[mode].get(body.get("style") or "", next(iter(STYLES[mode].values())))
    user_prompt = (body.get("prompt") or "").strip()
    optional = style.get("placeholder", "").startswith("(optional)")
    if not user_prompt and not optional:
        raise HTTPException(400, "Describe what you want first.")
    full = style["template"].replace("{prompt}", user_prompt).strip()
    full = re.sub(r"\s+([.,])", r"\1", re.sub(r"\s{2,}", " ", full)).strip(" .") + "."
    if MODES[mode]["needs_image"] and not body.get("image"):
        raise HTTPException(400, "Add a photo first.")

    fast = bool(body.get("fast", True))
    cfg = await config()
    engine = body.get("engine") if mode in VIDEO_MODES and body.get("engine") in wf.ENGINES else "wan"
    st = cfg["status"][status_key(mode, engine)]
    if not cfg["online"]:
        raise HTTPException(503, "ComfyUI isn't running. On the Spark: systemctl --user restart spark-studio-comfy")
    if not st["ready"] or (fast and not st["fast_ready"]):
        pack = "minimax" if engine == "minimax" else mode
        raise HTTPException(409, f"Models for this aren't downloaded yet. On the Spark run: ./spark-studio models {pack}")

    seed = body.get("seed")
    seed = int(seed) if str(seed or "").strip().lstrip("-").isdigit() else random.randint(1, 2**50)
    params = {
        "prompt": full, "seed": seed, "fast": fast,
        "aspect": body.get("aspect") or style.get("aspect") or ("16:9" if mode in ("t2v", "i2v") else "1:1"),
        "count": min(4, max(1, int(body.get("count", 1)))),
        "image": body.get("image"),
        "seconds": min(15, max(5, float(body.get("seconds", 5)))) if engine == "minimax" else min(10, max(1, float(body.get("seconds", 5)))),
        "resolution": body.get("resolution") if body.get("resolution") in wf.sizes_for(engine) else "480p",
        "engine": engine,
    }
    if mode == "edit":  # output follows the photo's shape; aspect is only used for display
        try:
            w, h = await input_size(params["image"])
            params["aspect"] = wf.nearest_aspect(w, h, wf.IMAGE_SIZES)
        except Exception:
            raise HTTPException(400, "Couldn't read the uploaded photo. Try adding it again.")
    if mode == "i2v":
        try:
            w, h = await input_size(params["image"])
        except Exception:
            raise HTTPException(400, "Couldn't read the uploaded photo. Try adding it again.")
        sizes = wf.sizes_for(engine)[params["resolution"]]
        params["aspect"] = wf.nearest_aspect(w, h, sizes)
        params["size"] = sizes[params["aspect"]]

    graph, samplers = wf.build(mode, params)
    r = await http.post(f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": CLIENT_ID})
    res = r.json()
    if r.status_code != 200 or res.get("node_errors"):
        detail = res.get("error", {}).get("message", "") if isinstance(res.get("error"), dict) else str(res.get("error", ""))
        for ne in (res.get("node_errors") or {}).values():
            for e in ne.get("errors", []):
                detail += f" {ne.get('class_type', '')}: {e.get('message', '')} {e.get('details', '')}"
        raise HTTPException(500, f"ComfyUI rejected the job: {detail.strip()[:500]}")

    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid, "prompt_id": res["prompt_id"], "mode": mode, "style": style["id"], "style_name": style["name"],
        "user_prompt": user_prompt, "prompt": full, "seed": seed, "fast": fast, "aspect": params["aspect"],
        "seconds": params["seconds"] if mode in ("t2v", "i2v") else None,
        "resolution": params["resolution"] if mode in ("t2v", "i2v") else None,
        "engine": engine if mode in VIDEO_MODES else None,
        "image": params["image"], "samplers": samplers, "status": "queued", "phase": "Waiting in line",
        "progress": 0.0, "created": time.time(), "outputs": [], "error": None,
    }
    jobs[jid] = job
    by_prompt[job["prompt_id"]] = jid
    save_history()
    return public(job)


async def reconcile(active):
    """Fill in queue positions and catch anything the websocket missed."""
    try:
        q = (await http.get(f"{COMFY_URL}/queue")).json()
    except Exception:
        return
    running = {item[1] for item in q.get("queue_running", [])}
    pending = [item[1] for item in sorted(q.get("queue_pending", []), key=lambda i: i[0])]
    for job in active:
        pid = job["prompt_id"]
        if pid in pending:
            ahead = pending.index(pid) + len(running)
            job.update(status="queued", phase=f"Waiting in line ({ahead} ahead)" if ahead else "Up next")
        elif pid in running:
            if job["status"] != "running":
                job.update(status="running", phase="Loading models")
        else:
            await finish(job)
            if job["status"] in ("queued", "running", "checking"):
                job.update(status="error", error="ComfyUI lost this job (it may have restarted).")
                save_history()


@app.get("/api/jobs")
async def list_jobs(limit: int = 200):
    active = [j for j in jobs.values() if j["status"] in ("queued", "running", "checking")]
    if active:
        await reconcile(active)
    items = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)[:limit]
    return {"jobs": [public(j) for j in items], "live": ws_state["connected"]}


@app.post("/api/jobs/{jid}/cancel")
async def cancel(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    if job["status"] == "running":
        await http.post(f"{COMFY_URL}/interrupt", json={"prompt_id": job["prompt_id"]})
    elif job["status"] == "queued":
        await http.post(f"{COMFY_URL}/queue", json={"delete": [job["prompt_id"]]})
        job.update(status="cancelled", error="Cancelled.", finished=time.time())
        save_history()
    return public(job)


@app.delete("/api/jobs/{jid}")
async def delete_job(jid: str):
    job = jobs.pop(jid, None)
    if job:
        by_prompt.pop(job["prompt_id"], None)
        save_history()
    return {"ok": True}


def local_file(kind, filename, subfolder):
    if not COMFY_DIR:
        return None
    base = (COMFY_DIR / kind).resolve()
    p = (base / subfolder / filename).resolve()
    if base not in p.parents:
        raise HTTPException(400, "Bad path")
    return p if p.is_file() else None


@app.get("/api/file")
async def get_file(filename: str, subfolder: str = "", type: str = "output", download: int = 0):
    if type not in ("output", "input"):
        raise HTTPException(400, "Bad type")
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Bad name")
    headers = {"Cache-Control": "max-age=86400"}
    p = local_file(type, filename, subfolder)
    if p:  # served directly so video seeking (range requests) works
        return FileResponse(p, headers=headers, filename=filename if download else None)
    r = await http.get(f"{COMFY_URL}/view", params={"filename": filename, "subfolder": subfolder, "type": type})
    if r.status_code != 200:
        raise HTTPException(404, "Not found")
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(r.content, media_type=r.headers.get("content-type", "application/octet-stream"), headers=headers)


@app.get("/api/health")
async def health():
    try:
        await http.get(f"{COMFY_URL}/system_stats")
        return {"ok": True}
    except Exception:
        return JSONResponse({"ok": False}, status_code=503)
