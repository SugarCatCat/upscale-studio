#!/usr/bin/env python3
"""upscale-studio 的本地 Web 界面：不依赖 ComfyUI。

启动：
    ./run.sh            # 自动准备环境、检查/下载模型、起服务
    python -m upscaler.webui.server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

from .. import registry, videoio
from ..device import pick_device
from ..pipeline import Canceled, run_images, run_videos

ROOT = Path(__file__).resolve().parent.parent.parent     # 项目根
WEB = ROOT / "upscaler" / "webui"
UPLOADS = ROOT / "workspace" / "uploads"
OUTPUTS = ROOT / "workspace" / "outputs"
SETTINGS_FILE = ROOT / "workspace" / "settings.json"

state = {
    "jobs": {},
    "queue": None,
    "pool": None,
    "device": None,          # None = 自动；可被 --device 覆盖
    "setup": {"ready": False, "msg": "正在检查模型…"},
    "models": {"upscale": [], "interp": []},
}


def load_settings() -> dict:
    try:
        d = json.loads(SETTINGS_FILE.read_text("utf-8"))
    except Exception:
        d = {}
    d.setdefault("output_dir", str(OUTPUTS))
    return d


def save_settings(d: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")


# ------------------------------------------------------------------ 模型自检
def prepare_models():
    try:
        for m in registry.MANIFEST:
            name = m["name"]
            p = registry.find_existing(name)
            if p and registry.verify(p, m):
                state["setup"]["msg"] = f"模型就绪：{name}"
                continue
            state["setup"]["msg"] = f"正在下载 {name}（{m['size'] // 1048576} MB）…"
            registry.ensure(name, quiet=True)
        state["models"] = {
            "upscale": [m["name"] for m in registry.MANIFEST if m["kind"] == "upscale"],
            "interp": [m["name"] for m in registry.MANIFEST if m["kind"] == "interp"],
        }
        state["setup"] = {"ready": True, "msg": "模型齐全"}
    except Exception as e:
        state["setup"] = {"ready": False, "msg": f"模型准备失败：{e}"}
        state["models"] = {
            "upscale": [m["name"] for m in registry.MANIFEST if m["kind"] == "upscale"],
            "interp": [m["name"] for m in registry.MANIFEST if m["kind"] == "interp"],
        }


# ------------------------------------------------------------------ 任务
def new_job(mode, params, inputs, jid, src, out_dir):
    job = {
        "id": jid, "mode": mode, "params": params, "inputs": inputs,
        "src": str(src), "out_dir": str(out_dir),
        "source": params.get("source", "upload"),
        "status": "queued", "created": time.time(), "started": None, "finished": None,
        "events": [], "index": 0, "total": len(inputs), "seq": 0, "segs": 0,
        "outputs": [], "error": None, "log": [],
        "_stop": threading.Event(),
    }
    state["jobs"][jid] = job
    return job


def run_job(job):
    job["status"] = "running"
    job["started"] = time.time()

    def emit(kind, msg="", **data):
        if kind in ("item_start", "item_done", "item_fail", "item_skip"):
            job["index"] = data.get("index", job["index"])
            job["total"] = data.get("total", job["total"])
            job["seq"], job["segs"] = 0, 0
        elif kind == "progress":
            job["seq"] = data.get("seg", 0)
            job["segs"] = data.get("segs", 0)
        elif kind == "job_done":
            job["ok"] = data.get("ok")
            job["failed"] = data.get("fail")
        if msg:
            job["log"].append(msg)
            if len(job["log"]) > 4000:          # 留全量，方便排查；只在极长时裁掉最早的
                job["log"] = job["log"][-2000:]
        job["events"].append({"k": kind, "msg": msg, **{k: v for k, v in data.items() if k != "msg"}})
        if len(job["events"]) > 300:
            job["events"] = job["events"][-150:]

    p = job["params"]
    caps = videoio.encoder_caps()
    job["log"].append(f"[env] ffmpeg={videoio.ffmpeg_path()} ({caps.get('version', '')})")
    job["log"].append(f"[env] libx264={caps.get('libx264')} preset={caps.get('preset')} "
                      f"crf={caps.get('crf')} ffprobe={videoio.ffprobe_path()}")
    job["log"].append(f"[job] mode={job['mode']} params={json.dumps(p, ensure_ascii=False)}")
    job["log"].append(f"[job] src={job['src']} → out={job['out_dir']}")
    try:
        if job["mode"] == "image":
            run_images(job["src"], job["out_dir"], scale=p.get("scale", 2),
                       model=p.get("model", "4x-UltraSharp.pth"),
                       device=state["device"], emit=emit,
                       should_stop=job["_stop"].is_set)
        else:
            run_videos(job["src"], job["out_dir"], scale=p.get("scale", 2),
                      multiplier=p.get("multiplier", 2),
                      model=p.get("model", "RealESRGAN_x2plus.pth"),
                      interp_model=p.get("interp", "rife_v4.26.safetensors"),
                      budget_gb=p.get("budget_gb", 4), emit=emit,
                      device=state["device"],
                      keep_audio=not p.get("drop_audio"),
                      should_stop=job["_stop"].is_set)
        job["status"] = "canceled" if job["_stop"].is_set() else (
            "done" if job.get("failed", 0) == 0 else "error")
    except Canceled:
        job["status"] = "canceled"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append(f"异常：{e}")
    finally:
        job["finished"] = time.time()
        collect(job)


def collect(job):
    d = Path(job["out_dir"])
    files = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append({"name": f.name, "size": f.stat().st_size})
    job["outputs"] = files


def public(job):
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ------------------------------------------------------------------ 路由
async def h_index(request):
    return web.FileResponse(WEB / "static" / "index.html")


async def h_status(request):
    running = [j["id"] for j in state["jobs"].values() if j["status"] == "running"]
    dev = pick_device(state["device"])
    return web.json_response({
        "engine": {"device": str(dev), "name": dev.type.upper()},
        "setup": state["setup"],
        "models": state["models"],
        "running": running,
        "queue": sum(1 for j in state["jobs"].values() if j["status"] == "queued"),
        "settings": load_settings(),
    })


async def h_settings(request):
    if request.method == "POST":
        data = await request.json()
        raw = (data.get("output_dir") or "").strip()
        if raw:
            d = Path(os.path.expanduser(raw)).resolve()
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise web.HTTPBadRequest(text=f"目录不可用：{d}（{e}）")
            save_settings({"output_dir": str(d)})
    return web.json_response(load_settings())


async def h_jobs(request):
    jobs = sorted(state["jobs"].values(), key=lambda j: j["created"], reverse=True)
    return web.json_response([public(j) for j in jobs[:50]])


async def h_job(request):
    job = state["jobs"].get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    return web.json_response(public(job))


def _pick_name(rel, seen, idx):
    base = os.path.basename((rel or "").replace("\\", "/")) or f"file{idx}"
    name, k = base, 2
    stem, ext = os.path.splitext(base)
    while name in seen:
        name = f"{stem}_{k}{ext}"
        k += 1
    seen.add(name)
    return name


def _unique_dir(base: Path, mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M")
    d = base / f"{stamp}_{mode}"
    k = 2
    while d.exists():
        d = base / f"{stamp}_{mode}_{k}"
        k += 1
    d.mkdir(parents=True)
    return d


async def h_create(request):
    reader = await request.multipart()
    params, mode, jid, saved, updir, seen = {}, None, None, [], None, set()
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "params":
            params = json.loads(await part.text())
            mode = params.get("mode")
            continue
        if part.name == "files":
            if mode not in ("image", "video"):
                raise web.HTTPBadRequest(text="params 必须放在 files 之前，且 mode 只能是 image/video")
            jid = jid or uuid.uuid4().hex[:10]
            updir = UPLOADS / jid
            updir.mkdir(parents=True, exist_ok=True)
            dest = updir / _pick_name(part.filename, seen, len(saved))
            with open(dest, "wb") as f:
                while True:
                    chunk = await part.read_chunk(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            saved.append(dest)

    if mode not in ("image", "video"):
        raise web.HTTPBadRequest(text=f"未知模式 {mode}")

    if (params.get("source") or "upload") == "path":
        raw = (params.get("input_path") or "").strip()
        if not raw:
            raise web.HTTPBadRequest(text="请填写本地文件或文件夹路径")
        src = Path(os.path.expanduser(raw)).resolve()
        if not src.exists():
            raise web.HTTPBadRequest(text=f"路径不存在：{src}")
        if src.is_dir():
            exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"} if mode == "video" else \
                   {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            found = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower() in exts]
            if not found:
                raise web.HTTPBadRequest(text=f"目录里没有可处理的文件：{src}")
            inputs = [p.name for p in found]
        else:
            inputs = [src.name]
    else:
        if not saved:
            raise web.HTTPBadRequest(text="没有收到文件")
        src = saved[0] if len(saved) == 1 else updir
        inputs = [p.name for p in saved]

    base_raw = (params.get("output_dir") or "").strip() or load_settings()["output_dir"]
    base = Path(os.path.expanduser(base_raw)).resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise web.HTTPBadRequest(text=f"输出目录不可用：{base}（{e}）")
    out_dir = _unique_dir(base, mode)
    save_settings({"output_dir": str(base)})

    jid = jid or uuid.uuid4().hex[:10]
    job = new_job(mode, params, inputs, jid, src, out_dir)
    state["queue"].put_nowait(job)
    return web.json_response({"id": job["id"], "out_dir": str(out_dir)})


async def h_log(request):
    job = state["jobs"].get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    body = "\n".join(job["log"])
    if job.get("error"):
        body += f"\n\n[error] {job['error']}"
    return web.Response(text=body, content_type="text/plain", charset="utf-8")


async def h_cancel(request):
    job = state["jobs"].get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    if job["status"] in ("queued", "running"):
        job["_stop"].set()
        job["status"] = "canceled"
        return web.json_response({"ok": True})
    raise web.HTTPBadRequest(text="任务已结束")


async def h_delete(request):
    jid = request.match_info["id"]
    job = state["jobs"].get(jid)
    if not job:
        raise web.HTTPNotFound()
    job["_stop"].set()
    shutil.rmtree(UPLOADS / jid, ignore_errors=True)
    shutil.rmtree(Path(job["out_dir"]), ignore_errors=True)
    state["jobs"].pop(jid, None)
    return web.json_response({"ok": True})


def _src_dir(job):
    p = Path(job["src"])
    return p if p.is_dir() else p.parent


async def h_file(request):
    job = state["jobs"].get(request.match_info["id"])
    name = request.match_info["name"]
    if not job or "/" in name or name.startswith("."):
        raise web.HTTPBadRequest()
    p = Path(job["out_dir"]) / name
    if not p.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-cache"})


async def h_input(request):
    job = state["jobs"].get(request.match_info["id"])
    name = request.match_info["name"]
    if not job or "/" in name or name.startswith("."):
        raise web.HTTPBadRequest()
    p = _src_dir(job) / name
    if not p.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ 后台
async def worker():
    """单线程串行跑任务：同时跑两个只会互相抢显存。"""
    loop = asyncio.get_running_loop()
    while True:
        job = await state["queue"].get()
        try:
            if job["status"] == "canceled":
                continue
            await loop.run_in_executor(state["pool"], run_job, job)
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"内部错误：{e}"
        finally:
            state["queue"].task_done()


async def on_startup(app):
    for d in (UPLOADS, OUTPUTS, SETTINGS_FILE.parent):
        d.mkdir(parents=True, exist_ok=True)
    state["queue"] = asyncio.Queue()
    state["pool"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    asyncio.create_task(worker())
    threading.Thread(target=prepare_models, daemon=True).start()


async def on_cleanup(app):
    state["pool"].shutdown(wait=False)


def make_app():
    app = web.Application(client_max_size=16 * 1024 ** 3)
    app.router.add_get("/", h_index)
    app.router.add_get("/api/status", h_status)
    app.router.add_get("/api/settings", h_settings)
    app.router.add_post("/api/settings", h_settings)
    app.router.add_get("/api/jobs", h_jobs)
    app.router.add_get("/api/jobs/{id}", h_job)
    app.router.add_get("/api/jobs/{id}/log", h_log)
    app.router.add_post("/api/jobs", h_create)
    app.router.add_post("/api/jobs/{id}/cancel", h_cancel)
    app.router.add_delete("/api/jobs/{id}", h_delete)
    app.router.add_get("/files/{id}/{name}", h_file)
    app.router.add_get("/input/{id}/{name}", h_input)
    app.router.add_static("/static", WEB / "static")
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"],
                    help="强制设备；默认自动（cuda > mps > cpu）")
    a = ap.parse_args()
    state["device"] = a.device
    if a.output_dir:
        p = Path(os.path.expanduser(a.output_dir)).resolve()
        p.mkdir(parents=True, exist_ok=True)
        save_settings({"output_dir": str(p)})
    print(f"upscale-studio → http://{a.host}:{a.port}")
    web.run_app(make_app(), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
