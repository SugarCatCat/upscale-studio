"""流水线：图片批量 / 视频补帧+超分。CLI 与 WebUI 共用。

事件回调签名同 WebUI：emit(kind, msg="", **data)
  job_start / item_start / progress / item_done / item_fail / item_skip / log / job_done
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

from . import registry, videoio
from .device import INT_MAX
from .interp import Interpolator
from .upscale import Upscaler, resize_to

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _noop(*a, **k):
    pass


# ------------------------------------------------------------------ 图片
def load_image(path: str) -> torch.Tensor:
    im = Image.open(path)
    im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    import numpy as np
    arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]          # 1,3/4,H,W


def save_image(t: torch.Tensor, path: str):
    import numpy as np
    arr = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    Image.fromarray(arr, mode).save(path)


def list_images(src: str, recursive=True) -> list[Path]:
    p = Path(src)
    if p.is_file():
        return [p]
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted(f for f in it if f.is_file() and f.suffix.lower() in IMG_EXT)


class Canceled(Exception):
    pass


def run_images(src: str, out_dir: str, scale: float = 2.0, model: str = "4x-UltraSharp.pth",
               method: str = "lanczos", device: str | None = None, emit=_noop,
               on_engine=None, should_stop=None) -> dict:
    files = list_images(src)
    if not files:
        emit("error", "没找到图片")
        return {"ok": 0, "fail": 1}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mpath = registry.ensure(model)
    up = Upscaler(str(mpath), device=device)
    if on_engine:
        on_engine({"upscaler": up})
    post = scale / up.scale
    ok = fail = 0
    t_all = time.time()
    emit("job_start", f"图片 {len(files)} 张 | 模型 {model}(x{up.scale:g}) | 目标 {scale:g}x",
         mode="image", total=len(files),
         params={"model": model, "scale": scale, "post_resize": round(post, 4)})
    try:
        for i, f in enumerate(files, 1):
            if should_stop and should_stop():
                raise Canceled()
            stem, suffix = f.stem, f.suffix.lower()
            dst = Path(out_dir) / f"{stem}_{scale:g}x{'.png' if suffix not in ('.jpg', '.jpeg') else suffix}"
            emit("item_start", f"[{i}/{len(files)}] {f.name}", index=i, total=len(files), name=f.name)
            t0 = time.time()
            try:
                x = load_image(str(f))
                y = up.upscale(x)
                y = resize_to(y, post, method)
                save_image(y, str(dst))
                ok += 1
                emit("item_done",
                     f"[{i}/{len(files)}] OK {f.name} -> {dst.name}  {time.time()-t0:.1f}s",
                     index=i, total=len(files), name=f.name, out=dst.name,
                     seconds=round(time.time() - t0, 1),
                     size=[int(y.shape[3]), int(y.shape[2])])
            except Exception as e:
                fail += 1
                emit("item_fail", f"[{i}/{len(files)}] 失败 {f.name}: {e}",
                     index=i, total=len(files), name=f.name, error=str(e))
    finally:
        up.close()
    emit("job_done", f"完成：成功 {ok} / 失败 {fail}，用时 {time.time()-t_all:.0f}s",
         ok=ok, fail=fail, seconds=round(time.time() - t_all))
    return {"ok": ok, "fail": fail}


# ------------------------------------------------------------------ 视频
def frames_per_segment(w: int, h: int, mult: int, budget_gb: float,
                       tile: int = 768) -> tuple[int, str]:
    """每段处理多少**源帧**。

    段大小只由「输出帧张量能否放得下」决定（内存 + INT_MAX）；
    送进模型的批量另有上限，由 Upscaler 内部按 tile 再切一次，两者互不干扰。
    """
    per_out_frame = w * h * 3 * 4                 # 输出帧(已放大)的 float32 字节
    by_mem = int(budget_gb * (1024 ** 3) / max(per_out_frame * max(mult, 1), 1))
    by_int = int(INT_MAX * 0.5 / max(w * h * 3 * max(mult, 1), 1))
    seg = max(4, min(by_mem, by_int))
    return seg, "内存" if by_mem <= by_int else "元素数上限"


def run_video(src: str, out_dir: str, scale: float = 2.0, multiplier: int = 2,
              model: str = "RealESRGAN_x2plus.pth", interp_model: str = "rife_v4.26.safetensors",
              method: str = "lanczos", budget_gb: float = 4.0, max_batch: int = 96,
              crf: int = 18, preset: str = "medium", keep_audio: bool = True,
              device: str | None = None, emit=_noop, on_engine=None, should_stop=None) -> dict:
    if not videoio.have_ffmpeg():
        emit("error", "找不到 ffmpeg / ffprobe，请先安装（brew install ffmpeg）")
        return {"ok": 0, "fail": 1}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    info = videoio.probe(src)
    fps = info["fps"]
    ow, oh = int(round(info["width"] * scale)), int(round(info["height"] * scale))
    mult = max(1, int(multiplier))
    tag = "" if mult == 1 else f"{mult}xfps_"
    dst = Path(out_dir) / f"{Path(src).stem}_{tag}{scale:g}x.mp4"

    emit("item_start",
         f"{Path(src).name}  {info['width']}x{info['height']}@{fps:.2f} → {ow}x{oh}@{fps*mult:.2f}",
         index=1, total=1, name=Path(src).name,
         src={"w": info["width"], "h": info["height"], "fps": round(fps, 2), "frames": info["frames"]},
         out_dims={"w": ow, "h": oh, "fps": round(fps * mult, 2)})

    seg, reason = frames_per_segment(ow, oh, mult, budget_gb, max_batch)
    n_seg = (info["frames"] + seg - 1) // seg
    emit("log", f"分段：每段 {seg} 源帧，共 {n_seg} 段（受{reason}限制）", segs=n_seg)

    up = Upscaler(str(registry.ensure(model)), device=device, budget_gb=budget_gb)
    itp = Interpolator(str(registry.ensure(interp_model)), device=device) if mult > 1 else None
    if on_engine:
        on_engine({"upscaler": up, "interpolator": itp})
    post = scale / up.scale

    tmp_video = str(dst) + ".tmp.mp4"
    writer = videoio.VideoWriter(tmp_video, ow, oh, fps * mult, crf=crf, preset=preset)
    written = 0
    aborted = False
    t0 = time.time()
    try:
        for s in range(n_seg):
            if should_stop and should_stop():
                raise Canceled()
            a = s * seg
            n = min(seg, info["frames"] - a)
            # 第 2 段起向前多取 1 帧，随后丢掉这段的第 1 个输出帧，
            # 这样交界处的「中间帧」也能生成，总帧数严格 = N*mult-1
            overlap = 1 if (s > 0 and mult > 1) else 0
            start = max(0, a - overlap)
            count = n + overlap
            frames = videoio.read_frames(src, info["width"], info["height"], start, count, fps)
            if frames.shape[0] == 0:
                emit("log", f"  段 {s+1}: 读不到帧，跳过")
                continue
            if frames.shape[0] < count:
                emit("log", f"  段 {s+1}: 只读到 {frames.shape[0]}/{count} 帧")
            y = itp.interpolate(frames, mult) if itp else frames
            y = up.upscale(y)
            y = resize_to(y, post, method)
            if overlap:
                y = y[1:]
            writer.write(y)
            written += y.shape[0]
            emit("progress", f"  段 {s+1}/{n_seg} OK {y.shape[0]} 帧 {time.time()-t0:.0f}s",
                 seg=s + 1, segs=n_seg)
    except Canceled:
        aborted = True
        raise
    finally:
        try:
            writer.close()
        except Exception as e:
            emit("item_fail", f"编码失败：{e}", index=1, total=1, name=Path(src).name, error=str(e))
            up.close()
            if itp:
                itp.close()
            Path(tmp_video).unlink(missing_ok=True)
            return {"ok": 0, "fail": 1}
        up.close()
        if itp:
            itp.close()
        if aborted:
            Path(tmp_video).unlink(missing_ok=True)

    videoio.mux_audio(tmp_video, src, str(dst), info["has_audio"] and keep_audio)
    o = videoio.probe(str(dst))
    emit("item_done",
         f"完成 -> {dst.name}  {o['width']}x{o['height']}@{o['fps']:.2f} {o['frames']}f "
         f"时长 {o['duration']:.3f}s 音轨={o['has_audio']}",
         index=1, total=1, name=Path(src).name, out=dst.name,
         info={"w": o["width"], "h": o["height"], "fps": round(o["fps"], 2),
               "frames": o["frames"], "duration": round(o["duration"], 3), "audio": o["has_audio"]})
    emit("job_done", f"完成：用时 {time.time()-t0:.0f}s", ok=1, fail=0, seconds=round(time.time() - t0))
    return {"ok": 1, "fail": 0}
