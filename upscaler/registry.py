"""模型清单 + 首次启动自检 + 自动下载。

优先复用本机已有的模型（默认会去 ../ComfyUI/models 里找），找不到或者哈希不对才下载。
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# 本机可能已经有一份的目录（按顺序找）
SEARCH_DIRS = {
    "upscale": [MODELS / "upscale_models", ROOT.parent / "ComfyUI" / "models" / "upscale_models"],
    "interp": [MODELS / "frame_interpolation", ROOT.parent / "ComfyUI" / "models" / "frame_interpolation"],
}

HF = "https://huggingface.co"
GH = "https://github.com"

MANIFEST = [
    # ---- 超分 ----
    dict(name="RealESRGAN_x2plus.pth", kind="upscale", scale=2, size=67061725,
         sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
         url=f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
         note="视频默认：原生 2 倍，速度与稳定性最好"),
    dict(name="RealESRGAN_x4plus.pth", kind="upscale", scale=4, size=67040989,
         sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
         url=f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
         note="通用 4 倍"),
    dict(name="4x-UltraSharp.pth", kind="upscale", scale=4, size=66961958,
         sha256="a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
         url=f"{HF}/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
         note="图片默认：锐利，适合二次元/截图"),
    dict(name="4x_foolhardy_Remacri.pth", kind="upscale", scale=4, size=67025055,
         sha256="e1a73bd89c2da1ae494774746398689048b5a892bd9653e146713f9df8bca86a",
         url=f"{HF}/uwg/upscaler/resolve/main/ESRGAN/4x_foolhardy_Remacri.pth",
         note="自然，适合人像/写实"),
    dict(name="4xPSNR.pth", kind="upscale", scale=4, size=66954395,
         sha256="985fba37421353716ef2fbd7fbbab061e76409994d168cf52527aa8ee53f81ec",
         url=f"{HF}/uwg/upscaler/resolve/main/ESRGAN/4xPSNR.pth",
         note="最保守保真"),
    # ---- 补帧 ----
    dict(name="rife_v4.26.safetensors", kind="interp", scale=1, size=22674688,
         sha256="151874592c877740e5db11522f4514df569eeafb0a0fcb2696f16e9e8d317c94",
         url=f"{HF}/Comfy-Org/Frame-Interpolation/resolve/main/frame_interpolation/rife_v4.26.safetensors",
         note="默认补帧模型"),
    dict(name="rife_v4.25_lite.safetensors", kind="interp", scale=1, size=22506384,
         sha256="e5e5fe0286d30708f4c36aa23639a38d3d7cd0c724922c66e6b04130ae12c6e4",
         url=f"{HF}/Comfy-Org/Frame-Interpolation/resolve/main/frame_interpolation/rife_v4.25_lite.safetensors",
         note="更快的补帧模型"),
]

BY_NAME = {m["name"]: m for m in MANIFEST}


def sha256_of(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_existing(name: str) -> Path | None:
    for d in SEARCH_DIRS[BY_NAME[name]["kind"]]:
        p = d / name
        if p.is_file() and p.stat().st_size == BY_NAME[name]["size"]:
            return p
    return None


def verify(path: Path, meta: dict) -> bool:
    return path.is_file() and path.stat().st_size == meta["size"] and sha256_of(path) == meta["sha256"]


def download(name: str, dest_dir: Path | None = None, quiet=False) -> Path:
    meta = BY_NAME[name]
    dest_dir = dest_dir or SEARCH_DIRS[meta["kind"]][0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    part = dest.with_suffix(dest.suffix + ".part")

    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(meta["url"], headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0
            part.unlink(missing_ok=True)
        total = meta["size"]
        mode = "ab" if have else "wb"
        done = have
        with open(part, mode) as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if not quiet and total:
                    pct = done * 100 // total
                    print(f"\r  下载 {name}  {pct:3d}%  {done/1048576:.0f}/{total/1048576:.0f} MB",
                          end="", flush=True)
    if not quiet:
        print()
    part.replace(dest)
    return dest


def ensure(name: str, quiet=False) -> Path:
    """返回可用模型路径：本机有就直接用，没有就下。"""
    meta = BY_NAME[name]
    p = find_existing(name)
    if p and verify(p, meta):
        return p
    if p and not verify(p, meta):
        print(f"  {name} 校验失败，重新下载", file=sys.stderr)
    p = download(name, quiet=quiet)
    if not verify(p, meta):
        raise RuntimeError(f"{name} 下载后校验失败")
    return p


def ensure_all(quiet=False) -> dict:
    """首启自检：把清单里所有模型都准备好。"""
    print(f"模型目录 {MODELS}（会复用 ../ComfyUI/models 里已有的）")
    out = {}
    for m in MANIFEST:
        name = m["name"]
        p = find_existing(name)
        if p and verify(p, m):
            print(f"  ✓ {name}  {m['note']}")
            out[name] = p
        else:
            print(f"  ↓ {name}  {m['note']}")
            out[name] = ensure(name, quiet=quiet)
    return out


if __name__ == "__main__":
    ensure_all()
