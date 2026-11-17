"""命令行入口： python -m upscaler.cli {doctor,image,video}"""
from __future__ import annotations

import argparse
import os
import sys

from . import registry
from .pipeline import run_images, run_videos


def _printer():
    def emit(kind, msg="", **data):
        if msg:
            print(msg, flush=True)
    return emit


def main(argv=None):
    ap = argparse.ArgumentParser(prog="upscaler", description="独立超分 / 补帧工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--doctor", action="store_true", help=argparse.SUPPRESS)

    d = sub.add_parser("doctor", help="检查模型是否齐全（缺了自动下载）")
    d.add_argument("--quiet", action="store_true")

    def common(p):
        p.add_argument("-i", "--input", required=True, help="文件或目录")
        p.add_argument("-o", "--output", required=True, help="输出目录")
        p.add_argument("--scale", type=float, default=2, help="最终倍率（默认 2）")
        p.add_argument("--device", default=None, help="mps / cuda / cpu（默认自动）")
        p.add_argument("--method", default="lanczos", help="后缩放算法")

    pi = sub.add_parser("image", help="图片超分")
    common(pi)
    pi.add_argument("--model", default="4x-UltraSharp.pth")

    pv = sub.add_parser("video", help="视频补帧 + 超分")
    common(pv)
    pv.add_argument("--model", default="realesr-animevideov3.pth",
                    help="默认轻量模型（快 30 倍）；要极致细节换 RealESRGAN_x4plus.pth")
    pv.add_argument("--interp", default="rife_v4.26.safetensors")
    pv.add_argument("--mult", type=int, default=2, help="补帧倍率，1=只超分")
    pv.add_argument("--budget-gb", type=float, default=4.0)
    pv.add_argument("--max-batch", type=int, default=96)
    pv.add_argument("--crf", type=int, default=18)
    pv.add_argument("--preset", default="medium")
    pv.add_argument("--no-audio", action="store_true")

    a = ap.parse_args(argv)
    emit = _printer()

    if a.cmd == "doctor":
        registry.ensure_all(quiet=getattr(a, "quiet", False))
        return 0

    if a.cmd == "image":
        r = run_images(a.input, a.output, scale=a.scale, model=a.model,
                       method=a.method, device=a.device, emit=emit)
    else:
        r = run_videos(a.input, a.output, scale=a.scale, multiplier=a.mult,
                      model=a.model, interp_model=a.interp, method=a.method,
                      budget_gb=a.budget_gb, max_batch=a.max_batch, crf=a.crf,
                      preset=a.preset, keep_audio=not a.no_audio,
                      device=a.device, emit=emit)
    return 0 if r["fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
