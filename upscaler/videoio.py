"""视频读写：直接和 ffmpeg 打交道，不依赖 PyAV。

- 解码：ffmpeg 输出 rawvideo(rgb24) 到管道，按帧读，帧数精确可控
- 编码：单次编码器常驻，逐段写入同一路输出（避免多段拼接导致的帧数漂移）
- 音轨：最后把源文件的音轨混流进去
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


# Windows 下不弹黑框
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _popen(args, **kw):
    kw.setdefault("creationflags", _NO_WINDOW)
    return subprocess.Popen(args, **kw)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run_json(args: list[str]) -> dict:
    out = subprocess.run(args, capture_output=True, check=True,
                         creationflags=_NO_WINDOW).stdout
    return json.loads(out)


def probe(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"文件不存在：{path}")
    if not p.is_file():
        raise RuntimeError(f"这不是一个视频文件：{path}（目录里的视频会逐个处理，不应直接探测目录）")
    d = _run_json(["ffprobe", "-v", "error", "-print_format", "json",
                   "-show_format", "-show_streams", path])
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = (v.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den or 1) else 0.0
    if fps <= 0:
        num, den = (v.get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 25.0
    nb = int(v.get("nb_frames") or 0)
    dur = float(d["format"].get("duration") or 0)
    if nb <= 0:
        nb = int(round(dur * fps))
    return {
        "fps": fps, "frames": nb, "duration": dur,
        "width": int(v["width"]), "height": int(v["height"]),
        "rotation": _rotation(v),
        "has_audio": any(s["codec_type"] == "audio" for s in d["streams"]),
    }


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            return int(sd["rotation"])
    return int(stream.get("tags", {}).get("rotate", 0) or 0)


def read_frames(path: str, width: int, height: int, start_frame: int, count: int,
                fps: float) -> torch.Tensor:
    """按帧号精确读取 count 帧，返回 (N,3,H,W) float32 [0,1]。

    用 -ss（放在 -i 前，配合转码会自动精确 seek）+ -frames:v 控制帧数。
    """
    ss = max(0.0, start_frame / fps)
    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           "-ss", f"{ss:.6f}", "-i", path,
           "-frames:v", str(count), "-an", "-sn",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = width * height * 3
    buf = bytearray(frame_bytes * count)
    with tempfile.TemporaryFile() as err:
        p = _popen(cmd, stdout=subprocess.PIPE, stderr=err)
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            n = p.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        p.stdout.close()
        rc = p.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg 解码失败: {err.read().decode('utf-8', 'ignore')[:400]}")
    frames = got // frame_bytes
    if frames == 0:
        return torch.empty((0, 3, height, width), dtype=torch.float32)
    arr = np.frombuffer(buf, dtype=np.uint8, count=frames * frame_bytes)
    arr = arr.reshape(frames, height, width, 3)
    t = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    return t


class VideoWriter:
    """常驻编码器：一路 output，可分段写入（同一时间轴）。"""

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 crf: int = 18, preset: str = "medium"):
        self.out_path = out_path
        self.w, self.h = width, height
        self.err = tempfile.TemporaryFile()
        vf = []
        if width % 2 or height % 2:                     # yuv420p 要求偶数
            vf = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
               *vf, "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        self.proc = _popen(cmd, stdin=subprocess.PIPE, stderr=self.err)

    def write(self, frames: torch.Tensor):
        if frames.numel() == 0:
            return
        arr = (frames.clamp(0, 1) * 255).round().to(torch.uint8)
        arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        self.proc.stdin.write(arr.tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        if rc != 0:
            self.err.seek(0)
            msg = self.err.read().decode("utf-8", "ignore")[:800]
            raise RuntimeError(f"ffmpeg 编码失败: {msg}")
        self.err.close()


def mux_audio(video: str, source: str, out: str, has_audio: bool):
    """把源音轨混进编码结果（视频流直接 copy，不二次压缩）。"""
    if not has_audio:
        Path(video).replace(out)
        return
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", video, "-i", source,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("混流失败: " + r.stderr.decode("utf-8", "ignore")[:400])
    Path(video).unlink(missing_ok=True)
