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
import time
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


_BIN: dict = {}


def _candidates() -> list[str]:
    """ffmpeg 候选：环境变量 > PATH > 常见安装位置（有的 PATH 里是包装器/残缺构建）。"""
    env = os.environ.get("UPSCALER_FFMPEG")
    cands = [env] if env else []
    cands += [shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg",
              "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
    return [c for c in cands if c and Path(c).exists()]


def ffmpeg_path() -> str:
    if _BIN.get("ffmpeg"):
        return _BIN["ffmpeg"]
    for c in _candidates():
        try:
            r = subprocess.run([c, "-hide_banner", "-version"], capture_output=True,
                               timeout=10, creationflags=_NO_WINDOW)
            if r.returncode == 0:
                _BIN["ffmpeg"] = c
                return c
        except Exception:
            continue
    _BIN["ffmpeg"] = shutil.which("ffmpeg") or "ffmpeg"
    return _BIN["ffmpeg"]


def ffprobe_path() -> str:
    if _BIN.get("ffprobe"):
        return _BIN["ffprobe"]
    sibling = Path(ffmpeg_path()).with_name("ffprobe")
    p = str(sibling) if sibling.exists() else (shutil.which("ffprobe") or "ffprobe")
    _BIN["ffprobe"] = p
    return p


def have_ffmpeg() -> bool:
    return Path(ffmpeg_path()).exists() and Path(ffprobe_path()).exists()


_CAPS: dict = {}


def encoder_caps() -> dict:
    """探测本机 ffmpeg 是否支持 libx264 的 -preset/-crf（探一次，缓存）。

    有的 ffmpeg 构建/包装器会拒掉这两个选项（报 Unrecognized option 'preset'），
    探测后按实际情况拼命令，而不是想当然地传。
    """
    if _CAPS:
        return _CAPS
    caps = {"libx264": False, "preset": False, "crf": False, "version": ""}
    try:
        v = subprocess.run([ffmpeg_path(), "-hide_banner", "-version"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=_NO_WINDOW)
        caps["version"] = (v.stdout or "").splitlines()[0] if v.stdout else ""
        e = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=_NO_WINDOW)
        caps["libx264"] = "libx264" in (e.stdout or "")
    except Exception as ex:
        caps["error"] = str(ex)
    if caps["libx264"]:
        try:
            r = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
            caps["preset"] = caps["crf"] = r.returncode == 0
            if r.returncode != 0:
                caps["probe_error"] = (r.stderr or "")[:300]
        except Exception as ex:
            caps["probe_error"] = str(ex)
    _CAPS.update(caps)
    return _CAPS


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
    d = _run_json([ffprobe_path(), "-v", "error", "-print_format", "json",
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
    cmd = [ffmpeg_path(), "-v", "error", "-nostdin",
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
        caps = encoder_caps()
        if not caps.get("libx264"):
            raise RuntimeError(
                f"当前 ffmpeg 不支持 libx264 编码（{ffmpeg_path()}）。"
                f"请安装完整版：brew install ffmpeg。探测信息：{caps.get('version', '')}")

        vf = []
        if width % 2 or height % 2:                     # yuv420p 要求偶数
            vf = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        cmd = [ffmpeg_path(), "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
               *vf, "-c:v", "libx264"]
        if caps.get("preset"):
            cmd += ["-preset", preset]
        if caps.get("crf"):
            cmd += ["-crf", str(crf)]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
        self.cmd = cmd

        self.err = tempfile.TemporaryFile()
        self.proc = _popen(cmd, stdin=subprocess.PIPE, stderr=self.err)

        # 启动即校验：ffmpeg 若在解析参数时就退出，这里立刻抛错，
        # 而不是等一整个分段算完、写管道时才炸（那样会白跑几分钟）。
        deadline = time.time() + 0.3
        while time.time() < deadline and self.proc.poll() is None:
            time.sleep(0.02)
        if self.proc.poll() is not None:
            self.err.seek(0)
            msg = self.err.read().decode("utf-8", "ignore")[:800]
            raise RuntimeError(f"ffmpeg 启动失败（{ffmpeg_path()}）：{msg.strip()}\n"
                               f"命令：{' '.join(cmd)}")

    def write(self, frames: torch.Tensor):
        if frames.numel() == 0:
            return
        if self.proc.poll() is not None:               # 编码器中途挂了，别硬写
            self.err.seek(0)
            msg = self.err.read().decode("utf-8", "ignore")[:800]
            raise RuntimeError(f"ffmpeg 编码中断（{ffmpeg_path()}）：{msg.strip()}\n"
                               f"命令：{' '.join(self.cmd)}")
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
            raise RuntimeError(f"ffmpeg 编码失败（{ffmpeg_path()}）：{msg.strip()}\n"
                               f"命令：{' '.join(self.cmd)}")
        self.err.close()


def mux_audio(video: str, source: str, out: str, has_audio: bool):
    """把源音轨混进编码结果（视频流直接 copy，不二次压缩）。"""
    if not has_audio:
        Path(video).replace(out)
        return
    cmd = [ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", video, "-i", source,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("混流失败: " + r.stderr.decode("utf-8", "ignore")[:400])
    Path(video).unlink(missing_ok=True)
