"""设备与内存策略。

Apple Silicon 上有个必须记住的硬限制：MPSGraph 单个张量的元素数不能超过 INT_MAX(2^31-1)，
超了会抛 `MPSGraph does not support tensor dims larger than INT_MAX`，而且跟显存是否够用无关。
所以所有「一次送进模型」的批次都要按元素数夹紧，本项目在 upscale/interp 里都做了。
"""
from __future__ import annotations

import os

import torch

INT_MAX = 2 ** 31 - 1


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """MPS / CUDA 上用 fp16；CPU 上必须 fp32（不然很多算子会崩或极慢）。"""
    return torch.float16 if device.type in ("mps", "cuda") else torch.float32


def apply_mps_watermark():
    """给 MPS 设个水位，避免一路吃到把整机拖进 swap。必须在 import torch 之前调用才完全生效，
    但因为 torch 是惰性申请，这里设置仍然有用。"""
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")


def mps_batch_limit(tile: int, scale: int = 1) -> int:
    """单次送进模型的帧数上限。

    MPSGraph 在张量过大时不是报错，而是**直接段错误**（EXC_BAD_ACCESS in _mps_convolution），
    所以必须按模型倍率一起算：内部特征图 ≈ 帧数 × 64通道 × (tile × scale)²。

    基准来自实测：tile=768 / scale=2 / 24 帧 = 3.6e9 元素能稳定跑；
    scale=4 时同样 24 帧会到 1.45e10，直接崩。这里按 3.2e9 留余量。
    """
    area = (tile * scale) ** 2
    by_elems = int(3.2e9 / (64 * area)) if area else 64
    base = 64 if tile <= 512 else (24 if tile <= 768 else 16)
    return max(1, min(base, by_elems))


def frames_per_batch(h: int, w: int, out_h: int, out_w: int, budget_gb: float = 3.0,
                     tile: int = 768, scale: int = 1) -> int:
    """按「输出帧张量占多少内存」和「INT_MAX 元素数」算一次能处理多少帧。"""
    per_frame_out = out_h * out_w * 3 * 4          # float32 累加缓冲
    by_mem = int(budget_gb * (1024 ** 3) / max(per_frame_out, 1))
    by_int = int(INT_MAX * 0.5 / max(out_h * out_w * 3, 1))
    return max(1, min(by_mem, by_int, mps_batch_limit(tile, scale)))
