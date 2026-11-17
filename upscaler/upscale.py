"""超分：spandrel 加载模型 + 自写分块推理（重叠羽化）。

为什么不用整图一次推理：
  - 大图/长视频一次放不下，且 MPS 有 INT_MAX 元素数上限；
  - 分块后内存与分辨率解耦，且结果与整图推理一致（重叠区域羽化融合）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from spandrel import ModelLoader

from .device import INT_MAX, frames_per_batch, pick_device, pick_dtype


def _axis_weights(n: int, ov: int, left: bool, right: bool) -> torch.Tensor:
    w = torch.ones(n, dtype=torch.float32)
    if ov > 0:
        ramp = torch.linspace(0.0, 1.0, ov + 2)[1:-1]      # 不含 0/1，避免出现全零权重
        if left:
            w[:ov] = ramp
        if right:
            w[-ov:] = torch.flip(ramp, [0])
    return w


FULL_FRAME_ELEMS = int(1.2e9)      # 整帧推理时内部特征图的元素上限（MPS 实测安全区）


class Upscaler:
    def __init__(self, model_path: str, device: str | None = None,
                 tile: int = 768, overlap: int = 48, budget_gb: float = 3.0):
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        desc = ModelLoader().load_from_file(model_path)
        self.scale = float(desc.scale)
        self.model = desc.model.eval().to(self.device, dtype=self.dtype)
        self.tile = tile
        self.overlap = overlap
        self.budget_gb = budget_gb
        self._wcache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ 权重
    def _weight_map(self, h: int, w: int) -> tuple[torch.Tensor, list[int], list[int]]:
        """返回 (整幅图的权重和, ys, xs)。权重和只用于最后归一化，
        每个 tile 自己的窗口权重在 _tile_weight 里算——两者不能混用。"""
        key = (h, w, self.tile, self.overlap)
        if key in self._wcache:
            return self._wcache[key]
        s, tile, ov = int(self.scale), self.tile, self.overlap
        ys, xs = self._starts(h), self._starts(w)
        acc = torch.zeros(h * s, w * s, dtype=torch.float32)
        for y in ys:
            th = min(tile, h - y)
            wy = _axis_weights(th * s, ov * s if y > 0 else 0, y > 0, y + tile < h)
            for x in xs:
                tw = min(tile, w - x)
                wx = _axis_weights(tw * s, ov * s if x > 0 else 0, x > 0, x + tile < w)
                acc[y * s:y * s + th * s, x * s:x * s + tw * s] += torch.outer(wy, wx)
        acc.clamp_min_(1e-6)
        self._wcache = {key: (acc, ys, xs)}
        return self._wcache[key]

    def _starts(self, total: int) -> list[int]:
        tile, ov = self.tile, self.overlap
        if total <= tile:
            return [0]
        step = max(1, tile - ov)
        xs = list(range(0, total - tile, step))
        if xs[-1] != total - tile:
            xs.append(total - tile)
        return xs

    def _tile_weight(self, y: int, x: int, h: int, w: int) -> torch.Tensor:
        """单个 tile 自己的窗口权重（羽化用），与位置有关。"""
        s, tile, ov = int(self.scale), self.tile, self.overlap
        th, tw = min(tile, h - y), min(tile, w - x)
        wy = _axis_weights(th * s, ov * s if y > 0 else 0, y > 0, y + tile < h)
        wx = _axis_weights(tw * s, ov * s if x > 0 else 0, x > 0, x + tile < w)
        return torch.outer(wy, wx)

    def _upscale_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: (N,3,H,W) float32 [0,1]（CPU），返回放大结果（CPU float32）。

        整块先搬到设备，再逐 tile 切片推理——避免每个 tile 都做一次 CPU↔设备拷贝。
        """
        n, c, h, w = chunk.shape
        s, tile, ov = int(self.scale), self.tile, self.overlap
        dev = chunk.to(self.device, self.dtype)
        # 整帧推理比分块快 20~26%（省掉 tile 边界和多次调用），
        # 只要内部特征图放得下就别分块；放不下再退回 tile。
        full_frame = 64 * (h * s) * (w * s) <= FULL_FRAME_ELEMS
        if full_frame or (h <= tile and w <= tile):
            with torch.no_grad():
                out = self.model(dev).float().clamp_(0, 1)
            self._check_frames(chunk, out)
            return out.cpu()

        wmap, ys, xs = self._weight_map(h, w)
        out = torch.zeros((n, c, h * s, w * s), dtype=torch.float32)
        for y in ys:
            th = min(tile, h - y)
            for x in xs:
                tw = min(tile, w - x)
                with torch.no_grad():
                    up = self.model(dev[:, :, y:y + th, x:x + tw])
                    # 在设备上按 fp16 相乘（不做 float 转换、不逐块 clamp），
                    # 传回来的数据量减半；clamp 放到最后统一做。
                    ws = self._tile_weight(y, x, h, w).to(self.device, up.dtype)
                    out[:, :, y * s:(y + th) * s, x * s:(x + tw) * s] += (up * ws).cpu()
        out /= wmap
        out.clamp_(0, 1)
        self._check_frames(chunk, out)
        return out

    @staticmethod
    def _check_frames(src: torch.Tensor, out: torch.Tensor):
        """MPS 在批次过大时可能"静默"把某一帧算成全黑（不报错），这里显式拦截。"""
        if out.shape[0] < 2:
            return
        a = src.mean(dim=(1, 2, 3)).float().cpu()
        b = out.mean(dim=(1, 2, 3)).float().cpu()
        bad = (a > 0.02) & (b < a * 0.2)
        if bool(bad.any()):
            idx = [int(i) for i in bad.nonzero().flatten()]
            raise RuntimeError(
                f"超分结果异常：第 {idx} 帧几乎全黑（疑似 MPS 批次过大导致的静默丢帧）。"
                f"请减小 --max-batch 或 tile，当前 tile={src.shape[-1]}")

    def upscale(self, frames: torch.Tensor, on_progress=None) -> torch.Tensor:
        """frames: (N,3,H,W) float32 [0,1]（CPU）。返回 (N,3,H*s,W*s)。"""
        n, _, h, w = frames.shape
        oh, ow = int(h * self.scale), int(w * self.scale)
        chunk = frames_per_batch(h, w, oh, ow, self.budget_gb, self.tile, int(self.scale))
        if 64 * oh * ow <= FULL_FRAME_ELEMS:
            # 整帧推理时单次调用已经吃满 GPU，实测批量没有额外收益（±1%），
            # 逐帧调用还能把内存峰值压到最低（每帧内部张量 ≈ 64×输出像素×2 字节）。
            chunk = 1
        outs = []
        for i in range(0, n, chunk):
            outs.append(self._upscale_chunk(frames[i:i + chunk]))
            if on_progress:
                on_progress(min(i + chunk, n), n)
        return torch.cat(outs, 0) if len(outs) > 1 else outs[0]

    def close(self):
        self.model = None
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def resize_to(tensor: torch.Tensor, scale: float, method: str = "lanczos") -> torch.Tensor:
    """把结果再缩放到目标倍率（模型倍率 × 这个系数 = 最终倍率）。"""
    if abs(scale - 1.0) < 1e-6:
        return tensor
    n, c, h, w = tensor.shape
    oh, ow = int(round(h * scale)), int(round(w * scale))
    m = {"lanczos": "bilinear", "bicubic": "bicubic", "area": "area",
         "bilinear": "bilinear", "nearest": "nearest"}.get(method, "bilinear")
    out = F.interpolate(tensor, size=(oh, ow), mode=m, align_corners=False,
                        antialias=True) if m in ("bilinear", "bicubic") else \
        F.interpolate(tensor, size=(oh, ow), mode=m)
    return out.clamp_(0, 1)
