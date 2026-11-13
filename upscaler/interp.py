"""RIFE 补帧（IFNet）——从 ComfyUI 的实现移植，去掉 comfy.* 依赖。

权重直接用 Comfy-Org 重打包的 rife_v4.26.safetensors（键名已是 blocks.N.*）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import pick_device, pick_dtype


# ---------------------------------------------------------------- 网络结构
def _warp(img, flow, grids):
    B, _, H, W = img.shape
    base, div = grids[(H, W)]
    f = torch.cat([flow[:, 0:1] / div[0], flow[:, 1:2] / div[1]], 1).float()
    grid = (base.expand(B, -1, -1, -1) + f).permute(0, 2, 3, 1)
    return F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=True).to(img.dtype)


class Head(nn.Module):
    def __init__(self, out_ch=4):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, out_ch, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(torch.addcmul(x, self.conv(x), self.beta))


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(nn.Conv2d(in_planes, c // 2, 3, 2, 1), nn.LeakyReLU(0.2, True)),
            nn.Sequential(nn.Conv2d(c // 2, c, 3, 2, 1), nn.LeakyReLU(0.2, True)))
        self.convblock = nn.Sequential(*(ResConv(c) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    pad_align = 64

    def __init__(self, head_ch=4, channels=(192, 128, 96, 64, 32)):
        super().__init__()
        self.encode = Head(out_ch=head_ch)
        block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
        self.blocks = nn.ModuleList([IFBlock(block_in[i], channels[i]) for i in range(5)])
        self.scale_list = [16, 8, 4, 2, 1]
        self._grids = {}

    def _build_grids(self, H, W, device):
        if (H, W) in self._grids:       # 同一分辨率复用，避免每次 forward 重建 meshgrid
            return
        self._grids = {}
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij")
        self._grids[(H, W)] = (torch.stack((gx, gy), 0).unsqueeze(0),
                               torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0],
                                            dtype=torch.float32, device=device))

    def extract(self, img):
        return self.encode(img)

    def forward(self, img0, img1, timestep=0.5, cache=None):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]),
                                  timestep, device=img0.device, dtype=img0.dtype)
        self._build_grids(img0.shape[2], img0.shape[3], img0.device)
        B = img0.shape[0]
        f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else self.encode(img0)
        f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else self.encode(img1)
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None,
                                         scale=self.scale_list[i])
            else:
                fd, mask, feat = block(
                    torch.cat((w0, w1, _warp(f0, flow[:, :2], self._grids),
                               _warp(f1, flow[:, 2:4], self._grids), timestep, mask, feat), 1),
                    flow, scale=self.scale_list[i])
                flow = flow.add_(fd)
            w0 = _warp(img0, flow[:, :2], self._grids)
            w1 = _warp(img1, flow[:, 2:4], self._grids)
        return torch.lerp(w1, w0, torch.sigmoid(mask))


def detect_config(sd):
    head_ch = sd["encode.cnn3.weight"].shape[1]
    channels = [sd[f"blocks.{i}.conv0.1.0.weight"].shape[0] for i in range(5)]
    return head_ch, tuple(channels)


# ---------------------------------------------------------------- 推理封装
class Interpolator:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        sd = torch.load(model_path, map_location="cpu", weights_only=True) \
            if model_path.endswith(".pth") else _load_safetensors(model_path)
        head_ch, ch = detect_config(sd)
        self.model = IFNet(head_ch, ch).eval()
        self.model.load_state_dict(sd)
        self.model.to(self.device, dtype=self.dtype)
        self.align = self.model.pad_align

    def interpolate(self, frames: torch.Tensor, multiplier: int, on_progress=None) -> torch.Tensor:
        """frames: (N,3,H,W) float32 [0,1]。返回 N*multiplier-1 帧。"""
        n, c, h, w = frames.shape
        if multiplier < 2 or n < 2:
            return frames

        pad_h = (self.align - h % self.align) % self.align
        pad_w = (self.align - w % self.align) % self.align
        if pad_h or pad_w:
            x = F.pad(frames, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x = frames

        out = torch.empty((n - 1) * multiplier + 1, c, h + pad_h, w + pad_w,
                          dtype=torch.float32)
        out[0] = x[0]
        idx = 1
        cache = {}
        steps = total = (n - 1) * (multiplier - 1)
        done = 0
        for i in range(n - 1):
            i0 = x[i:i + 1].to(self.device, self.dtype)
            i1 = x[i + 1:i + 2].to(self.device, self.dtype)
            prev = cache.pop("next", None)
            cache["img0"] = prev if prev is not None else self.model.extract(i0)
            cache["img1"] = self.model.extract(i1)
            cache["next"] = cache["img1"]
            for t in range(1, multiplier):
                with torch.no_grad():
                    mid = self.model(i0, i1, timestep=t / multiplier, cache=cache)
                out[idx] = mid[0].float().cpu().clamp_(0, 1)
                idx += 1
                done += 1
                if on_progress and done % 8 == 0:
                    on_progress(done, total)
            out[idx] = x[i + 1]
            idx += 1
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out

    def close(self):
        self.model = None
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def _load_safetensors(path: str) -> dict:
    from safetensors.torch import load_file
    sd = load_file(path)
    remap = {}
    for k in sd:
        for i in range(5):
            if k.startswith(f"block{i}."):
                remap[k] = f"blocks.{i}.{k[len(f'block{i}.'):]}"
    return {remap.get(k, k): v for k, v in sd.items()
            if not k.startswith(("teacher.", "caltime."))}
