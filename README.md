# upscale-studio

独立的图片超分 / 视频补帧超分工具。**不依赖 ComfyUI**，不需要开它的画布，装完就能用。

- 超分：spandrel + 自写分块推理（重叠羽化），模型用 ESRGAN 系（`4x-UltraSharp` / `RealESRGAN_x2plus` / …）
- 补帧：RIFE（IFNet），2×/3×/4×，与超分可组合
- 视频：直接调 ffmpeg 解码/编码，**单次编码器常驻**，分段处理不会出现音画漂移
- 首次运行自动检查模型，缺了自动下载（带 sha256 校验、断点续传）
- 自带 Web 界面 + 命令行

## 和 ComfyUI 的关系

完全独立进程、独立代码：只依赖 `torch / spandrel / safetensors / numpy / pillow / aiohttp` + 系统 `ffmpeg`。
如果本机已经装过 ComfyUI，会把 `../ComfyUI/models` 里现成的模型**直接复用**（不重复下载）。

## 启动（macOS / Linux / Windows）

```bash
cd upscale-studio
./run.sh          # macOS / Linux
```
```bat
run.bat           :: Windows
```

- macOS：自动用 MPS；Apple Silicon 之外自动回落 CPU
- Windows：有 NVIDIA 卡时走 CUDA（需装对应 torch 轮子），否则 **CPU 兜底**，功能不缺、只是慢
- 强制指定：`./run.sh --device cpu`（或 `mps` / `cuda`）

它会挑一个带 torch 的 Python（优先项目内 `.venv`，其次 `$UPSCALER_PYTHON`，最后系统 python3），
没有就创建 `.venv` 并装依赖；然后检查模型 → 起服务 → 打开 <http://127.0.0.1:8199>。

只跑命令行不需要起服务：

```bash
python -m upscaler.cli doctor                       # 只检查/下载模型
python -m upscaler.cli image -i ~/Pics/in -o ~/out --scale 2
python -m upscaler.cli video -i a.mp4 -o ~/out --mult 2 --scale 2
python -m upscaler.cli video -i ~/Videos/in -o ~/out --mult 2 --scale 2   # 目录：逐个处理
python -m upscaler.cli video -i a.mp4 -o ~/out --mult 1 --scale 2      # 只超分
```

## 界面

两个页签，输入支持三种方式（上传文件 / 选择文件夹 / 直接填本地路径），输出目录可点按钮选、也可填路径。
长视频自动分段，任务可取消（在分段边界生效），关掉页面任务继续跑。

## 模型

`upscaler/registry.py` 里是清单（名称 / URL / sha256 / 大小 / 用途）。首次启动会逐个校验，
缺失或哈希不符就下载到 `models/`，下载前会先在 `../ComfyUI/models` 找同尺寸同名文件。

| 类型 | 默认 | 其它 |
|---|---|---|
| 超分 | `RealESRGAN_x2plus`（视频）/ `4x-UltraSharp`（图片） | `RealESRGAN_x4plus`、`4x_foolhardy_Remacri`、`4xPSNR` |
| 补帧 | `rife_v4.26` | `rife_v4.25_lite` |

## 两个必须知道的限制

1. **MPS 单张量元素数上限 2^31-1**。超分/补帧都是「一批帧一起进模型」，帧数太多会抛
   `MPSGraph does not support tensor dims larger than INT_MAX`——**和内存够不够无关**。
   本项目按元素数 + 内存预算自动切段（默认一次不超过 96 帧），所以不会撞。
2. **取消是协作式的**：在图片张与张之间、视频段与段之间检查，正在跑的那一段会跑完。
   取消后临时文件会被清掉，不会留半个视频。
3. **tile=768 时单批不能超过 28 帧**：实测 32 帧会让 MPS **静默把末帧算成全黑**（不报错）。
   代码已把上限压到 24，并内建「结果帧几乎全黑就抛错」的检测，宁可报错也不吐坏帧。

## 目录

```
upscale-studio/
├── run.sh                     # 一键启动
├── pyproject.toml
├── upscaler/
│   ├── device.py              # 设备/内存策略（含 MPS INT_MAX 夹紧）
│   ├── registry.py            # 模型清单 + 校验 + 自动下载
│   ├── upscale.py             # spandrel + 分块推理
│   ├── interp.py              # RIFE(IFNet)
│   ├── videoio.py             # ffmpeg 解码/编码/混音
│   ├── pipeline.py            # 图片/视频流水线（CLI 与 WebUI 共用）
│   ├── cli.py
│   └── webui/{server.py, static/index.html}
├── models/                    # 自动下载到这里（也可复用 ComfyUI 的）
└── workspace/                 # uploads / outputs / settings.json
```

## 与 ComfyUI 版本的正面对比

同一段实拍素材（1280×720@60fps，取 1 秒共 60 帧），
同参数（2 倍补帧 + 2 倍超分，`RealESRGAN_x2plus` + `rife_v4.26`），在同一台 Apple Silicon 机器上先后跑：

| 指标 | ComfyUI 版 | 独立版 |
|---|---|---|
| 输出规格 | 2560×1440@120fps，119 帧，带音轨 | 完全相同 |
| 耗时 | 280 s | **277 s** |
| 画质差异 | — | PSNR **44.1 dB** / SSIM **0.990**（肉眼不可区分） |
| 依赖 | ComfyUI + 一堆 custom node/前端 | torch/spandrel + ffmpeg |
| 启动 | 先起服务、开浏览器、加载工作流 | `./run.sh` |

样片在 `workspace/compare/`（`source_1s.mp4` / `comfyui.mp4` / `standalone.mp4`）。

## 性能（Apple Silicon / MPS）

| 任务 | 耗时 |
|---|---|
| 160×120 → 320×240 | ≈0.3 s |
| 720p 60 帧，2 倍补帧 + 2 倍超分（→1440p/119 帧） | 277 s（≈2.3 s/帧） |
| 160×120 96 帧，10 段拼接 | ≈9 s，输出 191 帧、时长与源一致 |

CPU 兜底大约慢 7~10 倍（小图 2 s vs 0.3 s），可用但别指望跑长视频。

## License

MIT © 2026 SugarCatCat
