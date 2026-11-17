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

选好文件或填入路径后，参数区下方会显示**耗时预估**（如「预计 1 小时 38 分 · 720 帧 × 8.2 秒/帧 · 17 段 · 输出 3840×2160@24fps」），
调倍率/模型/补帧会实时刷新。系数来自本机实测（720p 源：2 倍模型 ≈2.0 秒/帧、4 倍模型 ≈8.5 秒/帧，
编码 ≈0.1 秒/4K 帧），CPU 模式按 8 倍估算，误差约 ±20%。
长视频自动分段，任务可取消（在分段边界生效），关掉页面任务继续跑。

## 模型

`upscaler/registry.py` 里是清单（名称 / URL / sha256 / 大小 / 用途）。首次启动会逐个校验，
缺失或哈希不符就下载到 `models/`，下载前会先在 `../ComfyUI/models` 找同尺寸同名文件。

| 类型 | 默认 | 其它 |
|---|---|---|
| 超分 | `realesr-animevideov3`（视频，轻量）/ `4x-UltraSharp`（图片） | `realesr-general-x4v3`、`RealESRGAN_x2plus`、`RealESRGAN_x4plus`、`4x_foolhardy_Remacri`、`4xPSNR` |
| 补帧 | `rife_v4.26` | `rife_v4.25_lite` |

**轻量模型（SRVGGNetCompact，0.6~1.2M 参数）是 Mac 上的关键提速手段**：比 RRDBNet 小 14~27 倍，
720p→4K 单帧 **175 ms**（RRDBNet x4plus 是 5879 ms），保真度只低约 1 dB，1:1 对比几乎看不出差别。
素材是实拍就换 `realesr-general-x4v3`（速度约为 RRDBNet 的 15 倍）。

## 两个必须知道的限制

1. **MPS 单张量元素数上限 2^31-1**。超分/补帧都是「一批帧一起进模型」，帧数太多会抛
   `MPSGraph does not support tensor dims larger than INT_MAX`——**和内存够不够无关**。
   本项目按元素数 + 内存预算自动切段（默认一次不超过 96 帧），所以不会撞。
2. **取消是协作式的**：在图片张与张之间、视频段与段之间检查，正在跑的那一段会跑完。
   取消后临时文件会被清掉，不会留半个视频。
3. **conda 环境自带的 ffmpeg 没有 libx264**（conda-forge 不含这个 GPL 编码器）。
   程序会扫描候选二进制、按可用编码器挑选：`libx264` → `h264_videotoolbox`（Mac 硬件）→
   `libopenh264` → `mpeg4`，并把结果写进任务日志的 `[env]` 行。
   想手动指定：`UPSCALER_FFMPEG=/opt/homebrew/bin/ffmpeg`、`UPSCALER_ENCODER=videotoolbox`。
   用 videotoolbox 编码在 4K 下比 x264 快很多（硬件编码），画质略逊但差别不大。
4. **编码器能力是探测出来的，不是假设的**：有的 ffmpeg 构建/包装器不认 `-preset`/`-crf`，
   启动时会先跑一个 64×64 的空编码探测，不支持的选项自动去掉；ffmpeg 若在解析参数阶段就退出，
   `VideoWriter` 会在 0.3 秒内报错并把**完整命令**写进异常里（不会再白跑一整个分段）。
   ffmpeg/ffprobe 的路径会依次尝试 `$UPSCALER_FFMPEG` → PATH → `/opt/homebrew/bin` → `/usr/local/bin`。
5. **单批帧数要跟模型倍率一起算**：MPSGraph 张量过大时不是报错，是**直接段错误**
   （`EXC_BAD_ACCESS` in `_mps_convolution`）。内部特征图 ≈ 帧数 × 64 × (tile × 倍率)²，
   所以 `tile=768` 下：2× 模型约 21 帧、**4× 模型只能 5 帧**。
   早期版本只按 tile 定死 24 帧，跑 4× 模型必崩。另有「末帧全黑」的检测兜底（宁可报错不吐坏帧）。

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
| 720p → 4K，轻量模型 | **0.18 s/帧** |
| 720p → 4K，`RealESRGAN_x4plus` | 5.9 s/帧（整帧推理，原分块是 8.5 s） |
| 720p → 1440p，`RealESRGAN_x2plus` | 1.5 s/帧 |
| 端到端：720p@24fps 2 秒 → 4K（48 帧，含模型加载与编码） | **16 s** |

**整帧推理**：只要内部特征图放得下（64 × 输出像素 ≤ 1.2e9），就不分块，直接整帧跑。
实测比 tile 分块快 19~30%，且少了接缝风险；放不下（1080p→8K 那种）才自动退回分块。

**编码器**：macOS 默认走 `h264_videotoolbox`（硬件编码），比 libx264 medium 快约 7%、体积小约 14%，
两者输出 PSNR 50.3 dB / SSIM 0.995（肉眼无差别）。要极限画质可 `UPSCALER_ENCODER=libx264`。

**异步流水线**（`--async` 或界面勾选）：解码预取下一段 + 编码写盘放后台线程。
6 秒素材实测 49s → 45s（-8%）；短片段收益更小，对内存要求略高，默认关闭。

实测（720p@24fps 6 秒 / 144 帧 → 4K）：同步 49s、异步 45s，约 0.31 秒/帧。

CPU 兜底大约慢 7~10 倍（小图 2 s vs 0.3 s），可用但别指望跑长视频。

## License

MIT © 2026 SugarCatCat
