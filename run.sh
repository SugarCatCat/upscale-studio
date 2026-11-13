#!/usr/bin/env bash
# upscale-studio 启动脚本：准备环境 → 检查/下载模型 → 起 Web 界面
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8199}"

# ---- 找一个已经装好依赖的 python ----
# 顺序：项目内 .venv → $UPSCALER_PYTHON → 系统 python3 → 已有的 conda 环境（复用其 torch，省 2GB 下载）
usable() { "$1" -c "import torch, spandrel, aiohttp, PIL, numpy" >/dev/null 2>&1; }
PY=""
CANDIDATES=(
  "./.venv/bin/python"
  "${UPSCALER_PYTHON:-}"
  "$(command -v python3 || true)"
  "$HOME/miniconda3/envs/comfyui/bin/python"
  "$HOME/anaconda3/envs/comfyui/bin/python"
)
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -x "$c" ] && usable "$c" && { PY="$c"; break; }
done

if [ -z "$PY" ]; then
  echo "没有找到带 torch/spandrel 的 Python。"
  echo "正在创建 .venv 并安装依赖（torch 约 2GB，首次会慢）…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -e .
  PY="./.venv/bin/python"
fi

echo "使用 $PY"
# -u 让日志实时输出；MPS 水位限制避免把整机拖进 swap
exec env -u PYTHONPATH \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.7}" \
  PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5}" \
  "$PY" -u -m upscaler.webui.server --port "$PORT" "$@"
