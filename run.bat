@echo off
rem upscale-studio 启动脚本（Windows）
setlocal
cd /d "%~dp0"
if "%PORT%"=="" set PORT=8199

set PY=
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 找不到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
  exit /b 1
)

python -c "import torch, spandrel, aiohttp, PIL, numpy" >nul 2>&1
if errorlevel 1 (
  echo 正在创建 .venv 并安装依赖（首次约 2GB，需要几分钟）...
  python -m venv .venv || exit /b 1
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q -e .
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)

echo 使用 %PY%
"%PY%" -u -m upscaler.webui.server --port %PORT% %*
