@echo off
setlocal
call conda activate zi2zi-jit
if errorlevel 1 (
  echo Failed to activate the zi2zi-jit Conda environment.
  exit /b 1
)
python webui.py %*
