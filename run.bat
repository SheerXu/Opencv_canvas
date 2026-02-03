@echo off
REM OpenCV算子预览工具 启动脚本

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python!
    echo 请先安装Python 3.7+
    pause
    exit /b 1
)

REM 检查依赖库
echo 检查依赖库...
python -c "import cv2; import PyQt5; import numpy" >nul 2>&1
if errorlevel 1 (
    echo 安装依赖库...
    pip install -r requirements.txt
)

REM 启动应用
echo 启动OpenCV算子预览工具...
python main.py

pause
