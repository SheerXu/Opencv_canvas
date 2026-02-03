#!/bin/bash
# OpenCV算子预览工具 启动脚本（Linux/Mac）

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python!"
    echo "请先安装Python 3.7+"
    exit 1
fi

# 检查依赖库
echo "检查依赖库..."
python3 -c "import cv2; import PyQt5; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖库..."
    pip install -r requirements.txt
fi

# 启动应用
echo "启动OpenCV算子预览工具..."
python3 main.py
