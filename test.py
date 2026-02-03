"""
OpenCV算子预览工具 - 项目测试文件
此文件可用于快速测试各个模块的功能
"""

import sys
import numpy as np
from operators import OPERATORS


def test_operators():
    """测试所有算子"""
    print("测试所有算子...")
    
    # 创建简单的测试图像
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 255
    
    print(f"\n测试图像大小: {test_image.shape}")
    print(f"测试图像非零像素数: {np.sum(test_image > 0)}")
    
    # 测试所有分类
    for category, operators in OPERATORS.items():
        print(f"\n{category}:")
        for op_name, op_func in operators.items():
            try:
                if "Canny" in op_name:
                    result, stats = op_func(test_image, 50, 150)
                elif "Sobel" in op_name or "Laplacian" in op_name:
                    result, stats = op_func(test_image, ksize=3)
                else:
                    result, stats = op_func(test_image, kernel_size=3)
                
                print(f"  ✓ {op_name}: 成功")
            except Exception as e:
                print(f"  ✗ {op_name}: 失败 - {str(e)}")


def test_import():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from ui import MainWindow, DrawingCanvas, ResultDisplay
        print("  ✓ UI模块导入成功")
    except Exception as e:
        print(f"  ✗ UI模块导入失败: {str(e)}")
    
    try:
        import cv2
        print("  ✓ OpenCV导入成功")
    except Exception as e:
        print(f"  ✗ OpenCV导入失败: {str(e)}")
    
    try:
        import PyQt5
        print("  ✓ PyQt5导入成功")
    except Exception as e:
        print(f"  ✗ PyQt5导入失败: {str(e)}")


if __name__ == "__main__":
    print("=" * 50)
    print("OpenCV算子预览工具 - 测试")
    print("=" * 50)
    
    test_import()
    test_operators()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
