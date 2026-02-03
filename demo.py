"""
示例脚本 - 演示如何使用各个算子
无需GUI，直接调用算子进行图像处理
"""

import cv2
import numpy as np
from operators import OPERATORS


def create_sample_image():
    """创建示例二值化图像"""
    image = np.zeros((300, 300), dtype=np.uint8)
    
    # 绘制一个矩形
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    
    # 绘制一个圆
    cv2.circle(image, (200, 100), 40, 255, -1)
    
    # 添加一些噪点
    noise_points = np.random.rand(300, 300) > 0.95
    image[noise_points] = 255
    
    return image


def demo_morphology():
    """演示形态学操作"""
    print("\n=== 形态学操作演示 ===")
    image = create_sample_image()
    
    morphology_ops = OPERATORS["形态学操作"]
    
    for op_name, op_func in morphology_ops.items():
        result, stats = op_func(image, kernel_size=5)
        print(f"\n{op_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_edge_detection():
    """演示边缘检测"""
    print("\n=== 边缘检测演示 ===")
    image = create_sample_image()
    
    edge_ops = OPERATORS["边缘检测"]
    
    for op_name, op_func in edge_ops.items():
        if "Canny" in op_name:
            result, stats = op_func(image, 100, 200)
        else:
            result, stats = op_func(image, ksize=3)
        print(f"\n{op_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_contour():
    """演示轮廓操作"""
    print("\n=== 轮廓操作演示 ===")
    image = create_sample_image()
    
    contour_ops = OPERATORS["轮廓操作"]
    
    for op_name, op_func in contour_ops.items():
        result, stats = op_func(image)
        print(f"\n{op_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_skeleton():
    """演示骨架提取"""
    print("\n=== 骨架提取演示 ===")
    image = create_sample_image()
    
    skeleton_ops = OPERATORS["骨架提取"]
    
    for op_name, op_func in skeleton_ops.items():
        result, stats = op_func(image)
        print(f"\n{op_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def demo_distance_transform():
    """演示距离变换"""
    print("\n=== 距离变换演示 ===")
    image = create_sample_image()
    
    dist_ops = OPERATORS["距离变换"]
    
    for op_name, op_func in dist_ops.items():
        result, stats = op_func(image)
        print(f"\n{op_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def main():
    """主程序"""
    print("OpenCV算子预览工具 - 命令行演示")
    print("=" * 50)
    
    demo_morphology()
    demo_edge_detection()
    demo_contour()
    demo_skeleton()
    demo_distance_transform()
    
    print("\n" + "=" * 50)
    print("演示完成！")


if __name__ == "__main__":
    main()
