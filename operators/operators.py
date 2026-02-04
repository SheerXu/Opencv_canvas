"""
OpenCV算子集合
包含各种形态学操作、轮廓检测等算子
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any


class MorphologyOperator:
    """形态学操作类"""
    
    @staticmethod
    def erode(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """腐蚀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.erode(image, kernel, iterations=1)
        
        stats = {
            "操作": "腐蚀",
            "核大小": f"{kernel_size}x{kernel_size}",
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def dilate(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """膨胀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.dilate(image, kernel, iterations=1)
        
        stats = {
            "操作": "膨胀",
            "核大小": f"{kernel_size}x{kernel_size}",
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def open(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """开运算（先腐蚀后膨胀）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        stats = {
            "操作": "开运算",
            "核大小": f"{kernel_size}x{kernel_size}",
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def close(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """闭运算（先膨胀后腐蚀）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        stats = {
            "操作": "闭运算",
            "核大小": f"{kernel_size}x{kernel_size}",
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def gradient(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """形态学梯度（膨胀-腐蚀）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        stats = {
            "操作": "形态学梯度",
            "核大小": f"{kernel_size}x{kernel_size}",
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats


class EdgeDetectionOperator:
    """边缘检测操作类"""
    
    @staticmethod
    def canny(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> Tuple[np.ndarray, Dict]:
        """Canny边缘检测"""
        result = cv2.Canny(image, threshold1, threshold2)
        
        stats = {
            "操作": "Canny边缘检测",
            "低阈值": threshold1,
            "高阈值": threshold2,
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def sobel_x(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, Dict]:
        """Sobel X方向边缘检测"""
        result = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
        result = np.uint8(np.absolute(result))
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        stats = {
            "操作": "Sobel X",
            "核大小": ksize,
            "平均灰度值": np.mean(result),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def sobel_y(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, Dict]:
        """Sobel Y方向边缘检测"""
        result = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
        result = np.uint8(np.absolute(result))
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        stats = {
            "操作": "Sobel Y",
            "核大小": ksize,
            "平均灰度值": np.mean(result),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 1) -> Tuple[np.ndarray, Dict]:
        """Laplacian边缘检测"""
        result = cv2.Laplacian(image, cv2.CV_32F, ksize=ksize)
        result = np.uint8(np.absolute(result))
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        stats = {
            "操作": "Laplacian",
            "核大小": ksize,
            "平均灰度值": np.mean(result),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats


class ContourOperator:
    """轮廓操作类"""
    
    @staticmethod
    def find_contours(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """轮廓检测"""
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(image)
        cv2.drawContours(result, contours, -1, 255, 1)
        
        stats = {
            "操作": "轮廓检测",
            "轮廓数量": len(contours),
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats
    
    @staticmethod
    def convex_hull(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """凸包检测"""
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = np.zeros_like(image)
        for cnt in contours:
            if len(cnt) > 2:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(result, [hull], 0, 255, 1)
        
        stats = {
            "操作": "凸包",
            "轮廓数量": len(contours),
            "白色像素数": np.sum(result > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats


class SkeletonOperator:
    """骨架提取操作类"""
    
    @staticmethod
    def skeleton(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """骨架提取"""
        result = image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skeleton = np.zeros_like(image)
        
        while True:
            eroded = cv2.erode(result, kernel)
            dilated = cv2.dilate(eroded, kernel)
            diff = cv2.subtract(result, dilated)
            skeleton = cv2.bitwise_or(skeleton, diff)
            result = eroded
            
            if cv2.countNonZero(result) == 0:
                break
        
        stats = {
            "操作": "骨架提取",
            "白色像素数": np.sum(skeleton > 0),
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return skeleton, stats


class DistanceOperator:
    """距离变换操作类"""
    
    @staticmethod
    def distance_transform(image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """欧氏距离变换"""
        dist = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(dist_norm)
        
        stats = {
            "操作": "距离变换",
            "最大距离": np.max(dist),
            "平均距离": np.mean(dist[dist > 0]) if np.any(dist > 0) else 0,
            "图像大小": f"{image.shape[0]}x{image.shape[1]}"
        }
        return result, stats


class TemplateMatchingOperator:
    """模板匹配操作类"""
    
    @staticmethod
    def template_match(source_image: np.ndarray, template_image: np.ndarray = None, show_heatmap: bool = False) -> Tuple[np.ndarray, Dict]:
        """模板匹配"""
        if template_image is None or template_image.size == 0:
            raise ValueError("模板图像为空，请先指定模板区域")
        
        if source_image.shape[0] < template_image.shape[0] or source_image.shape[1] < template_image.shape[1]:
            raise ValueError("模板图像大于源图像，无法进行匹配")
        
        # 使用归一化相关系数匹配
        match_result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)
        
        # 找到最优匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        top_left = max_loc
        score = max_val  # 使用 max_val 作为置信度分数
        bottom_right = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])
        
        if show_heatmap:
            # 归一化到 0-255 并转为 uint8
            heatmap = cv2.normalize(match_result, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = np.uint8(heatmap)
            # 应用伪彩色 colormap
            result_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # 调整热力图大小以便于观看（如果太小）
            if result_image.shape[0] < source_image.shape[0] or result_image.shape[1] < source_image.shape[1]:
                 result_image = cv2.resize(result_image, (source_image.shape[1], source_image.shape[0]))
        else:
            # 在源图像上绘制匹配框（转为 BGR 彩色图以保留绿色）
            result_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # 返回 BGR 彩色图
        stats = {
            "操作": "模板匹配",
            "模式": "热力图" if show_heatmap else "框选",
            "置信度 (Score)": f"{max_val:.4f}",
            "匹配位置": f"({top_left[0]}, {top_left[1]})",
            "模板大小": f"{template_image.shape[1]}x{template_image.shape[0]}",
            "源图像大小": f"{source_image.shape[1]}x{source_image.shape[0]}"
        }
        return result_image, stats


# 算子注册表
OPERATORS = {
    "形态学操作": {
        "腐蚀": MorphologyOperator.erode,
        "膨胀": MorphologyOperator.dilate,
        "开运算": MorphologyOperator.open,
        "闭运算": MorphologyOperator.close,
        "形态学梯度": MorphologyOperator.gradient,
    },
    "边缘检测": {
        "Canny": EdgeDetectionOperator.canny,
        "Sobel X": EdgeDetectionOperator.sobel_x,
        "Sobel Y": EdgeDetectionOperator.sobel_y,
        "Laplacian": EdgeDetectionOperator.laplacian,
    },
    "轮廓操作": {
        "轮廓检测": ContourOperator.find_contours,
        "凸包": ContourOperator.convex_hull,
    },
    "骨架提取": {
        "骨架提取": SkeletonOperator.skeleton,
    },
    "距离变换": {
        "距离变换": DistanceOperator.distance_transform,
    },
    "模板匹配": {
        "模板匹配": TemplateMatchingOperator.template_match,
    }
}
