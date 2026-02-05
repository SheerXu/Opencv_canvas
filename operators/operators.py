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


class ClusterOperator:
    """聚类算法类"""

    @staticmethod
    def _extract_points(image: np.ndarray) -> np.ndarray:
        """从二值图像中提取黑色点的中心坐标 (x, y)
        使用距离变换 + 局部极大值检测来分离重叠的点
        """
        # 假设背景为白(255)，前景为黑(0)
        # 如果是灰度图，先二值化
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 使用距离变换找到点的中心
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 归一化距离变换结果
        if dist_transform.max() > 0:
            dist_normalized = dist_transform / dist_transform.max()
        else:
            dist_normalized = dist_transform
        
        # 找到局部极大值作为点的中心
        # 使用膨胀操作找局部极大值
        kernel_size = 7  # 调整此值可改变检测灵敏度
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(dist_transform, kernel)
        
        # 局部极大值：距离变换值等于膨胀后的值，且距离大于阈值
        threshold = 0.3 * dist_transform.max() if dist_transform.max() > 0 else 0
        local_max = (dist_transform == dilated) & (dist_transform > threshold)
        
        # 获取局部极大值的坐标
        ys, xs = np.where(local_max)
        points = np.column_stack((xs, ys))
        
        # 如果没有找到点，退回到轮廓检测
        if len(points) == 0:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    points.append([cX, cY])
            points = np.array(points) if points else np.array([])
        
        # 如果仍然没有点，检查是否有任何前景像素
        if len(points) == 0:
            ys, xs = np.where(binary > 0)
            if len(xs) > 0:
                points = np.column_stack((xs, ys))
            
        return np.array(points, dtype=np.float32) if len(points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)

    @staticmethod
    def _draw_cluster_result(image: np.ndarray, points: np.ndarray, labels: np.ndarray, k_or_n_clusters: int) -> np.ndarray:
        """绘制聚类结果"""
        h, w = image.shape[:2]
        result = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白底
        
        # 生成颜色表
        if k_or_n_clusters > 0:
            # 固定随机种子以保持颜色一致
            np.random.seed(42)
            colors = np.random.randint(0, 255, (k_or_n_clusters, 3)).tolist()
        else:
            colors = []

        # 噪点颜色 (黑色或灰色)
        noise_color = (128, 128, 128)

        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])
            if label == -1:
                color = noise_color
            else:
                # 确保 label 在颜色范围内
                color_idx = label % len(colors)
                color = colors[color_idx]
                # BGR
                color = (int(color[0]), int(color[1]), int(color[2]))

            cv2.circle(result, (x, y), 6, color, -1)
            cv2.circle(result, (x, y), 7, (0, 0, 0), 1) # 描边

        return result

    @staticmethod
    def kmeans(image: np.ndarray, k: int = 3) -> Tuple[np.ndarray, Dict]:
        """KMeans 聚类"""
        points = ClusterOperator._extract_points(image)
        
        if len(points) < k:
             stats = {"状态": "错误", "信息": f"点数量 ({len(points)}) 少于簇数量 ({k})"}
             return image, stats

        # OpenCV kmeans 要求 float32
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        result_image = ClusterOperator._draw_cluster_result(image, points, labels.flatten(), k)
        
        stats = {
            "操作": "KMeans",
            "点数量": len(points),
            "簇数量(K)": k,
            "中心点": str([list(map(int, c)) for c in centers])
        }
        return result_image, stats

    @staticmethod
    def _dbscan_impl(X, eps, min_samples):
        """DBSCAN numpy 实现"""
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)  # -1 表示未分类/噪点
        cluster_id = 0
        
        # 计算距离矩阵 (N, N)
        dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
        
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            # 找到邻域内的所有点（包括自己）
            neighbors = np.where(dists[i] <= eps)[0]
            
            # 如果邻居数量不足，标记为噪点
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                # 是核心点，开始新簇
                labels[i] = cluster_id
                
                # 使用队列扩展簇
                seed_set = list(neighbors)
                
                j = 0
                while j < len(seed_set):
                    q = seed_set[j]
                    j += 1
                    
                    # 如果 q 之前被标记为噪点，现在改为当前簇的边界点
                    if labels[q] == -1:
                        labels[q] = cluster_id
                    
                    # 如果 q 已经被访问过，跳过
                    if visited[q]:
                        continue
                    
                    visited[q] = True
                    labels[q] = cluster_id
                    
                    # 检查 q 的邻域
                    q_neighbors = np.where(dists[q] <= eps)[0]
                    
                    # 如果 q 也是核心点，将其邻居加入seed_set
                    if len(q_neighbors) >= min_samples:
                        for neighbor in q_neighbors:
                            if neighbor not in seed_set:
                                seed_set.append(neighbor)
                
                cluster_id += 1
        
        return labels, cluster_id

    @staticmethod
    def dbscan(image: np.ndarray, eps: float = 30.0, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """DBSCAN 聚类"""
        points = ClusterOperator._extract_points(image)
        
        if len(points) == 0:
             stats = {"状态": "错误", "信息": "没有检测到点"}
             return image, stats

        labels, n_clusters = ClusterOperator._dbscan_impl(points, eps, min_samples)
        
        result_image = ClusterOperator._draw_cluster_result(image, points, labels, n_clusters)
        
        # 统计噪点
        n_noise = list(labels).count(-1)
        
        stats = {
            "操作": "DBSCAN",
            "点数量": len(points),
            "Epsilon": eps,
            "Min Samples": min_samples,
            "发现簇数量": n_clusters,
            "噪点数量": n_noise
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
    },
    "聚类算法": {
        "KMeans": ClusterOperator.kmeans,
        "DBSCAN": ClusterOperator.dbscan,
    }
}
