"""
ROI（感兴趣区域）选择画布
用于图像导入和矩形框选择
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
import numpy as np
import cv2
from PIL import Image


class ROICanvas(QWidget):
    """ROI 选择画布"""
    
    roi_changed = pyqtSignal()  # ROI 变化信号
    
    def __init__(self, width: int = 400, height: int = 300):
        super().__init__()
        self.width = width
        self.height = height
        
        # 原始图像
        self.original_image = None
        self.display_pixmap = None
        
        # ROI 矩形框
        self.roi_rect = None
        self.start_point = None
        self.is_drawing = False
        self.drawing_enabled = False  # 是否允许绘制（需要点击"指定区域"按钮才能启用）
        
        # 显示信息（用于坐标转换）
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.display_scale = 1.0
        
        self.setMouseTracking(True)
        self.setMinimumSize(width, height)
        self.setCursor(Qt.CrossCursor)
    
    def load_image(self, image_path: str):
        """加载图像"""
        try:
            # 使用 PIL 读取图像以支持中文路径
            pil_image = Image.open(image_path).convert('L')  # 转换为灰度图
            image = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"无法加载图像: {str(e)}")
        
        # 缩放图像到指定大小
        if image.shape[0] > self.height or image.shape[1] > self.width:
            scale = min(self.width / image.shape[1], self.height / image.shape[0])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
        
        self.original_image = image
        self.roi_rect = None  # 重置 ROI
        self.display_pixmap = None
        self.update()
        self.roi_changed.emit()
    
    def load_image_array(self, image_array: np.ndarray):
        """从 NumPy 数组加载图像"""
        if len(image_array.shape) != 2:
            raise ValueError("仅支持灰度图像")
        
        # 缩放图像到指定大小
        if image_array.shape[0] > self.height or image_array.shape[1] > self.width:
            scale = min(self.width / image_array.shape[1], self.height / image_array.shape[0])
            new_size = (int(image_array.shape[1] * scale), int(image_array.shape[0] * scale))
            image_array = cv2.resize(image_array, new_size)
        
        self.original_image = image_array
        self.roi_rect = None
        self.display_pixmap = None
        self.update()
        self.roi_changed.emit()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.original_image is None or not self.drawing_enabled:
            return
        
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.start_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.original_image is None or not self.is_drawing or not self.drawing_enabled:
            return
        
        # 更新 ROI 矩形
        current_point = event.pos()
        self.roi_rect = QRect(self.start_point, current_point).normalized()
        self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.original_image is None or not self.drawing_enabled:
            return
        
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.drawing_enabled = False  # 绘制完成后自动禁用绘制模式
            if self.roi_rect and self.roi_rect.width() > 0 and self.roi_rect.height() > 0:
                self.roi_changed.emit()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        
        if self.original_image is None:
            painter.fillRect(self.rect(), QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "点击 '导入图片' 加载图像")
            return
        
        # 转换图像为 QImage 并显示
        if len(self.original_image.shape) == 2:
            h, w = self.original_image.shape
            bytes_per_line = w
            q_image = QImage(self.original_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 计算实际显示位置（考虑 KeepAspectRatio 导致的偏移）
        x_offset = (self.width - scaled_pixmap.width()) // 2
        y_offset = (self.height - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # 保存显示信息供 get_roi_image 使用
        self.display_offset_x = x_offset
        self.display_offset_y = y_offset
        self.display_scale = scaled_pixmap.width() / self.original_image.shape[1]
        
        # 绘制 ROI 矩形框
        if self.roi_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(self.roi_rect)
    
    def get_roi_image(self) -> np.ndarray:
        """获取 ROI 区域的图像"""
        if self.original_image is None or self.roi_rect is None or self.roi_rect.width() <= 0 or self.roi_rect.height() <= 0:
            return None
        
        # 获取原始图像尺寸
        if len(self.original_image.shape) == 2:
            h, w = self.original_image.shape
        else:
            h, w = self.original_image.shape[:2]
        
        # 使用保存的显示信息进行坐标转换
        # 如果还没有绘制过，先计算显示参数
        if not hasattr(self, 'display_scale'):
            # 计算缩放因子
            scale_x = min(self.width / w, self.height / h)
            self.display_scale = scale_x
            self.display_offset_x = (self.width - int(w * scale_x)) // 2
            self.display_offset_y = (self.height - int(h * scale_x)) // 2
        
        # 将屏幕坐标转换回原始图像坐标
        # 先移除显示偏移，再除以缩放因子
        x1 = max(0, int((self.roi_rect.left() - self.display_offset_x) / self.display_scale))
        y1 = max(0, int((self.roi_rect.top() - self.display_offset_y) / self.display_scale))
        x2 = min(w, int((self.roi_rect.right() - self.display_offset_x) / self.display_scale))
        y2 = min(h, int((self.roi_rect.bottom() - self.display_offset_y) / self.display_scale))
        
        # 确保坐标有效
        if x1 >= x2 or y1 >= y2:
            return None
        
        roi_image = self.original_image[y1:y2, x1:x2]
        return roi_image
    
    def get_image_array(self) -> np.ndarray:
        """获取整个图像"""
        return self.original_image
    
    def enable_drawing(self):
        """启用绘制模式"""
        self.drawing_enabled = True
        self.roi_rect = None  # 清除之前的 ROI
        self.update()
    
    def clear_roi(self):
        """清除 ROI 选择"""
        self.roi_rect = None
        self.drawing_enabled = False
        self.update()
        self.roi_changed.emit()
    
    def clear_image(self):
        """清除图像"""
        self.original_image = None
        self.roi_rect = None
        self.display_pixmap = None
        self.drawing_enabled = False
        self.update()
        self.roi_changed.emit()
