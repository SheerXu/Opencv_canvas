"""
绘画画布控件
提供二值化图像的绘画功能
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import numpy as np
import cv2
from config import *


class DrawingCanvas(QWidget):
    """绘画画布控件"""
    
    image_changed = pyqtSignal()  # 图像变化信号
    
    def __init__(self, width: int = 400, height: int = 300):
        super().__init__()
        self.width = width
        self.height = height
        self.drawing = False
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.brush_color = QColor(*DEFAULT_BRUSH_COLOR)  # 笔刷颜色
        
        # 初始化画布
        self.image = QImage(width, height, QImage.Format_Grayscale8)
        self.image.fill(QColor(*CANVAS_BG_COLOR))  # 背景颜色
        
        self.last_point = QPoint()
        
        self.setMouseTracking(False)
        self.setMinimumSize(width, height)
        self.setCursor(Qt.CrossCursor)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, 
                               Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            
            self.last_point = event.pos()
            self.update()
            self.image_changed.emit()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
    
    def clear_canvas(self):
        """清空画布"""
        self.image.fill(QColor(255, 255, 255))
        self.update()
        self.image_changed.emit()
    
    def set_brush_size(self, size: int):
        """设置笔刷大小"""
        self.brush_size = max(1, size)
    
    def set_brush_color(self, color: QColor):
        """设置笔刷颜色"""
        self.brush_color = color
    
    def get_image_array(self) -> np.ndarray:
        """获取画布内容为NumPy数组"""
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(self.image.byteCount())
        arr = np.array(ptr).reshape(height, self.image.bytesPerLine())
        # 移除行填充，只保留实际的宽度数据
        arr = arr[:, :width]
        return arr
    
    def set_image_array(self, arr: np.ndarray):
        """从NumPy数组设置画布内容"""
        if len(arr.shape) == 2:
            height, width = arr.shape
            arr = np.ascontiguousarray(arr)
            bytes_per_line = width
            q_image = QImage(arr.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.image = q_image.copy()
            self.update()
            self.image_changed.emit()
    
    def undo(self):
        """撤销操作 - 简单实现，清空画布"""
        self.clear_canvas()
