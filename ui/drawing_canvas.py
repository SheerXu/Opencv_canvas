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
        
        # 点绘制模式（用于聚类等场景，点击生成固定大小的点）
        self.point_mode = False
        self.point_radius = 5  # 点的半径
        
        # 聚类模式（白底黑字）
        self.cluster_mode = False
        
        # 标尺显示
        self.show_ruler = False
        self.ruler_spacing = 50  # 标尺间距（像素）
        
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
            if self.point_mode:
                # 点模式：直接在点击位置绘制一个固定大小的圆点
                self.draw_point(event.pos())
            else:
                self.drawing = True
                self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.point_mode:
            return  # 点模式不处理拖拽
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
    
    def draw_point(self, pos):
        """在指定位置绘制一个点"""
        painter = QPainter(self.image)
        # 聚类模式使用黑色点，其他模式使用笔刷颜色
        if self.cluster_mode:
            point_color = QColor(0, 0, 0)  # 黑色点
        else:
            point_color = self.brush_color
        painter.setPen(QPen(point_color, 1))
        painter.setBrush(QBrush(point_color))
        painter.drawEllipse(pos, self.point_radius, self.point_radius)
        painter.end()
        self.update()
        self.image_changed.emit()
    
    def set_point_mode(self, enabled: bool):
        """设置点绘制模式"""
        self.point_mode = enabled
        if enabled:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)
    
    def set_point_radius(self, radius: int):
        """设置点的半径"""
        self.point_radius = max(1, radius)
    
    def set_cluster_mode(self, enabled: bool):
        """设置聚类模式（白底黑字）"""
        self.cluster_mode = enabled
        self.show_ruler = enabled  # 聚类模式自动显示标尺
        if enabled:
            # 切换到白底
            self.image.fill(QColor(255, 255, 255))
        else:
            # 切换回黑底
            self.image.fill(QColor(*CANVAS_BG_COLOR))
        self.update()
    
    def set_ruler_visible(self, visible: bool):
        """设置标尺是否可见"""
        self.show_ruler = visible
        self.update()
    
    def set_ruler_spacing(self, spacing: int):
        """设置标尺间距"""
        self.ruler_spacing = max(10, spacing)
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        
        # 绘制标尺
        if self.show_ruler:
            self._draw_ruler(painter)
    
    def _draw_ruler(self, painter):
        """绘制标尺"""
        # 设置标尺样式
        ruler_color = QColor(100, 149, 237, 180)  # 浅蓝色半透明
        text_color = QColor(70, 130, 180)
        painter.setPen(QPen(ruler_color, 1))
        
        # 绘制竖线和横线网格
        spacing = self.ruler_spacing
        
        # 竖线
        for x in range(0, self.width + 1, spacing):
            if x == 0:
                continue
            painter.drawLine(x, 0, x, self.height)
            # 绘制刻度数字
            painter.setPen(text_color)
            painter.drawText(x + 2, 12, str(x))
            painter.setPen(QPen(ruler_color, 1))
        
        # 横线
        for y in range(0, self.height + 1, spacing):
            if y == 0:
                continue
            painter.drawLine(0, y, self.width, y)
            # 绘制刻度数字
            painter.setPen(text_color)
            painter.drawText(2, y - 2, str(y))
            painter.setPen(QPen(ruler_color, 1))
    
    def clear_canvas(self):
        """清空画布"""
        if self.cluster_mode:
            self.image.fill(QColor(255, 255, 255))  # 聚类模式用白底
        else:
            self.image.fill(QColor(*CANVAS_BG_COLOR))
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
