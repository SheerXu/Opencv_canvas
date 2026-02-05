"""
结果显示控件
用于显示处理后的图像
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from config import RULER_SPACING

class ResultDisplay(QWidget):
    """结果显示控件"""
    
    def __init__(self, width: int = 400, height: int = 300):
        super().__init__()
        self.width = width
        self.height = height
        
        # 标尺显示
        self.show_ruler = False
        self.ruler_spacing = RULER_SPACING  # 标尺间距（像素）
        
        # 当前显示的pixmap
        self.current_pixmap = None
        
        self.layout = QVBoxLayout(self)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setMinimumSize(width, height)
        self.image_label.setMaximumSize(width, height)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
    
    def set_ruler_visible(self, visible: bool):
        """设置标尺是否可见"""
        self.show_ruler = visible
        self._update_display()
    
    def set_ruler_spacing(self, spacing: int):
        """设置标尺间距"""
        self.ruler_spacing = max(10, spacing)
        self._update_display()
    
    def _draw_ruler_on_pixmap(self, pixmap):
        """在pixmap上绘制标尺"""
        if pixmap is None or pixmap.isNull():
            return pixmap
        
        result = QPixmap(pixmap)
        painter = QPainter(result)
        
        # 设置标尺样式
        ruler_color = QColor(100, 149, 237, 180)  # 浅蓝色半透明
        text_color = QColor(70, 130, 180)
        painter.setPen(QPen(ruler_color, 1))
        
        spacing = self.ruler_spacing
        w = result.width()
        h = result.height()
        
        # 竖线
        for x in range(0, w + 1, spacing):
            if x == 0:
                continue
            painter.drawLine(x, 0, x, h)
            painter.setPen(text_color)
            painter.drawText(x + 2, 12, str(x))
            painter.setPen(QPen(ruler_color, 1))
        
        # 横线
        for y in range(0, h + 1, spacing):
            if y == 0:
                continue
            painter.drawLine(0, y, w, y)
            painter.setPen(text_color)
            painter.drawText(2, y - 2, str(y))
            painter.setPen(QPen(ruler_color, 1))
        
        painter.end()
        return result
    
    def _update_display(self):
        """更新显示（添加或移除标尺）"""
        if self.current_pixmap is None:
            return
        
        if self.show_ruler:
            display_pixmap = self._draw_ruler_on_pixmap(self.current_pixmap)
        else:
            display_pixmap = self.current_pixmap
        
        self.image_label.setPixmap(display_pixmap)
    
    def set_image(self, image_array: np.ndarray):
        """设置显示的图像"""
        if image_array is None:
            return
        
        # 确保数据是连续的
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        height = image_array.shape[0]
        width = image_array.shape[1]
        
        # 处理不同格式的图像
        if len(image_array.shape) == 2:
            # 灰度图像 - 需要反转以显示为白底黑字
            # image_array = 255 - image_array
            bytes_per_line = width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 3:
                # BGR 彩色图像 - 转换为 RGB 以显示正确的颜色（绿色框）
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                rgb_image = np.ascontiguousarray(rgb_image)
                q_image = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
            else:
                # BGRA 或其他格式
                q_image = QImage(image_array.data, width, height, image_array.shape[2] * width, QImage.Format_RGBA8888)
        else:
            return
        
        # 缩放到标签大小
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.current_pixmap = scaled_pixmap
        self._update_display()
    
    def clear(self):
        """清空显示"""
        self.current_pixmap = None
        self.image_label.clear()
