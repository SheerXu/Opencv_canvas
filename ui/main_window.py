"""
主窗口
整合所有UI组件和处理逻辑
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QLabel, QComboBox, QSpinBox, 
                             QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import cv2

from .drawing_canvas import DrawingCanvas
from .result_display import ResultDisplay
from operators import OPERATORS


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV算子预览工具")
        self.setGeometry(100, 100, 1400, 750)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局（竖直）
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部：标签栏
        top_layout = QHBoxLayout()
        
        left_label = QLabel("绘画区（黑色笔刷绘画在白色背景上）")
        left_label.setFont(QFont("Consolas", 11, QFont.Bold))
        
        middle_label = QLabel("参数设置与操作")
        middle_label.setFont(QFont("Consolas", 11, QFont.Bold))
        middle_label.setAlignment(Qt.AlignCenter)
        
        right_label = QLabel("处理结果")
        right_label.setFont(QFont("Consolas", 11, QFont.Bold))
        right_label.setAlignment(Qt.AlignRight)
        
        top_layout.addWidget(left_label, 1)
        top_layout.addWidget(middle_label, 0)
        top_layout.addWidget(right_label, 1)
        
        main_layout.addLayout(top_layout)
        
        # 中间：内容区域（水平三栏）
        content_layout = QHBoxLayout()
        
        # ===== 左侧：绘画区域 =====
        left_content_layout = QVBoxLayout()
        self.canvas = DrawingCanvas(450, 450)
        left_content_layout.addWidget(self.canvas)
        left_content_layout.addStretch()
        
        # ===== 中间：参数选择和操作区域 =====
        middle_content_layout = QVBoxLayout()
        
        # 笔刷设置区域
        brush_group = QGroupBox("笔刷设置")
        brush_layout = QFormLayout(brush_group)
        
        brush_size_label = QLabel("笔刷大小:")
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setMinimum(1)
        self.brush_size_spinbox.setMaximum(50)
        self.brush_size_spinbox.setValue(5)
        self.brush_size_spinbox.valueChanged.connect(
            lambda v: self.canvas.set_brush_size(v)
        )
        brush_layout.addRow(brush_size_label, self.brush_size_spinbox)
        
        clear_btn = QPushButton("清空画布")
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        brush_layout.addRow(clear_btn)
        
        middle_content_layout.addWidget(brush_group)
        
        # 算子分类选择
        category_label = QLabel("选择算子分类")
        category_label.setFont(QFont("Consolas", 10, QFont.Bold))
        middle_content_layout.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(OPERATORS.keys())
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        middle_content_layout.addWidget(self.category_combo)
        
        # 具体算子选择
        operator_label = QLabel("选择具体算子")
        operator_label.setFont(QFont("Consolas", 10, QFont.Bold))
        middle_content_layout.addWidget(operator_label)
        
        self.operator_combo = QComboBox()
        self.update_operator_combo()
        middle_content_layout.addWidget(self.operator_combo)
        
        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout(params_group)
        
        # 核大小参数
        kernel_label = QLabel("核大小")
        self.kernel_spinbox = QSpinBox()
        self.kernel_spinbox.setMinimum(1)
        self.kernel_spinbox.setMaximum(21)
        self.kernel_spinbox.setValue(5)
        self.kernel_spinbox.setSingleStep(2)  # 步长为2，保证为奇数
        params_layout.addRow(kernel_label, self.kernel_spinbox)
        
        # Canny阈值
        threshold1_label = QLabel("低阈值")
        self.threshold1_spinbox = QSpinBox()
        self.threshold1_spinbox.setMinimum(0)
        self.threshold1_spinbox.setMaximum(500)
        self.threshold1_spinbox.setValue(100)
        params_layout.addRow(threshold1_label, self.threshold1_spinbox)
        
        threshold2_label = QLabel("高阈值")
        self.threshold2_spinbox = QSpinBox()
        self.threshold2_spinbox.setMinimum(0)
        self.threshold2_spinbox.setMaximum(500)
        self.threshold2_spinbox.setValue(200)
        params_layout.addRow(threshold2_label, self.threshold2_spinbox)
        
        middle_content_layout.addWidget(params_group)
        
        # 运行按钮
        run_btn = QPushButton("运行算子")
        run_btn.setFont(QFont("Consolas", 12, QFont.Bold))
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        run_btn.clicked.connect(self.run_operator)
        middle_content_layout.addWidget(run_btn)
        
        middle_content_layout.addStretch()
        
        # ===== 右侧：结果显示区域 =====
        right_content_layout = QVBoxLayout()
        self.result_display = ResultDisplay(450, 450)
        right_content_layout.addWidget(self.result_display)
        right_content_layout.addStretch()
        
        # 组合左中右三栏
        content_layout.addLayout(left_content_layout, 1)
        content_layout.addLayout(middle_content_layout, 0)
        content_layout.addLayout(right_content_layout, 1)
        
        main_layout.addLayout(content_layout)
        central_widget.setLayout(main_layout)
    
    def on_category_changed(self, category):
        """当分类改变时更新算子列表"""
        self.update_operator_combo()
    
    def update_operator_combo(self):
        """更新算子下拉框"""
        current_category = self.category_combo.currentText()
        if current_category in OPERATORS:
            operators = list(OPERATORS[current_category].keys())
            self.operator_combo.clear()
            self.operator_combo.addItems(operators)
    
    def run_operator(self):
        """运行选定的算子"""
        try:
            # 获取原始图像
            input_image = self.canvas.get_image_array()
            
            # 反转图像：将白色(255)的背景转为黑色(0)，黑色(0)的笔刷转为白色(255)
            input_image = 255 - input_image
            
            # 检查是否只有背景（全黑）
            if np.sum(input_image) == 0:
                QMessageBox.warning(self, "警告", "请先在画布上绘画")
                return
            
            # 获取选定的算子
            category = self.category_combo.currentText()
            operator_name = self.operator_combo.currentText()
            operator_func = OPERATORS[category][operator_name]
            
            # 准备参数
            kernel_size = self.kernel_spinbox.value()
            # 确保核大小为奇数
            if kernel_size % 2 == 0:
                kernel_size -= 1
            
            threshold1 = self.threshold1_spinbox.value()
            threshold2 = self.threshold2_spinbox.value()
            
            # 调用算子函数
            if "Canny" in operator_name or "canny" in operator_name:
                result_image, stats = operator_func(input_image, threshold1, threshold2)
            elif "Sobel" in operator_name or "Laplacian" in operator_name:
                result_image, stats = operator_func(input_image, kernel_size)
            else:
                result_image, stats = operator_func(input_image, kernel_size)
            
            # 显示结果
            self.result_display.set_image(result_image)
            self.result_display.set_stats(stats)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出错:\n{str(e)}")
