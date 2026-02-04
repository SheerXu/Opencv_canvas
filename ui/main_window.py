"""
主窗口
整合所有UI组件和处理逻辑
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QLabel, QComboBox, QSpinBox, 
                             QGroupBox, QFormLayout, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import cv2

from .drawing_canvas import DrawingCanvas
from .result_display import ResultDisplay
from .roi_canvas import ROICanvas
from operators import OPERATORS
from config import *


# 算子参数映射表
OPERATOR_PARAMS = {
    "形态学操作": {
        "腐蚀": ["kernel_size"],
        "膨胀": ["kernel_size"],
        "开运算": ["kernel_size"],
        "闭运算": ["kernel_size"],
        "形态学梯度": ["kernel_size"],
    },
    "边缘检测": {
        "Canny": ["threshold1", "threshold2"],
        "Sobel X": ["kernel_size"],
        "Sobel Y": ["kernel_size"],
        "Laplacian": ["kernel_size"],
    },
    "轮廓操作": {
        "轮廓检测": [],
        "凸包": [],
    },
    "骨架提取": {
        "骨架提取": [],
    },
    "距离变换": {
        "距离变换": [],
    },
    "模板匹配": {
        "模板匹配": [],
    }
}


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(100, 100, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
        
        # 存储模板图像和源图像
        self.template_image = None
        self.source_image = None
        self.is_template_matching = False
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局（竖直）
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部：标签栏
        top_layout = QHBoxLayout()
        
        self.left_label = QLabel("绘画区（黑色笔刷绘画在白色背景上）")
        self.left_label.setFont(QFont("Consolas", 11, QFont.Bold))
        
        middle_label = QLabel("参数设置与操作")
        middle_label.setFont(QFont("Consolas", 11, QFont.Bold))
        middle_label.setAlignment(Qt.AlignCenter)
        
        right_label = QLabel("处理结果")
        right_label.setFont(QFont("Consolas", 11, QFont.Bold))
        right_label.setAlignment(Qt.AlignRight)
        
        top_layout.addWidget(self.left_label, 1)
        top_layout.addWidget(middle_label, 0)
        top_layout.addWidget(right_label, 1)
        
        main_layout.addLayout(top_layout)
        
        # 中间：内容区域（水平三栏）
        content_layout = QHBoxLayout()
        
        # ===== 左侧：绘画或ROI选择区域（动态切换）=====
        left_content_layout = QVBoxLayout()
        self.canvas = DrawingCanvas(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.roi_canvas = ROICanvas(CANVAS_WIDTH, CANVAS_HEIGHT)
        
        self.left_stack_widget = QWidget()
        self.left_stack_layout = QVBoxLayout(self.left_stack_widget)
        
        # 画布容器
        self.canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addStretch()
        
        # ROI 容器
        self.roi_container = QWidget()
        roi_layout = QVBoxLayout(self.roi_container)
        roi_layout.addWidget(self.roi_canvas)
        
        # ROI 按钮
        roi_button_layout = QHBoxLayout()
        self.import_btn = QPushButton("导入图片")
        self.import_btn.clicked.connect(self.import_template_image)
        self.set_roi_btn = QPushButton("指定区域")
        self.set_roi_btn.clicked.connect(self.confirm_template_roi)
        roi_button_layout.addWidget(self.import_btn)
        roi_button_layout.addWidget(self.set_roi_btn)
        roi_layout.addLayout(roi_button_layout)
        
        self.left_stack_layout.addWidget(self.canvas_container)
        self.left_stack_layout.addWidget(self.roi_container)
        self.roi_container.hide()
        
        left_content_layout.addWidget(self.left_stack_widget)
        left_content_layout.addStretch()
        
        # ===== 中间：参数选择和操作区域 =====
        middle_content_layout = QVBoxLayout()
        
        # 笔刷设置区域
        self.brush_group = QGroupBox("笔刷设置")
        brush_layout = QFormLayout(self.brush_group)
        
        brush_size_label = QLabel("笔刷大小:")
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setMinimum(1)
        self.brush_size_spinbox.setMaximum(50)
        self.brush_size_spinbox.setValue(DEFAULT_BRUSH_SIZE)
        self.brush_size_spinbox.valueChanged.connect(
            lambda v: self.canvas.set_brush_size(v)
        )
        brush_layout.addRow(brush_size_label, self.brush_size_spinbox)
        
        clear_btn = QPushButton("清空画布")
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        brush_layout.addRow(clear_btn)
        
        middle_content_layout.addWidget(self.brush_group)
        
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
        self.update_operator_combo()  # 先填充操作符列表
        self.operator_combo.currentTextChanged.connect(self.on_operator_changed)
        middle_content_layout.addWidget(self.operator_combo)
        
        # 参数设置区域 - 使用QVBoxLayout而不是QFormLayout
        self.params_group = QGroupBox("参数设置")
        self.params_layout = QVBoxLayout(self.params_group)
        
        # 核大小参数 - 使用子容器
        kernel_container = QWidget()
        kernel_h_layout = QHBoxLayout(kernel_container)
        self.kernel_label = QLabel("核大小")
        self.kernel_spinbox = QSpinBox()
        self.kernel_spinbox.setMinimum(1)
        self.kernel_spinbox.setMaximum(21)
        self.kernel_spinbox.setValue(DEFAULT_KERNEL_SIZE)
        self.kernel_spinbox.setSingleStep(2)
        kernel_h_layout.addWidget(self.kernel_label)
        kernel_h_layout.addWidget(self.kernel_spinbox)
        self.kernel_container = kernel_container
        self.params_layout.addWidget(kernel_container)
        kernel_container.hide()  # 初始隐藏
        
        # Canny低阈值 - 使用子容器
        threshold1_container = QWidget()
        threshold1_h_layout = QHBoxLayout(threshold1_container)
        self.threshold1_label = QLabel("低阈值")
        self.threshold1_spinbox = QSpinBox()
        self.threshold1_spinbox.setMinimum(0)
        self.threshold1_spinbox.setMaximum(255)
        self.threshold1_spinbox.setValue(DEFAULT_CANNY_THRESHOLD1)
        threshold1_h_layout.addWidget(self.threshold1_label)
        threshold1_h_layout.addWidget(self.threshold1_spinbox)
        self.threshold1_container = threshold1_container
        self.params_layout.addWidget(threshold1_container)
        threshold1_container.hide()  # 初始隐藏
        
        # Canny高阈值 - 使用子容器
        threshold2_container = QWidget()
        threshold2_h_layout = QHBoxLayout(threshold2_container)
        self.threshold2_label = QLabel("高阈值")
        self.threshold2_spinbox = QSpinBox()
        self.threshold2_spinbox.setMinimum(0)
        self.threshold2_spinbox.setMaximum(255)
        self.threshold2_spinbox.setValue(DEFAULT_CANNY_THRESHOLD2)
        threshold2_h_layout.addWidget(self.threshold2_label)
        threshold2_h_layout.addWidget(self.threshold2_spinbox)
        self.threshold2_container = threshold2_container
        self.params_layout.addWidget(threshold2_container)
        threshold2_container.hide()  # 初始隐藏
        
        self.params_layout.addStretch()
        
        middle_content_layout.addWidget(self.params_group)
        
        # 导入源图像按钮（仅在模板匹配时显示）
        self.import_source_btn = QPushButton("导入源图像")
        self.import_source_btn.clicked.connect(self.import_source_image)
        self.import_source_btn.hide()
        middle_content_layout.addWidget(self.import_source_btn)
        
        # 运行按钮
        run_btn = QPushButton("运行算子")
        run_btn.setFont(QFont("Consolas", 12, QFont.Bold))
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        run_btn.clicked.connect(self.run_operator)
        middle_content_layout.addWidget(run_btn)
        
        middle_content_layout.addStretch()
        
        # ===== 右侧：结果显示区域 =====
        right_content_layout = QVBoxLayout()
        self.result_display = ResultDisplay(CANVAS_WIDTH, CANVAS_HEIGHT)
        right_content_layout.addWidget(self.result_display)
        right_content_layout.addStretch()
        
        # 组合左中右三栏
        content_layout.addLayout(left_content_layout, 1)
        content_layout.addLayout(middle_content_layout, 0)
        content_layout.addLayout(right_content_layout, 1)
        
        main_layout.addLayout(content_layout)
        
        # ===== 下方：统计信息面板 =====
        stats_panel_layout = QVBoxLayout()
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-top: 1px solid #ddd;")
        stats_panel_layout.addWidget(self.stats_label)
        
        # 创建容器widget来容纳统计面板
        stats_container = QWidget()
        stats_container.setLayout(stats_panel_layout)
        stats_container.setMaximumHeight(STATS_PANEL_HEIGHT)
        
        main_layout.addWidget(stats_container)
        central_widget.setLayout(main_layout)
        
        # 初始化参数显示
        self.update_params_display()
    
    def on_category_changed(self, category):
        """当分类改变时更新算子列表"""
        self.is_template_matching = (category == "模板匹配")
        
        # 切换左侧面板
        if self.is_template_matching:
            self.left_label.setText("模板选择（导入图片并指定模板区域）")
            self.canvas_container.hide()
            self.roi_container.show()
            self.brush_group.hide()
            self.import_source_btn.show()
        else:
            self.left_label.setText("绘画区（黑色笔刷绘画在白色背景上）")
            self.canvas_container.show()
            self.roi_container.hide()
            self.brush_group.show()
            self.import_source_btn.hide()
        
        self.update_operator_combo()
        self.update_params_display()
    
    def on_operator_changed(self, operator_name):
        """当算子改变时更新参数显示"""
        self.update_params_display()
    
    def update_operator_combo(self):
        """更新算子下拉框"""
        current_category = self.category_combo.currentText()
        if current_category in OPERATORS:
            operators = list(OPERATORS[current_category].keys())
            self.operator_combo.blockSignals(True)
            self.operator_combo.clear()
            self.operator_combo.addItems(operators)
            self.operator_combo.blockSignals(False)
    
    def update_params_display(self):
        """根据选定的算子更新参数显示"""
        category = self.category_combo.currentText()
        operator_name = self.operator_combo.currentText()
        
        # 获取该算子需要的参数
        required_params = set()
        if category in OPERATOR_PARAMS:
            if operator_name in OPERATOR_PARAMS[category]:
                required_params = set(OPERATOR_PARAMS[category][operator_name])
        
        # 调试输出
        # print(f"DEBUG: update_params_display() - category='{category}', operator='{operator_name}', params={required_params}")
        
        # 隐藏/显示核大小容器
        if "kernel_size" in required_params:
            self.kernel_container.show()
        else:
            self.kernel_container.hide()
        
        # 隐藏/显示低阈值容器
        if "threshold1" in required_params:
            self.threshold1_container.show()
        else:
            self.threshold1_container.hide()
        
        # 隐藏/显示高阈值容器
        if "threshold2" in required_params:
            self.threshold2_container.show()
        else:
            self.threshold2_container.hide()
        
        # 如果没有参数，显示提示
        if not required_params:
            self.params_group.setTitle("参数设置（无）")
        else:
            self.params_group.setTitle("参数设置")
    
    def update_stats_display(self, stats):
        """更新底部统计信息面板"""
        if not stats:
            self.stats_label.setText("")
            return
        
        # 格式化统计信息为HTML
        stats_html = "<b>处理结果统计：</b><br>"
        for key, value in stats.items():
            if isinstance(value, float):
                stats_html += f"<b>{key}:</b> {value:.4f}<br>"
            else:
                stats_html += f"<b>{key}:</b> {value}<br>"
        
        self.stats_label.setText(stats_html)
    
    def run_operator(self):
        """运行选定的算子"""
        try:
            category = self.category_combo.currentText()
            operator_name = self.operator_combo.currentText()
            
            # 模板匹配逻辑
            if category == "模板匹配":
                if self.template_image is None or self.template_image.size == 0:
                    QMessageBox.warning(self, "警告", "请先指定模板区域")
                    return
                
                if self.source_image is None or self.source_image.size == 0:
                    QMessageBox.warning(self, "警告", "请先导入源图像")
                    return
                
                operator_func = OPERATORS[category][operator_name]
                result_image, stats = operator_func(self.source_image, self.template_image)
                
                self.result_display.set_image(result_image)
                self.update_stats_display(stats)
                return
            
            # 其他算子逻辑
            input_image = self.canvas.get_image_array()
            # input_image = 255 - input_image
            
            # if np.sum(input_image) == 0:
            if np.sum(input_image) == 255:
                QMessageBox.warning(self, "警告", "请先在画布上绘画")
                return
            
            operator_func = OPERATORS[category][operator_name]
            
            required_params = set()
            if category in OPERATOR_PARAMS:
                if operator_name in OPERATOR_PARAMS[category]:
                    required_params = set(OPERATOR_PARAMS[category][operator_name])
            
            kernel_size = self.kernel_spinbox.value()
            if kernel_size % 2 == 0:
                kernel_size -= 1
            
            threshold1 = self.threshold1_spinbox.value()
            threshold2 = self.threshold2_spinbox.value()
            
            if "threshold1" in required_params and "threshold2" in required_params:
                result_image, stats = operator_func(input_image, threshold1, threshold2)
            elif "kernel_size" in required_params:
                result_image, stats = operator_func(input_image, kernel_size)
            else:
                result_image, stats = operator_func(input_image, kernel_size)
            
            self.result_display.set_image(result_image)
            self.update_stats_display(stats)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出错:\n{str(e)}")
    
    def import_template_image(self):
        """导入模板图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模板图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self.roi_canvas.load_image(file_path)
                self.template_image = None
                self.roi_canvas.clear_roi()
                QMessageBox.information(self, "提示", "请在图像上绘制矩形框选择模板区域")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")
    
    def confirm_template_roi(self):
        """启用指定区域模式（开始绘制矩形框）"""
        if self.roi_canvas.original_image is None:
            QMessageBox.warning(self, "警告", "请先导入图片")
            return
        
        # 启用绘制模式
        self.roi_canvas.enable_drawing()
        QMessageBox.information(self, "提示", "请在图像上绘制矩形框来指定模板区域")
        
        # 连接 ROI 变化信号，在绘制完成后自动确认
        self.roi_canvas.roi_changed.connect(self.auto_confirm_template_roi)
    
    def auto_confirm_template_roi(self):
        """自动确认模板 ROI（在鼠标抬起时调用）"""
        roi_image = self.roi_canvas.get_roi_image()
        
        if roi_image is not None and roi_image.size > 0:
            self.template_image = roi_image
            # 断开信号连接，避免重复确认
            try:
                self.roi_canvas.roi_changed.disconnect(self.auto_confirm_template_roi)
            except:
                pass
            QMessageBox.information(
                self, "成功", 
                f"模板区域已确定，大小: {roi_image.shape[1]}x{roi_image.shape[0]}"
            )
    
    def import_source_image(self):
        """导入源图像用于匹配"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择源图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 使用 PIL 读取图像以支持中文路径
                from PIL import Image
                pil_image = Image.open(file_path).convert('L')  # 转换为灰度图
                image = np.array(pil_image)
                
                if image is None or image.size == 0:
                    raise ValueError("无法加载图像")
                
                # 应用与 ROI 画布相同的缩放逻辑，确保尺寸一致
                # 如果图像大于 ROI 画布的显示区域，按比例缩放
                roi_canvas = self.roi_canvas
                if image.shape[0] > roi_canvas.height or image.shape[1] > roi_canvas.width:
                    scale = min(roi_canvas.width / image.shape[1], roi_canvas.height / image.shape[0])
                    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                    image = cv2.resize(image, new_size)
                
                self.source_image = image
                QMessageBox.information(
                    self, "成功",
                    f"源图像已导入，大小: {image.shape[1]}x{image.shape[0]}"
                )
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")
