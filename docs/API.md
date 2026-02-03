# OpenCV算子预览工具

### 1. 用户界面
- ✅ 交互式绘画画布（左侧）
  - 支持鼠标绘画二值化图像
  - 可调节笔刷大小（1-50像素）
  - 白色笔刷在黑色背景上绘画
  - 支持清空画布

- ✅ 参数控制面板（中间）
  - 算子分类选择（下拉框）
  - 具体算子选择（下拉框）
  - 参数设置组（核大小、阈值等）
  - 运行按钮

- ✅ 结果显示区域（右侧）
  - 处理结果图像显示
  - 实时统计信息显示
  - 可滚动的信息面板

### 2. 算子实现 (5大类, 13个算子)

#### 形态学操作 (5个)
- 腐蚀 (Erosion)
- 膨胀 (Dilation)
- 开运算 (Opening)
- 闭运算 (Closing)
- 形态学梯度 (Morphological Gradient)

#### 边缘检测 (4个)
- Canny边缘检测
- Sobel X方向检测
- Sobel Y方向检测
- Laplacian检测

#### 轮廓操作 (2个)
- 轮廓检测
- 凸包

#### 骨架提取 (1个)
- 骨架提取

#### 距离变换 (1个)
- 欧氏距离变换

### 3. 核心特性
- ✅ 实时处理和反馈
- ✅ 参数灵活调节
- ✅ 详细的统计信息输出
- ✅ 模块化架构设计

## 项目架构

```
Opencv_canvas/
│
├── main.py                      # 应用入口
├── config.py                    # 配置文件
├── demo.py                      # 命令行演示脚本
│
├── operators/                   # 算子模块
│   ├── __init__.py
│   └── operators.py            # 所有算子实现
│
├── ui/                          # UI模块
│   ├── __init__.py
│   ├── main_window.py          # 主窗口
│   ├── drawing_canvas.py       # 绘画画布
│   └── result_display.py       # 结果显示
│
├── README.md                    # 项目说明
├── GUIDE.md                     # 使用指南
├── requirements.txt             # 依赖列表
└── API.md                       # API文档
```

## 使用方法

### 启动应用
```bash
python main.py
```

### 运行演示
```bash
python demo.py
```

## API文档

### operators 模块

#### MorphologyOperator
```python
from operators import MorphologyOperator
import numpy as np

image = np.zeros((300, 300), dtype=np.uint8)
# ... 绘制二值化图像 ...

# 腐蚀操作
result, stats = MorphologyOperator.erode(image, kernel_size=5)

# 膨胀操作
result, stats = MorphologyOperator.dilate(image, kernel_size=5)

# 开运算
result, stats = MorphologyOperator.open(image, kernel_size=5)

# 闭运算
result, stats = MorphologyOperator.close(image, kernel_size=5)

# 形态学梯度
result, stats = MorphologyOperator.gradient(image, kernel_size=5)
```

#### EdgeDetectionOperator
```python
from operators import EdgeDetectionOperator

# Canny边缘检测
result, stats = EdgeDetectionOperator.canny(image, threshold1=100, threshold2=200)

# Sobel X检测
result, stats = EdgeDetectionOperator.sobel_x(image, ksize=3)

# Sobel Y检测
result, stats = EdgeDetectionOperator.sobel_y(image, ksize=3)

# Laplacian检测
result, stats = EdgeDetectionOperator.laplacian(image, ksize=1)
```

#### ContourOperator
```python
from operators import ContourOperator

# 轮廓检测
result, stats = ContourOperator.find_contours(image)

# 凸包检测
result, stats = ContourOperator.convex_hull(image)
```

#### SkeletonOperator
```python
from operators import SkeletonOperator

# 骨架提取
result, stats = SkeletonOperator.skeleton(image)
```

#### DistanceOperator
```python
from operators import DistanceOperator

# 距离变换
result, stats = DistanceOperator.distance_transform(image)
```

### UI 模块

#### DrawingCanvas
```python
from ui import DrawingCanvas
from PyQt5.QtGui import QColor

# 创建画布
canvas = DrawingCanvas(width=400, height=300)

# 获取画布内容为NumPy数组
image_array = canvas.get_image_array()

# 从NumPy数组设置画布内容
canvas.set_image_array(image_array)

# 设置笔刷大小
canvas.set_brush_size(10)

# 设置笔刷颜色
canvas.set_brush_color(QColor(255, 255, 255))

# 清空画布
canvas.clear_canvas()
```

#### ResultDisplay
```python
from ui import ResultDisplay
import numpy as np

# 创建结果显示器
display = ResultDisplay(width=400, height=300)

# 显示处理结果图像
result_image = np.zeros((300, 300), dtype=np.uint8)
display.set_image(result_image)

# 显示统计信息
stats = {
    "操作": "示例操作",
    "参数1": "值1",
    "参数2": "值2"
}
display.set_stats(stats)

# 清空显示
display.clear()
```

## 扩展指南

### 添加新的算子分类

1. 在 `operators/operators.py` 中创建新的算子类：

```python
class NewOperatorCategory:
    @staticmethod
    def new_operator(image: np.ndarray, param1: int = 5) -> Tuple[np.ndarray, Dict]:
        """新算子的实现"""
        # 处理逻辑
        result = ...
        
        # 统计信息
        stats = {
            "操作": "新算子",
            "参数1": param1,
            "结果": value
        }
        return result, stats
```

2. 在 `OPERATORS` 字典中注册：

```python
OPERATORS["新分类"] = {
    "新算子": NewOperatorCategory.new_operator,
    "另一个算子": NewOperatorCategory.another_operator
}
```

3. 重启应用后新算子会自动出现

### 自定义参数

编辑 `config.py` 文件修改默认参数。

### 修改UI样式

编辑 `ui/*.py` 文件中的QSS样式表。

## 依赖版本

- Python: 3.8+
- OpenCV: 4.10+
- PyQt5: 5.15+
- NumPy: 2.2.6

## 未来改进方向

- [ ] 支持导入/保存图像
- [ ] 添加撤销功能
- [ ] 优化ui画布缩放适配
- [ ] 优化ui窗体缩放适配
- [ ] 支持自定义结构元素形状
- [ ] 添加图像处理管道
- [ ] 支持批量/队列处理
- [ ] 添加更多统计信息
- [ ] 添加更多算法
  - [ ] Hough变换
  - [ ] 傅里叶变换
- [ ] 添加聚类算法
  - [ ] k-means聚类
  - [ ] dbscan聚类
- [ ] 性能优化和多线程支持
- [ ] 对于大图像处理，考虑添加进度条