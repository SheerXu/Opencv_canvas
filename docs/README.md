# OpenCV算子预览工具

一个功能完整的图形化OpenCV算子预览工具，用于展示各种图像处理算子的实时效果。

## 功能特性

### 1. 交互式绘画
- 在左侧画布上绘制二值化图像（黑色背景上白色笔刷绘画）
- 支持自定义笔刷大小
- 支持清空画布重新绘画

### 2. 支持的算子

#### 形态学操作
- 腐蚀（Erosion）
- 膨胀（Dilation）
- 开运算（Opening）
- 闭运算（Closing）
- 形态学梯度（Morphological Gradient）

#### 边缘检测
- Canny边缘检测
- Sobel X方向检测
- Sobel Y方向检测
- Laplacian边缘检测

#### 轮廓操作
- 轮廓检测（Contour Detection）
- 凸包（Convex Hull）

#### 骨架提取
- 骨架提取（Skeleton Extraction）

#### 距离变换
- 欧氏距离变换（Euclidean Distance Transform）

### 3. 实时处理
- 选择算子后立即显示参数设置
- 点击"运行算子"执行处理
- 右侧实时显示处理结果和统计信息

### 4. 参数调节
- 核大小（Kernel Size）：用于形态学操作
- 低阈值/高阈值：用于Canny边缘检测
- 其他相关参数

### 5. 统计信息
- 处理后的白色像素数量
- 操作类型和参数
- 图像尺寸
- 其他相关的数值指标

## 项目结构

```
Opencv_canvas/             # 主入口
├── docs
│   └── README.md              # 本文件
├── operators/
│   ├── __init__.py
│   └── operators.py       # 各类算子实现
├── ui/
│   ├── __init__.py
│   ├── main_window.py     # 主窗口
│   ├── drawing_canvas.py  # 绘画画布
│   └── result_display.py  # 结果显示
└── main.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行主程序
```bash
python main.py
```

2. 绘画
   - 在左侧白色区域使用鼠标绘画二值化图像
   - 黑色背景，白色笔刷

3. 选择算子
   - 在中间的下拉菜单选择算子分类
   - 选择具体的算子

4. 设置参数
   - 根据需要调整参数（核大小、阈值等）

5. 运行处理
   - 点击"运行算子"按钮
   - 在右侧查看处理结果和统计信息

## 快捷键

- 清空画布：点击"清空画布"按钮

## 扩展指南

### 添加新的算子

在 `operators/operators.py` 中：

1. 创建新的操作类
2. 在类中添加静态方法实现算子
3. 在 `OPERATORS` 字典中注册

示例：
```python
class NewOperator:
    @staticmethod
    def my_operator(image, param1=5):
        result = ...  # 处理逻辑
        stats = {"操作": "我的算子", "参数": param1}
        return result, stats

# 在OPERATORS中注册
OPERATORS["新分类"] = {
    "我的算子": NewOperator.my_operator
}
```