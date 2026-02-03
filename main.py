#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenCV算子预览工具 - 主程序
用于预览各个OpenCV算子的处理效果
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
