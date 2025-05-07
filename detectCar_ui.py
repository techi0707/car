import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QFrame
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QMainWindow
import cv2
import numpy as np
from ultralytics import YOLO

class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化YOLO模型
        self.model = YOLO('car.pt')  # 假设你使用YOLOv8模型，可以根据需要修改

        self.setWindowTitle("车辆损伤识别系统")
        # self.setGeometry(100, 100, 1000, 600)
        # 设置窗口最小尺寸
        self.setMinimumSize(1000, 600)
        # 创建UI
        self.init_ui()

    def init_ui(self):
       # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)  # 设置布局间距
        
        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Panel | QFrame.Raised)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(300)  # 限制左侧面板宽度
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)  # 设置边距
        
        # 标题
        title_label = QLabel("控制面板")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 20px;")
        left_layout.addWidget(title_label)
        
        # 文件选择按钮
        self.select_button = QPushButton('选择图片', self)
        self.select_button.setMinimumHeight(40)
        self.select_button.clicked.connect(self.select_image)
        left_layout.addWidget(self.select_button)
        
        # 置信度调整
        conf_label = QLabel('置信度调整:')
        left_layout.addWidget(conf_label)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        left_layout.addWidget(self.confidence_slider)
        
        # 显示当前置信度值
        self.conf_value_label = QLabel('50%')
        self.confidence_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f'{v}%'))
        left_layout.addWidget(self.conf_value_label)
        
        # 开始检测按钮
        self.start_button = QPushButton('开始检测', self)
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_detection)
        left_layout.addWidget(self.start_button)
        
        left_layout.addStretch()  # 添加弹性空间
        main_layout.addWidget(left_panel)
        
        # 右侧图像显示区域改为水平布局
        right_layout = QVBoxLayout()
        image_layout = QHBoxLayout()  # 新增水平布局用于并排显示图片
        
        # 原始图片显示
        original_frame = QFrame()
        original_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        original_layout = QVBoxLayout(original_frame)
        original_title = QLabel("原始图片")
        original_title.setAlignment(Qt.AlignCenter)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)  # 调整尺寸
        original_layout.addWidget(original_title)
        original_layout.addWidget(self.original_label)
        image_layout.addWidget(original_frame)
        
        # 检测结果显示
        result_frame = QFrame()
        result_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        result_layout = QVBoxLayout(result_frame)
        result_title = QLabel("检测结果")
        result_title.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 400)  # 调整尺寸
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        image_layout.addWidget(result_frame)
        
        right_layout.addLayout(image_layout)  # 将水平布局添加到右侧布局中
        main_layout.addLayout(right_layout)
        
        self.image_path = None

    
        # 修改图片显示区域的布局和大小策略
        self.original_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.original_label.setMinimumSize(400, 400)
        self.result_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.result_label.setMinimumSize(400, 400)
        
        # 添加事件过滤器来处理窗口大小改变
        self.installEventFilter(self)
    def select_image(self):
        # 选择文件对话框
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "Images (*.png *.jpg *.bmp)", options=options)
        if file_path:
            self.image_path = file_path
            self.display_image(self.image_path)

    # def display_image(self, path):
    #     # 显示原始图片
    #     pixmap = QPixmap(path)
    #     # 计算保持宽高比的缩放大小
    #     scaled_size = pixmap.size()
    #     scaled_size.scale(800, 800, Qt.KeepAspectRatio)
    #     self.original_label.setPixmap(pixmap.scaled(
    #         scaled_size.width(), 
    #         scaled_size.height(), 
    #         Qt.KeepAspectRatio,
    #         Qt.SmoothTransformation  # 使用平滑转换提高质量
    #     ))

    def start_detection(self):
        if not self.image_path:
            return

        # 获取滑块的值，作为置信度
        confidence = self.confidence_slider.value() / 100

        # 加载图像进行推理
        image = cv2.imread(self.image_path)
        results = self.model(image)  # 推理过程
        
        # 获取第一个图像的检测结果
        result = results[0]  # 获取第一张图片的结果
        boxes = result.boxes.data  # 获取边界框数据
        
        # 绘制检测框
        image_with_boxes = image.copy()
        for box in boxes:
            conf = float(box[4])
            cls = int(box[5])
            if conf >= confidence:
                x1, y1, x2, y2 = map(int, box[:4])
                # 使用更粗的线条绘制边界框
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # 改善文字显示
                class_name = result.names[cls]
                label = f'{class_name} {conf:.2f}'
                
                # 获取文字大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0  # 增大字体大小
                thickness = 2     # 增加字体粗细
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # 绘制文字背景框
                cv2.rectangle(image_with_boxes, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            (0, 255, 0), 
                            -1)  # -1表示填充矩形
                
                # 绘制文字（使用黑色以提高对比度）
                cv2.putText(image_with_boxes, 
                        label, 
                        (x1, y1 - 5),  # 稍微调整位置
                        font,
                        font_scale,
                        (0, 0, 0),     # 黑色文字
                        thickness,
                        cv2.LINE_AA)   # 使用抗锯齿

        # 显示检测结果
        self.display_result(image_with_boxes)

    def draw_boxes(self, image, boxes, confidence):
        # 绘制检测框
        for box in boxes:
            if box[4] >= confidence:  # 只绘制置信度大于等于滑块值的框
                x, y, w, h = map(int, box[:4])
                cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)

        return image


    # def display_result(self, image):
    #     # 将检测结果转化为QPixmap显示
    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     h, w, ch = image_rgb.shape
    #     bytes_per_line = ch * w
    #     # 创建QImage对象
    #     qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #     pixmap = QPixmap.fromImage(qimg)
        
    #     # 计算保持宽高比的缩放大小
    #     scaled_size = pixmap.size()
    #     scaled_size.scale(800, 800, Qt.KeepAspectRatio)
    #     self.result_label.setPixmap(pixmap.scaled(
    #         scaled_size.width(), 
    #         scaled_size.height(), 
    #         Qt.KeepAspectRatio,
    #         Qt.SmoothTransformation  # 使用平滑转换提高质量
    #     ))

    def eventFilter(self, obj, event):
        if obj is self and event.type() == QEvent.Resize:
            self.updateImageDisplay()
        return super().eventFilter(obj, event)

    def updateImageDisplay(self):
        if hasattr(self, '_current_original_pixmap'):
            self.displayScaledPixmap(self.original_label, self._current_original_pixmap)
        if hasattr(self, '_current_result_pixmap'):
            self.displayScaledPixmap(self.result_label, self._current_result_pixmap)

    def displayScaledPixmap(self, label, pixmap):
        # 获取标签的大小
        label_size = label.size()
        # 保持宽高比缩放图片
        scaled_pixmap = pixmap.scaled(
            label_size.width(), 
            label_size.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def display_image(self, path):
        pixmap = QPixmap(path)
        self._current_original_pixmap = pixmap
        self.displayScaledPixmap(self.original_label, pixmap)

    def display_result(self, image):
        # 将检测结果转化为QPixmap显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._current_result_pixmap = pixmap
        self.displayScaledPixmap(self.result_label, pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())
