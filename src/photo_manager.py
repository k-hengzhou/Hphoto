import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QScrollArea, QLabel, 
                            QFileDialog, QGridLayout, QMessageBox, QProgressBar, 
                            QDialog, QStackedWidget, QSizePolicy, QLineEdit, 
                            QMenu, QComboBox,QCheckBox,QFrame)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, 
                         QEasingCurve, QSize, QRect, QObject, QPoint, QEvent,QProcess)
from PyQt5.QtGui import QPixmap, QPainter, QImage, QImageReader
from pathlib import Path
from .face_organizer import FaceOrganizer
from .remove_duplicates import DuplicateRemover
import threading
import sip
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import QIcon
import qtawesome as qta
import time
import cv2
from .utils import (copy_photo,cv2_to_qpixmap,get_config,normalize_path)
from .clean_face_db import clean_face_database
from .face_clustering import FaceClusterer
import psutil
from .nsfw_classifier import NSFWClassifier,default_nsfw_class
class VSCodeTooltip(QWidget):
    def __init__(self, title, shortcut=None, description=None, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)  # 设置为工具提示窗口
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 设置样式
        self.setStyleSheet("""
            VSCodeTooltip {
                background-color: #252526;
                border: 1px solid #454545;
                border-radius: 4px;
            }
            QLabel {
                color: #cccccc;
                font-family: "Segoe UI", Arial;
                font-size: 12px;
                padding: 4px;
            }
        """)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # 添加内容
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        shortcut_label = QLabel(shortcut)
        shortcut_label.setStyleSheet("color: #858585;")
        
        header.addWidget(title_label)
        header.addWidget(shortcut_label)
        layout.addLayout(header)
        
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #858585;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
class QMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        # 设置透明背景
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QMessageBox {
                background-color: rgba(32, 32, 32, 0.85);
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-family: "Segoe UI", Arial;
                font-size: 24px;    
                color: rgba(255, 255, 255, 0.95);
            }
            QMessageBox QLabel {
                color: rgba(255, 255, 255, 0.95);
            }
            QMessageBox QPushButton {
                background-color: rgba(98, 114, 164, 0.9);
                color: rgba(255, 255, 255, 0.95);
                border: none;
                border-radius: 4px;
                padding: 10px 10px;
                font-family: "Segoe UI", Arial;
                font-size: 14px;
            }
            QMessageBox QPushButton:hover {
                background-color: rgba(128, 128, 128, 0.3);
            
            }   
        """) 
    @staticmethod
    def information(parent, title, text):
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        return msg.exec_()

    @staticmethod
    def warning(parent, title, text):
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        return msg.exec_()

    @staticmethod
    def critical(parent, title, text):
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        return msg.exec_()
    @staticmethod
    def question(parent, title, text):
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return msg.exec_()
class KeyButton(QPushButton):
    def __init__(self, text, tooltip=None, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(45, 45)  # 设置固定大小
        self.tooltip_widget = None
        if tooltip:
            self.tooltip_widget = VSCodeTooltip(tooltip)
            self.tooltip_widget.hide()  # 初始时隐藏    
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #cccccc;
                font-family: "Segoe UI", Arial;
                font-size: 24px;
                padding: 0px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: rgba(128, 128, 128, 0.3);
            }
                                       /* 工具提示样式 */
            QToolTip {
                background-color: #252526;
                border: 1px solid #454545;
                color: #cccccc;
                padding: 20px;
                font-family: "Segoe UI", Arial;
                font-size: 12px;
                border-radius: 4px;
            }
        """)
        self.installEventFilter(self)
    def showTooltip(self):
        # 获取提示框的大小
        tooltip_size = self.tooltip_widget.sizeHint()
            
        # 获取按钮的全局位置
        button_pos = self.mapToGlobal(QPoint(0, 0))
            
        # 计算理想的位置（按钮正下方居中）
        x = button_pos.x() + (self.width() - tooltip_size.width()) // 2
        y = button_pos.y() + self.height() + 5
            
        # 获取屏幕大小
        screen = QApplication.primaryScreen().geometry()
            
        # 确保提示框不会超出屏幕边界
        if x < screen.left():
            x = screen.left() + 5
        elif x + tooltip_size.width() > screen.right():
            x = screen.right() - tooltip_size.width() - 5
                
        if y + tooltip_size.height() > screen.bottom():
            # 如果下方空间不够，显示在按钮上方
            y = button_pos.y() - tooltip_size.height() - 5
            
        # 设置位置并显示
        self.tooltip_widget.move(x, y)
        self.tooltip_widget.show()
            
    def eventFilter(self, obj, event):
        if self.tooltip_widget is not None:
            if obj == self:
                if event.type() == QEvent.Enter:
                    self.showTooltip()
                elif event.type() == QEvent.Leave:
                    if hasattr(self, 'tooltip_widget'):
                        self.tooltip_widget.hide()
                        self.tooltip_widget.close()
        return super().eventFilter(obj, event)
class PhotoManager(QMainWindow):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("照片管理器")
        self.setGeometry(100, 100, 1500, 900)
        
        # 加载配置
        self.config_file = "config/config.json"
        self.load_config()
        self.style_name =["重复文件","未识别","无人脸","未知人物"]
        self.style_name.extend(default_nsfw_class)
        # 移除默认的窗口标题栏
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # 创建主布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建标题栏
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(70)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)
        title_layout.setSpacing(15)
        
        # 创建左侧功能按钮容器
        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(15)
        
        # 添加功能按钮到左侧容器
        self.open_btn = KeyButton("", tooltip="打开文件夹")
        self.open_btn.setIcon(qta.icon('fa5s.folder-open', color='#c8c8c8'))
        self.folder_btn = KeyButton("", tooltip="添加文件夹")
        self.folder_btn.setIcon(qta.icon('fa5s.folder', color='#c8c8c8'))
        self.file_btn = KeyButton("", tooltip="添加文件")
        self.file_btn.setIcon(qta.icon('fa5s.file-image', color='#c8c8c8'))
        self.register_btn = KeyButton("", tooltip="人脸注册")
        self.register_btn.setIcon(qta.icon('fa5s.user-plus', color='#c8c8c8'))
        self.refresh_btn = KeyButton("", tooltip="刷新")
        self.refresh_btn.setIcon(qta.icon('fa5s.sync', color='#c8c8c8'))
        self.setting_btn = KeyButton("", tooltip="设置")
        self.setting_btn.setIcon(qta.icon('fa5s.cog', color='#c8c8c8'))
        
        for btn in (self.open_btn, self.folder_btn, self.file_btn, 
                    self.register_btn, self.refresh_btn, self.setting_btn):
            left_buttons.addWidget(btn)
        
        # 创建中间的切换按钮容器
        center_buttons = QHBoxLayout()
        center_buttons.setSpacing(30)  # 增加按钮间距
        
        # 创建切换按钮
        self.left_btn = KeyButton("", tooltip="返回人物界面")
        self.right_btn = KeyButton("", tooltip="查看风格界面")
        
        # 使用更现代的图标
        self.left_btn.setIcon(qta.icon('fa5s.angle-left', color='#c8c8c8', scale_factor=1.2))
        self.right_btn.setIcon(qta.icon('fa5s.angle-right', color='#c8c8c8', scale_factor=1.2))
        
        # 创建页面标题标签
        self.page_title = QLabel("人物界面")
        self.page_title.setStyleSheet("""
            QLabel {
                color: #c8c8c8;
                font-size: 16px;  /* 增大字体 */
                font-weight: bold;
                background-color: rgba(45, 45, 45, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 18px;  /* 与按钮圆角一致 */
                padding: 5px 25px;    /* 增加水平内边距 */
                min-width: 150px;     /* 设置最小宽度 */
            }
        """)
        self.page_title.setFixedHeight(36)  # 与按钮高度一致
        self.page_title.setAlignment(Qt.AlignCenter)
        
        # 设置切换按钮样式
        for btn in (self.left_btn, self.right_btn):
            btn.setFixedSize(36, 36)  # 稍微增大按钮尺寸
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(45, 45, 45, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 18px;
                }
                QPushButton:hover {
                    background-color: rgba(60, 60, 60, 0.6);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                QPushButton:disabled {
                    background-color: rgba(45, 45, 45, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                }
            """)
        
        # 初始状态
        self.left_btn.setEnabled(False)
        self.right_btn.setEnabled(True)
        
        # 在添加按钮前后添加一些额外的空间
        center_buttons.addStretch(1)
        center_buttons.addWidget(self.left_btn)
        center_buttons.addSpacing(5)  # 减小间距
        center_buttons.addWidget(self.page_title)
        center_buttons.addSpacing(5)  # 减小间距
        center_buttons.addWidget(self.right_btn)
        center_buttons.addStretch(1)
        
        # 创建右侧窗口控制按钮容器
        right_buttons = QHBoxLayout()
        right_buttons.setSpacing(15)
        
        # 添加窗口控制按钮
        min_btn = KeyButton("")
        min_btn.setIcon(qta.icon('fa5s.window-minimize', color='#c8c8c8'))
        max_btn = KeyButton("")
        max_btn.setIcon(qta.icon('fa5s.window-maximize', color='#c8c8c8'))
        close_btn = KeyButton("")
        close_btn.setIcon(qta.icon('fa5s.window-close', color='#c8c8c8'))
        
        for btn in (min_btn, max_btn, close_btn):
            btn.setFixedSize(70, 70)
            right_buttons.addWidget(btn)
        
        # 将三个部分添加到标题栏布局
        title_layout.addLayout(left_buttons)
        title_layout.addStretch()
        title_layout.addLayout(center_buttons)
        title_layout.addStretch()
        title_layout.addLayout(right_buttons)
        
        # 连接按钮信号
        self.open_btn.clicked.connect(self.open_folder)
        self.folder_btn.clicked.connect(self.add_folder)
        self.file_btn.clicked.connect(self.add_file)
        self.register_btn.clicked.connect(lambda: self.register_face())
        self.refresh_btn.clicked.connect(lambda: self.refresh_folder())
        self.setting_btn.clicked.connect(self.setting)
        
        # 连接切换按钮信号
        self.left_btn.clicked.connect(lambda: self.switch_cards_page(True))
        self.right_btn.clicked.connect(lambda: self.switch_cards_page(False))
        
        # 连接窗口控制按钮信号
        min_btn.clicked.connect(self.showMinimized)
        max_btn.clicked.connect(self.toggle_maximize)
        close_btn.clicked.connect(self.close)
        
        # 设置标题栏样式
        self.title_bar.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 0.95);
            }
        """)
        
        # 添加标题栏到主布局
        self.main_layout.addWidget(self.title_bar)
        
        # 创建内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # 将原有的堆叠布局添加到内容容器
        self.stacked_layout = QStackedWidget()
        content_layout.addWidget(self.stacked_layout)
        
        # 添加内容容器到主布局
        self.main_layout.addWidget(content_widget)
        
        # 初始化人脸识别器
        try:
            # 获取配置中的人脸库路径
            faces_db_path = get_config(self.config,'face_recognition','db_path','known_faces_op.pkl')    
            threshold = get_config(self.config,'face_recognition','threshold',0.5)
            update_db = get_config(self.config,'face_recognition','update_db',True)
            backup_db = get_config(self.config,'face_recognition','backup_db',False)
            self.face_organizer = FaceOrganizer(
                model_path="models/buffalo_l.onnx",
                faces_db_path=faces_db_path,  # 数据库路径
                threshold=threshold,                    # 匹配阈值
                update_db=update_db,                  # 允许更新数据库
                backup_db=backup_db                   # 启用数据库备份
            )
            
            self.face_db={}
            self.duplicate_woker = RemoveDuplicatesWorker(self.config['last_folder'],0.9)
            self.duplicate_woker.finished.connect(self.handle_duplicates)
            self.duplicate_woker.start()
            # 初始化NSFW分类器
            unsafe_threshold = get_config(self.config,'nsfw_classifier','unsafe_threshold',0.6) 
            model_path = get_config(self.config,'nsfw_classifier','model_path','D:/project/小程序/model/640m.onnx')
            providers = get_config(self.config,'nsfw_classifier','providers',['CPUExecutionProvider'])
            nsfw_class = get_config(self.config,'nsfw_classifier','nsfw_class',default_nsfw_class)
            # self.face_db["重复文件"] = duplicates
            # print("重复文件",len(duplicates))   
            # 使用 FaceOrganizer 的数据库
            # self.face_db = self.face_organizer.known_faces
            if self.face_organizer.known_faces is not None:
                print("人脸识别器初始化成功")
            else:
                print("人脸识别器为空")
            # print("人脸识别器初始化成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化人脸识别器失败: {str(e)}")
            sys.exit(1)
        
        # 创建第一个页面和滚动区域
        self.cards_page = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_page)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: rgba(32, 32, 32, 1);
                border: none;
            }
        """)
        self.cards_layout.addWidget(self.scroll_area)
        
        # 创建第二个页面和滚动区域
        self.cards_page_2 = QWidget()
        self.cards_layout_2 = QVBoxLayout(self.cards_page_2)
        
        self.scroll_area_2 = QScrollArea()
        self.scroll_area_2.setWidgetResizable(True)
        self.scroll_area_2.setStyleSheet("""
            QScrollArea {
                background-color: rgba(32, 32, 32, 1);
                border: none;
            }
        """)
        self.cards_layout_2.addWidget(self.scroll_area_2)
        
        # 创建预览页面
        self.preview_page = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_page)
        
        # 将所有页面添加到堆叠布局中
        self.stacked_layout.addWidget(self.cards_page)
        self.stacked_layout.addWidget(self.cards_page_2)
        self.stacked_layout.addWidget(self.preview_page)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()  # 初始状态隐藏进度条
        self.cards_layout.addWidget(self.progress_bar)
        self.image_paths = []
        self.current_index = 0
        # 工作线程
        self.worker = None
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: rgba(32, 32, 32, 0.85);
            }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {
                background-color: rgba(32, 32, 32, 0.85);
                border: none;
            }
            QProgressBar {
                border: none;
                background-color: rgba(45, 45, 45, 0.85);
                height: 4px;
            }
            QProgressBar::chunk {
                background-color: rgba(98, 114, 164, 0.9);
            }
            QLabel {
                color: rgba(255, 255, 255, 0.95);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(128, 128, 128, 0.5);
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(128, 128, 128, 0.7);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
                height: 0;
            }
        """)
        
        # 添加防抖定时器，减少延迟时间
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_person_grid)
        
        # 保存上一次的布局信息
        self.last_layout_info = {
            'width': 0,
            'cols': 0,
            'card_width': 0
        }
        
        # 动画持续时间（毫秒）
        self.animation_duration = 100
        
        # 减少防抖延迟时间（从50ms改为30ms）
        self.resize_debounce_time = 10
        
        # 添加窗口大小变化事件处理
        self.scroll_area.resizeEvent = self.on_scroll_area_resize
        self.photo_view_page = None
        # 设置窗口属性
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.98)
        
        start_dir = self.config['last_folder']
        if os.path.exists(start_dir):
            self.process_folder(start_dir)
        # 保存原始的键盘事件处理函数
        # self.original_keyPressEvent = self.keyPressEvent
        # # 重写键盘事件处理函数
        # self.keyPressEvent = self.new_keyPressEvent
    def handle_face_clustering(self,clusters):
        self.face_db.update(clusters)
        print("人脸聚类",len(clusters))
    def handle_duplicates(self,duplicates):
        self.face_db["重复文件"] = duplicates
        print("重复文件",len(duplicates))
    def clean_face_db(self,db_path,output_path,method,eps,min_samples,backup):
        """清理人脸数据库"""
        msg = clean_face_database(db_path=db_path,output_path=output_path,method=method,eps=float(eps),min_samples=int(min_samples)   ,backup=backup)
        QMessageBox.information(self, "清理结果", msg)
    def add_face_clustering(self,min_samples,eps):
        """添加人脸聚类"""
        self.face_clustering_worker = FaceClusteringWorker(self.config['last_folder']+"/未识别",int(min_samples),float(eps))
        self.face_clustering_worker.finished.connect(self.handle_face_clustering)
        self.face_clustering_worker.start()
        # msg = add_face_clustering(min_samples=int(min_samples),eps=float(eps))
        # QMessageBox.information(self, "人脸聚类结果", msg)
    def nsfw_classify(self,model_path,providers,threshold,nsfw_class):
        """NSFW分类"""
        nsfw_classifier = NSFWClassifier(unsafe_threshold=float(threshold),
                                             model_path=model_path,
                                             providers=providers,
                                             nsfw_class=nsfw_class)
        self.nsfw_worker = NSFWWorker(self.config['last_folder'],nsfw_classifier)
        self.nsfw_worker.finished.connect(self.handle_nsfw)
        self.nsfw_worker.start()
    def handle_nsfw(self,results):
        """处理NSFW分类结果"""
        self.face_db.update(results)
        print("NSFW",results)
    def setting(self):
        """设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("设置")
        dialog.setFixedSize(500, 1000)
        
        # 创建布局
        layout = QVBoxLayout(dialog)
        layout.setAlignment(Qt.AlignTop)
        def add_line(layout,text):
            #添加分界线
            line_layout = QHBoxLayout()
            line_layout.setContentsMargins(0, 0, 0, 0)
            line_layout.setSpacing(0)
            h_line = QFrame()
            h_line.setFrameShape(QFrame.HLine)
            h_line.setFrameShadow(QFrame.Sunken)
            h_line.setStyleSheet("background-color: #3d3d3d;color: #ffffff;padding: 0;margin: 0;")
            line_label = QLabel(text)
            line_label.setStyleSheet("color: #ffffff; border: none; background-color: transparent;padding: 0;margin: 0;")
            line_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            line_layout.addWidget(line_label)
            line_layout.addWidget(h_line)
            layout.addLayout(line_layout)
        register_layout = QVBoxLayout()
        add_line(register_layout,"配置人脸识别模型")
        # 人脸识别阈值设置
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("人脸识别阈值:")
        threshold_input = QLineEdit()
        threshold_input.setText(str(get_config(self.config,'face_recognition','threshold',0.5)))
        threshold_input.setPlaceholderText("0.1-1.0之间")
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_input)
        register_layout.addLayout(threshold_layout)
        
        # 是否备份数据库
        backup_layout = QHBoxLayout()
        backup_label = QLabel("备份数据库:")
        backup_checkbox = QCheckBox()
        backup_checkbox.setChecked(get_config(self.config,'face_recognition','backup_db',False))
        backup_layout.addWidget(backup_label)
        backup_layout.addWidget(backup_checkbox)
        # backup_label和backup_checkbox保持对齐
        backup_layout.setAlignment(Qt.AlignLeft)
        register_layout.addLayout(backup_layout)
        
        # 数据库路径设置

        db_layout = QHBoxLayout()
        db_label = QLabel("人脸库路径:")
        db_input = QLineEdit()
        db_input.setText(self.face_organizer.faces_db_path)
        db_browse = QPushButton("浏览")
        db_browse.clicked.connect(lambda: browse_db())
        # db_layout.setContentsMargins(0, 0, 0, 0)
        db_layout.setSpacing(0)
        db_layout.addWidget(db_label)
        db_layout.addWidget(db_input)
        db_layout.addWidget(db_browse)
        register_layout.addLayout(db_layout)
        layout.addLayout(register_layout)
        clean_layout = QVBoxLayout()
        add_line(clean_layout,"清理人脸库")
        # 输出路径
        clean_db_layout = QHBoxLayout()
        clean_db_label = QLabel("清理后输出路径:")
        clean_db_output = QLineEdit()
        clean_db_output.setText(get_config(self.config,'clean_db','output_path',self.face_organizer.faces_db_path))
        clean_db_layout.addWidget(clean_db_label)
        clean_db_layout.addWidget(clean_db_output)  
        clean_db_browse = QPushButton("浏览")
        clean_db_browse.clicked.connect(lambda: browse_db())
        clean_db_layout.addWidget(clean_db_browse)
        clean_layout.addLayout(clean_db_layout)
        method_layout = QHBoxLayout()
        method_label = QLabel("清理方法:")
        method_input = QComboBox()
        method_input.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #000000;
                color: white;
                selection-background-color: #3d3d3d;
                selection-color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QComboBox:hover {
                background-color: #3d3d3d;
            }
        """)
        method_input.addItems(["dbscan", "kmeans", "mean"])
        method_input.setCurrentText(get_config(self.config,'clean_db','method','dbscan'))
        method_layout.addWidget(method_label)
        method_layout.addWidget(method_input)
        clean_layout.addLayout(method_layout)
        eps_layout = QHBoxLayout()
        eps_label = QLabel("eps:")
        eps_input = QLineEdit()
        eps_input.setText(str(get_config(self.config,'clean_db','eps',0.3)))
        eps_input.setFixedSize(100, 40)
        eps_layout.addWidget(eps_label)
        eps_layout.addWidget(eps_input)
        clean_layout.addLayout(eps_layout)
        min_samples_layout = QHBoxLayout()
        min_samples_label = QLabel("min_samples:")
        min_samples_input = QLineEdit()
        min_samples_input.setText(str(get_config(self.config,'clean_db','min_samples',1)))
        # 让其固定大小
        min_samples_input.setFixedSize(100, 40)
        min_samples_layout.addWidget(min_samples_label)
        min_samples_layout.addWidget(min_samples_input)
        clean_layout.addLayout(min_samples_layout)
        clean_backup_layout = QHBoxLayout()
        clean_backup_label = QLabel("备份:")
        clean_backup_checkbox = QCheckBox()
        clean_backup_checkbox.setChecked(get_config(self.config,'clean_db','backup',False))
        clean_backup_layout.addWidget(clean_backup_label)
        clean_backup_layout.addWidget(clean_backup_checkbox)
        clean_backup_layout.setAlignment(Qt.AlignLeft)
        clean_layout.addLayout(clean_backup_layout)

        clean_btn = QPushButton("清理")
        # clean_btn.setStyleSheet("background-color: #2d2d2d;color: #cccccc;border: none;padding: 8px 16px;border-radius: 4px;")
        # clean_btn.setFixedSize(100, 30)
        clean_btn.clicked.connect(lambda: self.clean_face_db(db_input.text(),
                                                            clean_db_output.text(),
                                                            method_input.currentText(),
                                                            eps_input.text(),
                                                            min_samples_input.text(),
                                                            clean_backup_checkbox.isChecked()))
        clean_layout.addWidget(clean_btn)
        layout.addLayout(clean_layout)

        # 添加人脸聚类
        add_face_clustering_layout = QVBoxLayout()
        add_line(add_face_clustering_layout,"未知人脸聚类")
        face_clustering_min_samples_layout = QHBoxLayout()
        face_clustering_min_samples_label = QLabel("最小样本数:")
        face_clustering_min_samples_input = QLineEdit()
        face_clustering_min_samples_input.setText(str(get_config(self.config,'face_clustering','min_samples',1)))
        face_clustering_min_samples_layout.addWidget(face_clustering_min_samples_label)
        face_clustering_min_samples_layout.addWidget(face_clustering_min_samples_input)
        face_clustering_eps_layout = QHBoxLayout()
        face_clustering_eps_label = QLabel("eps:")
        face_clustering_eps_input = QLineEdit()
        face_clustering_eps_input.setText(str(get_config(self.config,'face_clustering','eps',0.3)))
        face_clustering_eps_layout.addWidget(face_clustering_eps_label)
        face_clustering_eps_layout.addWidget(face_clustering_eps_input)
        face_clustering_min_samples_layout.addLayout(face_clustering_eps_layout)
        add_face_clustering_btn = QPushButton("未知人脸聚类")
        add_face_clustering_btn.clicked.connect(lambda: self.add_face_clustering(face_clustering_min_samples_input.text(),face_clustering_eps_input.text()))
        add_face_clustering_layout.addLayout(face_clustering_min_samples_layout)
        add_face_clustering_layout.addWidget(add_face_clustering_btn)
        # add_face_clustering_layout.addWidget(add_face_clustering_btn)
        add_nsfw_layout = QVBoxLayout()
        add_line(add_nsfw_layout,"NSFW")
        nsfw_threshold_layout = QHBoxLayout()
        nsfw_threshold_label = QLabel("NSFW阈值:")
        nsfw_threshold_input = QLineEdit()
        nsfw_threshold_input.setText(str(get_config(self.config,'nsfw_classifier','unsafe_threshold',0.6)))
        nsfw_threshold_layout.addWidget(nsfw_threshold_label)
        nsfw_threshold_layout.addWidget(nsfw_threshold_input)
        add_nsfw_layout.addLayout(nsfw_threshold_layout)
        nsfw_model_path_layout = QHBoxLayout()
        nsfw_model_path_label = QLabel("模型路径:")
        nsfw_model_path_input = QLineEdit()
        nsfw_model_path_browse = QPushButton("浏览")
        nsfw_model_path_browse.clicked.connect(lambda: browse_db())
        nsfw_model_path_input.setText(get_config(self.config,'nsfw_classifier','model_path','D:/project/小程序/model/640m.onnx'))
        nsfw_model_path_layout.addWidget(nsfw_model_path_label)
        nsfw_model_path_layout.addWidget(nsfw_model_path_input)
        nsfw_model_path_layout.addWidget(nsfw_model_path_browse)
        nsfw_providers_layout = QHBoxLayout()
        nsfw_providers_label = QLabel("提供者:")
        nsfw_providers_input = QComboBox()
        nsfw_providers_input.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #000000;
                color: white;
                selection-background-color: #3d3d3d;
                selection-color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QComboBox:hover {
                background-color: #3d3d3d;
            }
        """)
        nsfw_providers_input.addItems(["CPU", "CUDA"])
        nsfw_providers_input.setCurrentText(get_config(self.config,'nsfw_classifier','providers','CPUExecutionProvider'))
        nsfw_providers_layout.addWidget(nsfw_providers_label)
        nsfw_providers_layout.addWidget(nsfw_providers_input)
        add_nsfw_layout.addLayout(nsfw_providers_layout)
        add_nsfw_layout.addLayout(nsfw_model_path_layout)
        nsfw_button = QPushButton("NSFW")
        nsfw_button.clicked.connect(lambda: self.nsfw_classify(nsfw_model_path_input.text(),
                                                              nsfw_providers_input.currentText(),
                                                              nsfw_threshold_input.text(),
                                                              get_config(self.config,'nsfw_classifier','nsfw_class',default_nsfw_class)))
        add_nsfw_layout.addWidget(nsfw_button)


        layout.addLayout(add_face_clustering_layout)
        layout.addLayout(add_nsfw_layout)

        # 按钮
        button_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        cancel_btn = QPushButton("取消")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)
        # 设置样式
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                padding: 5px;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #cccccc;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QCheckBox {
                color: #cccccc;
            }
        """)
        
        def browse_db():
            path, _ = QFileDialog.getSaveFileName(
                dialog,
                "选择数据库文件",
                "",
                "PKL文件 (*.pkl)"
            )
            if path:
                db_input.setText(path)
        


        
        def save_settings():
            try:
                # 验证阈值
                threshold = float(threshold_input.text())
                if not 0.1 <= threshold <= 1.0:
                    raise ValueError("阈值必须在0.1到1.0之间")
                
                # 更新配置
                self.config['face_recognition'] = {
                    'threshold': threshold,
                    'backup_db': backup_checkbox.isChecked(),
                    'db_path': db_input.text(),
                    'update_db': True,  # 添加update_db属性
                }
                self.config['clean_db'] = {  # 添加清理数据库相关配置
                        'output_path': clean_db_output.text(),
                        'method': method_input.currentText(),
                        'eps': float(eps_input.text()),
                        'min_samples': int(min_samples_input.text()) if min_samples_input.text() else 2,
                        'backup': clean_backup_checkbox.isChecked()
                    }
                self.config['face_clustering'] = {
                    'min_samples': int(face_clustering_min_samples_input.text()) if face_clustering_min_samples_input.text() else 2,
                    'eps': float(face_clustering_eps_input.text()) if face_clustering_eps_input.text() else 0.3
                }
                self.config['nsfw_classifier'] = {
                    'unsafe_threshold': float(nsfw_threshold_input.text()) if nsfw_threshold_input.text() else 0.6,
                    'model_path': nsfw_model_path_input.text(),
                    'providers': nsfw_providers_input.currentText()
                }
                if "nsfw_class" not in self.config['nsfw_classifier']:
                    self.config['nsfw_classifier']['nsfw_class'] = default_nsfw_class
                # 保存配置
                self.save_config()
                
                # 更新人脸识别器设置
                self.face_organizer.threshold = threshold
                self.face_organizer.backup_db = backup_checkbox.isChecked()
                self.face_organizer.faces_db_path = db_input.text()
                
                QMessageBox.information(dialog, "成功", "设置已保存")
                dialog.accept()
                
            except ValueError as e:
                QMessageBox.warning(dialog, "错误", str(e))
            except Exception as e:
                QMessageBox.critical(dialog, "错误", f"保存设置失败: {str(e)}")
        
        # 连接信号
        # db_browse.clicked.connect(browse_db)
        save_btn.clicked.connect(save_settings)
        cancel_btn.clicked.connect(dialog.reject)
        
        # 显示对话框
        dialog.exec_()
    def add_folder(self):
        """添加文件夹或文件"""
        # 创建文件对话框
        folder_path= QFileDialog.getExistingDirectory(self,
            "选择文件夹")
        if folder_path == "":
            return
        # 显示进度条
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        
        # 创建工作线程实例
        self.process_worker = ProcessWorker(
            folder_path,
            self.face_organizer,
            self.config['last_folder'],
            "folder",
        )
        
        # 连接信号
        self.process_worker.progress.connect(self.progress_bar.setValue)
        self.process_worker.finished.connect(self.on_process_complete)
        self.process_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self.process_worker.register_request.connect(self.handle_unknown_face)  # 处理未识别人脸
        
        # 启动线程
        self.process_worker.start()
        
        # 禁用相关按钮
        self.open_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.register_btn.setEnabled(False)

    def add_file(self):
        """添加文件"""
        # 创建文件对话框
        file_path, _ = QFileDialog.getOpenFileName(self,
            "选择文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png)")
        print("file_path",file_path)
        if file_path == "":
            print("取消添加文件")
            return
        # 显示进度条
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        
        # 创建工作线程实例
        self.process_worker = ProcessWorker(
            file_path,
            self.face_organizer,
            self.config['last_folder'],
            "file",
        )
        
        # 连接信号
        self.process_worker.progress.connect(self.progress_bar.setValue)
        self.process_worker.finished.connect(self.on_process_complete)
        self.process_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self.process_worker.register_request.connect(self.handle_unknown_face)  # 处理未识别人脸
        
        # 启动线程
        self.process_worker.start()
        
        # 禁用相关按钮
        self.open_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.register_btn.setEnabled(False)

    def on_process_complete(self,total_register_person):
        """处理完成后的回调"""
        # 恢复按钮状态
        self.open_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.register_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.hide()
        
        # 刷新界面
        self.process_folder(self.config['last_folder'])
        
        # 显示完成消息,在一定时间后自动关闭
        str_msg = "添加完成\n"
        for person,count in total_register_person.items():
            str_msg += f"{person}：共 {count} 张照片\n"
        QMessageBox.information(self, "完成", str_msg)

    def resizeEvent(self, event):
        """窗口大小变化时重新排列缩略图"""
        super().resizeEvent(event)
        if hasattr(self, 'update_preview_grid'):
            self.show_preview(self.current_person, self.current_images)
    def on_scroll_area_resize(self, event):
        """处理滚动区域大小变化事件"""
        # 取消之前的定时器
        self.resize_timer.stop()
        
        # 计算新的布局信息
        new_layout = self.calculate_layout()
        
        # 减小阈值，使动画更敏感（从5改为3）
        if (new_layout['cols'] != self.last_layout_info['cols'] or 
            abs(new_layout['card_width'] - self.last_layout_info['card_width']) > 3):
            if hasattr(self, 'face_db'):
                self.resize_timer.start(self.resize_debounce_time)
        
        # 调用原始的 resizeEvent
        super(QScrollArea, self.scroll_area).resizeEvent(event)
    
    def calculate_layout(self):
        """计算布局信息"""
        available_width = self.scroll_area.viewport().width() - 40
        min_card_width = 200
        spacing = 20  # 固定间距
        
        # 计算可以放置的列数
        cols = max(1, (available_width + spacing) // (min_card_width + spacing))
        # 如果当前是风格界面，则设置为5列
        if self.stacked_layout.currentWidget() != self.cards_page:
            cols = 5
        # 计算实际的卡片宽度
        card_width = (available_width - (cols - 1) * spacing) // cols
        return {
            'width': available_width,
            'cols': cols,
            'card_width': card_width,
            'spacing': spacing
        }
    
    def update_person_grid(self):
        """更新人物网格"""
        # 计算布局信息
        layout_info = self.calculate_layout()
        
        current_widget = self.stacked_layout.currentWidget()
        
        # 确定当前是哪个页面
        is_first_page = (current_widget == self.cards_page)
        
        # 获取对应的网格容器
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(layout_info['spacing'])
        grid_layout.setContentsMargins(20, 20, 20, 20)
        # 找出重复文件
        
        # 预先筛选需要显示的人物
        people_to_show = []
        # print("self.style_name",self.style_name)
        for person_name, photos in self.face_db.items():
            if not photos:
                continue
            if is_first_page:
                if person_name not in self.style_name:
                    people_to_show.append((person_name, len(photos)))
            else:
                if person_name.split("_")[0] in self.style_name or person_name in self.style_name:
                    people_to_show.append((person_name, len(photos)))
        
        # 按照照片数量排序
        people_to_show.sort(key=lambda x: x[1], reverse=True)


        # 批量创建卡片
        row = col = 0
        if current_widget == self.cards_page:
            for person_name, _ in people_to_show:
                person_card = self.create_person_card(person_name, layout_info['card_width'])
                grid_layout.addWidget(person_card, row, col)

                col += 1
                if col >= layout_info['cols']:
                    col = 0
                    row += 1
        else:
            for person_name, _ in people_to_show:
                person_card = self.create_person_card(person_name, layout_info['card_width'], is_other=True)
                grid_layout.addWidget(person_card, row, col)
                col += 1
                if col >= layout_info['cols']:
                    col = 0
                    row += 1
        

        # 设置网格对齐和拉伸
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        grid_widget.setMinimumWidth(layout_info['width'])
        
        # 更新对应的滚动区域
        scroll_area = self.scroll_area if is_first_page else self.scroll_area_2
        if scroll_area.widget():
            scroll_area.widget().deleteLater()
        scroll_area.setWidget(grid_widget)
        
        # 保存当前布局信息用于动画
        self.last_layout_info = layout_info
    
    def add_card_animation(self, card, old_width, new_width):
        """为卡片添加大小变化动画"""
        # 创建动画对象
        animation = QPropertyAnimation(card, b"minimumSize")
        animation.setDuration(100)  # 更短的动画时间
        animation.setEasingCurve(QEasingCurve.OutQuad)  # 使用平滑的缓动曲线
        
        # 设置起始和结束状态
        animation.setStartValue(QSize(old_width, old_width))
        animation.setEndValue(QSize(new_width, new_width))
        
        # 同时设置最大尺寸
        card.setMaximumSize(new_width, new_width)
        
        # 启动动画
        animation.start(QPropertyAnimation.DeleteWhenStopped)
    #创建风格图片卡片的显示

    def create_person_card(self, person_name, width, is_other=False):
        """创建人物卡片
        
        Args:
            person_name: 人物名称
            width: 卡片宽度
            is_other: 是否是特殊卡片（未识别/无人脸）
        """
        # 创建卡片容器
        card = QWidget()
        card.setFixedSize(width, width)
        layout = QVBoxLayout(card)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建头像容器
        avatar_container = QWidget()
        avatar_container.setFixedSize(width, width)
        avatar_layout = QVBoxLayout(avatar_container)
        avatar_layout.setContentsMargins(0, 0, 0, 0)
        avatar_layout.setSpacing(0)
        
        # 设置头像
        avatar_label = QLabel()
        avatar_label.setFixedSize(width, width)
        avatar_label.setAlignment(Qt.AlignCenter)
        
        # 获取头像图片
        if person_name in self.face_db and not self.face_db[person_name] is None:
            avatar_path = self.face_db[person_name][0]
            try:
                pixmap = QPixmap(avatar_path)
                if not pixmap.isNull():
                    # 使用动态宽度进行缩放
                    scaled_pixmap = pixmap.scaled(width + 20, width + 20, 
                                             Qt.KeepAspectRatioByExpanding, 
                                             Qt.SmoothTransformation)
                    x = (scaled_pixmap.width() - width) // 2
                    y = (scaled_pixmap.height() - width) // 2
                    cropped_pixmap = scaled_pixmap.copy(x, y, width, width)
                    avatar_label.setPixmap(cropped_pixmap)
            except:
                avatar_label.setText("加载失败")
                avatar_label.setStyleSheet("color: white;")
        else:
            avatar_label.setText("无照片")
            avatar_label.setStyleSheet("color: white;")
        
        # 创建信息标签
        info_label = QLabel()
        info_label.setFixedSize(width, 50)
        info_label.setAlignment(Qt.AlignCenter)
        
        # 设置信息文本
        count = len(self.face_db.get(person_name, []))
        info_text = f"{person_name}\n{count}张照片"
        info_label.setText(info_text)
        info_label.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                padding: 5px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        # 将组件添加到布局中
        avatar_layout.addWidget(avatar_label)
        avatar_layout.addStretch()
        avatar_layout.addWidget(info_label)
        
        # 设置容器样式
        avatar_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
        """)
        
        layout.addWidget(avatar_container)
        
        # 设置卡片整体样式
        card.setStyleSheet("background: transparent;")
        
        # 添加悬停效果
        def enterEvent(event):
            avatar_container.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    border-radius: 8px;
                }
            """)
            info_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background: transparent;
                    padding: 5px;
                    font-weight: bold;
                    font-size: 14px;
                }
            """)
        
        def leaveEvent(event):
            avatar_container.setStyleSheet("""
                QWidget {
                    background-color: white;
                    border-radius: 8px;
                }
            """)
            info_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background: transparent;
                    padding: 5px;
                    font-weight: bold;
                    font-size: 14px;
                }
            """)
        
        card.enterEvent = enterEvent
        card.leaveEvent = leaveEvent
        
        # 添加点击事件
        def mouseReleaseEvent(event):
            if event.button() == Qt.LeftButton:
                # 获取该人物的所有照片路径
                image_paths = self.face_db.get(person_name, [])
                if image_paths:
                    # 显示预览界面
                    self.show_preview(person_name, image_paths)
        
        card.mouseReleaseEvent = mouseReleaseEvent
        
        # 修改鼠标样式
        card.setCursor(Qt.PointingHandCursor)
        
        return card

    def open_folder(self,folder=None):
        """打开文件夹"""
        # 使用上次的路径作为起始目录
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            # 保存当前路径
            self.config['last_folder'] = folder
            
            # 处理文件夹
            self.process_folder(folder)

    def process_folder(self, folder = None):
        """处理文件夹中的照片"""
        # print(f"处理文件夹 {folder}")
        if folder is None:
            return
        
        # 显示进度条
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        self.worker = PhotoProcessWorker(folder, self.face_organizer)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_complete)
        self.worker.start()
        
        # 禁用按钮，避免重复处理
        self.open_btn.setEnabled(False)
        self.register_btn.setEnabled(False)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def process_complete(self, results):
        """处理完成后更新界面"""
        self.face_db.update(results['face_db'])
        # print("face_db",self.face_db)
        self.update_person_grid()
        
        # 重新启用按钮
        self.open_btn.setEnabled(True)
        self.register_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.hide()
    def delete_photo(self, image_path):
        """删除照片"""
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除这张照片吗？"
        )
    
        if reply == QMessageBox.Yes:
            try:
                # 删除文件
                os.remove(image_path)
            
                # 从当前预览列表中移除
                if image_path in self.current_preview_images:
                    self.current_preview_images.remove(image_path)
            
                # 更新界面
                self.show_preview(self.title, self.current_preview_images)
            
                QMessageBox.information(self, "成功", "照片已删除")
            
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")
    def register_face(self,image_dir=None):
        """注册新人脸"""
        # 选择图片
        if self.stacked_layout.currentWidget() == self.photo_view_page:
            if self.image_paths is not []:
                image_path = self.image_paths[self.current_index]
        elif image_dir is not None :
            image_path = image_dir
        else:
            image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择照片",
            "",
            "图片文件 (*.jpg *.jpeg *.png)"
            )
        
        if not image_path:
            return
        
        # 显示预览窗口
        preview_dialog = QDialog(self)
        preview_dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)  # 保留Dialog标志
        preview_dialog.setWindowTitle("人脸注册")
        
        # preview_dialog.setFixedSize(500, 600)
        
        layout = QVBoxLayout(preview_dialog)
        
        # 显示图片
        image_label = QLabel()
        
        pixmap = QPixmap(image_path)
        # 提取人脸
        faces, img= self.face_organizer.detect_faces(normalize_path(image_path))
        if faces is None:
            QMessageBox.critical(preview_dialog, "错误", "无法检测到人脸")
            return
        # 将每个人画到图片上
        i = 0      
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for face in faces:
            bbox = face.bbox.astype(int)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, str(i), (bbox[0], bbox[3] + 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            i += 1
        pixmap = cv2_to_qpixmap(img)
        scaled_pixmap = pixmap.scaled(
            400, 400,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)
        #下拉选项框
        combo_box_layout = QHBoxLayout()
        combo_box_label = QLabel("选择人脸")
        combo_box = QComboBox()
        for k in range(len(faces)):
            combo_box.addItem(str(k))
        combo_box.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #000000;
                color: white;
                selection-background-color: #3d3d3d;
                selection-color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QComboBox:hover {
                background-color: #3d3d3d;
            }
        """)
        combo_box_layout.addWidget(combo_box_label)
        combo_box_layout.addWidget(combo_box)
        layout.addLayout(combo_box_layout)
        # 输入人名
        name_layout = QHBoxLayout()
        name_label = QLabel("请输入人名:")
        name_input = QLineEdit()
        name_input.setPlaceholderText("请输入人名")
        name_input.setStyleSheet("background-color: #ffffff; color: black;")
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        # 添加一个复选框
        check_box_layout = QHBoxLayout()
        check_box_label = QLabel("是否保存照片")
        check_box = QCheckBox()
        check_box_layout.addWidget(check_box_label)
        check_box_layout.addWidget(check_box)
        layout.addLayout(check_box_layout)
        # 确认和取消按钮
        button_layout = QHBoxLayout()
        confirm_btn = KeyButton("确认")
        cancel_btn = KeyButton("取消")
        button_layout.addWidget(confirm_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # 处理确认按钮点击
        def on_confirm():
            person_name = name_input.text().strip()
            if not person_name:
                QMessageBox.warning(preview_dialog, "警告", "请输入人名")
                return
                        
            try:
                # 注册人脸
                person_name = self.face_organizer.register_face(faces[combo_box.currentIndex()], person_name)
                if person_name is None:
                    QMessageBox.critical(preview_dialog, "错误", "无法检测到人脸")
                    return
                # 更新本地人脸数据库
                # self.face_db = self.face_organizer.get_face_db()
                
                
                if check_box.isChecked():
                    # 创建人名文件夹
                    # 如果照片已经在当前文件夹下
                    if not copy_photo(person_name,image_path,self.config['last_folder']):
                        QMessageBox.critical(
                                preview_dialog,
                                "错误",
                                f"无法复制照片: {str(e)}"
                        )
                        raise Exception(f"无法复制照片: {str(e)}")
                QMessageBox.information(
                    preview_dialog,
                    "成功",
                    f"已成功注册 {person_name}"
                )
                preview_dialog.accept()
                # 恢复处理线程
                if hasattr(self, 'process_worker'):
                    self.process_worker.resume()
                
            except Exception as e:
                QMessageBox.critical(
                    preview_dialog,
                    "错误",
                    f"注册失败: {str(e)}"
                )
                # 发生错误时也要恢复线程
                if hasattr(self, 'process_worker'):
                    self.process_worker.resume()
        
        # 绑定按钮事件
        confirm_btn.clicked.connect(on_confirm)
        cancel_btn.clicked.connect(preview_dialog.reject)
        
        # 添加鼠标拖动功能
        class MouseDragFilter(QObject):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.dragging = False
                self.offset = QPoint()
            
            def eventFilter(self, obj, event):
                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.LeftButton:
                        self.dragging = True
                        self.offset = event.pos()
                elif event.type() == QEvent.MouseMove:
                    if self.dragging:
                        preview_dialog.move(preview_dialog.pos() + event.pos() - self.offset)
                elif event.type() == QEvent.MouseButtonRelease:
                    self.dragging = False
                return False
        
        # 安装事件过滤器
        drag_filter = MouseDragFilter()
        preview_dialog.installEventFilter(drag_filter)
        # 显示对话框
        preview_dialog.exec_()
    def refresh_folder(self):
        """刷新文件夹"""
        # 禁用按钮，显示加载动画
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setIcon(qta.icon('fa5s.sync', color='white', animation=qta.Spin(self.refresh_btn)))
        
        # 使用定时器延迟执行刷新操作
        timer = QTimer(self)
        timer.setSingleShot(True)
        self.remove_duplicates_worker = RemoveDuplicatesWorker(self.config['last_folder'],0.9)
        self.remove_duplicates_worker.finished.connect(self.finish_refresh)
        self.remove_duplicates_worker.start()
        # 使用lambda正确传递self参数
        timer.timeout.connect(lambda: self.finish_refresh())
        timer.start(100)

    def finish_refresh(self):
        """完成刷新"""
        try:
            self.process_folder(self.config['last_folder'])
            self.update_person_grid()
            current_widget = self.stacked_layout.currentWidget()
            if current_widget != self.cards_page:
                if hasattr(self, 'title'):
                    image_paths = self.face_db.get(self.title, [])
                    if image_paths:
                        # 显示预览界面
                        self.show_preview(self.title, image_paths)
        finally:
            # 恢复按钮状态
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setIcon(qta.icon('fa5s.sync', color='#c8c8c8'))
    def show_preview(self, person_name, image_paths):
        """显示照片预览"""
        # 保存当前人名，用于刷新
        self.title = person_name  # 添加这行
        
        # 清除原有内容
        self.clear_layout(self.preview_layout)
        self.image_paths = image_paths
        # 创建顶部工具栏
        toolbar = QHBoxLayout()
        
        # 返回按钮
        back_btn = KeyButton("返回")
        # 正确的返  回cards_page或者cards_page_2
        if self.stacked_layout.currentWidget() == self.cards_page:
            back_btn.clicked.connect(lambda: self.stacked_layout.setCurrentWidget(self.cards_page))
        elif self.stacked_layout.currentWidget() == self.cards_page_2:
            back_btn.clicked.connect(lambda: self.stacked_layout.setCurrentWidget(self.cards_page_2))
        toolbar.addWidget(back_btn)
        toolbar.addStretch()
        
        # 添加标题
        if person_name is not None:
            title_label = QLabel(person_name)
            title_label.setStyleSheet("color: white; font-size: 16px;")
            toolbar.addWidget(title_label)
        
        self.preview_layout.addLayout(toolbar)
        
        # 创建网格容器
        grid_widget = QWidget()
        grid_widget.setStyleSheet("background: #1e1e1e;")  # 使用相同的深色背景
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # 保存图片路径用于重排
        self.current_preview_images = image_paths
        
        # 计算缩放后的尺寸和列数
        base_size = 200
        scale = 1.2
        thumbnail_size = int(base_size * scale)
        scroll_width = self.width()  # 使用完整宽度
        columns = max(1, scroll_width // thumbnail_size)
        
        # 添加缩略图
        for i, path in enumerate(image_paths):
            row = i // columns
            col = i % columns
            
            # 创建缩略图容器
            thumb_container = QWidget()
            thumb_container.setFixedSize(thumbnail_size, thumbnail_size)
            thumb_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            # 创建缩略图标签
            thumb_label = QLabel()
            thumb_label.setFixedSize(thumbnail_size, thumbnail_size)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setText("加载中...")
            
            # 设置样式
            container_style = """
                QWidget {
                    background: #1e1e1e;  /* 深色背景 */
                    border: none;
                    margin: 0;
                    padding: 0;
                }
                QWidget:hover {
                    background: #2d2d2d;  /* 悬停时稍微亮一点 */
                }
            """
            thumb_container.setStyleSheet(container_style)
            
            label_style = """
                QLabel {
                    border: none;
                    margin: 0;
                    padding: 0;
                    background: transparent;
                }
            """
            thumb_label.setStyleSheet(label_style)
            
            # 添加点击事件
            def create_click_handler(index):
                return lambda: self.show_photo(image_paths, index)
                
            def show_context_menu(event, image_paths, index):
                if event.button() == Qt.RightButton:
                    menu = QMenu()
                    menu.setStyleSheet("background-color: #1e1e1e; color: white;")
                    # 加人脸注册选项
                    register_action = menu.addAction("人脸注册")
                    register_action.triggered.connect(lambda: self.register_face(image_paths[index]))
                    
                    # 添加删除选项
                    delete_action = menu.addAction("删除照片")
                    delete_action.triggered.connect(lambda: self.delete_photo(image_paths[index]))
                    open_action = menu.addAction("打开文件夹")
                    open_action.triggered.connect(lambda: QProcess.startDetached('explorer.exe', ['/select,', image_paths[index]]))
                    # 在鼠标位置显示菜单

                    menu.exec_(event.globalPos())
            
            # 绑定右键菜单事件
            thumb_label.mousePressEvent = lambda e, i=i: (
                create_click_handler(i)() if e.button() == Qt.LeftButton 
                else show_context_menu(e, image_paths, i)
            )
            thumb_label.setCursor(Qt.PointingHandCursor)
            
            # 异步加载缩略图
            def load_thumbnail(label, img_path):
                if label and not sip.isdeleted(label):
                    try:
                        reader = QImageReader(img_path)
                        reader.setAutoTransform(True)
                        reader.setScaledSize(QSize(thumbnail_size, thumbnail_size))
                        
                        if reader.canRead():
                            image = reader.read()
                            if not image.isNull():
                                pixmap = QPixmap.fromImage(image)
                                if not sip.isdeleted(label):
                                    label.setPixmap(pixmap)
                    except Exception as e:
                        print(f"加载缩略图失败 {img_path}: {str(e)}")
            
            # 使用更大的时间间隔
            QTimer.singleShot(50 * i, lambda l=thumb_label, p=path: load_thumbnail(l, p))
            
            # 添加到容器
            layout = QVBoxLayout(thumb_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(thumb_label)
            
            # 添加到网格
            grid_layout.addWidget(thumb_container, row, col, Qt.AlignTop)
        
        # 设置滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        scroll_area.setWidget(grid_widget)
        self.preview_layout.addWidget(scroll_area)
        
        # 切换到预览页面
        self.stacked_layout.setCurrentWidget(self.preview_page)
    def load_thumbnail(self, path, label, size):
        """在线程中加载缩略图"""
        try:
            image = QImage(path)
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(size, size, 
                                            Qt.KeepAspectRatioByExpanding,
                                            Qt.SmoothTransformation)
                if scaled_pixmap.width() > size or scaled_pixmap.height() > size:
                    x = (scaled_pixmap.width() - size) // 2
                    y = (scaled_pixmap.height() - size) // 2
                    scaled_pixmap = scaled_pixmap.copy(x, y, size, size)
                label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"加载缩略图失败 {path}: {str(e)}")
        
    
        """更新预览界面的网格布局"""
        # 获取当前预览的图片路径列表
        if not hasattr(self, 'title'):
            return
            
        image_paths = self.face_db.get(self.title, [])
        if not image_paths:
            return
            
        # 计算布局信息
        layout_info = self.calculate_layout()
        
        # 创建网格容器
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(layout_info['spacing'])
        grid_layout.setContentsMargins(20, 20, 20, 20)
        
        # 批量创建预览缩略图
        row = col = 0
        for i, image_path in enumerate(image_paths):
            # 创建缩略图容器
            thumb_container = QWidget()
            thumb_container.setFixedSize(layout_info['card_width'], layout_info['card_width'])
            thumb_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            # 创建缩略图标签
            thumb_label = QLabel()
            thumb_label.setFixedSize(layout_info['card_width'], layout_info['card_width'])
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setText("加载中...")
            
            # 设置样式
            container_style = """
                QWidget {
                    background: #1e1e1e;
                    border: none;
                    margin: 0;
                    padding: 0;
                }
                QWidget:hover {
                    background: #2d2d2d;
                }
            """
            thumb_container.setStyleSheet(container_style)
            
            label_style = """
                QLabel {
                    border: none;
                    margin: 0;
                    padding: 0;
                    background: transparent;
                }
            """
            thumb_label.setStyleSheet(label_style)
            
            # 添加点击事件
            def create_click_handler(index):
                return lambda: self.show_photo(image_paths, index)
                
            def show_context_menu(event, image_paths, index):
                if event.button() == Qt.RightButton:
                    menu = QMenu()
                    menu.setStyleSheet("background-color: #1e1e1e; color: white;")
                    
                    # 添加菜单选项
                    register_action = menu.addAction("人脸注册")
                    register_action.triggered.connect(lambda: self.register_face(image_paths[index]))
                    
                    delete_action = menu.addAction("删除照片")
                    delete_action.triggered.connect(lambda: self.delete_photo(image_paths[index]))
                    
                    open_action = menu.addAction("打开文件夹")
                    open_action.triggered.connect(lambda: QProcess.startDetached('explorer.exe', ['/select,', image_paths[index]]))
                    
                    menu.exec_(event.globalPos())
            
            # 绑定事件
            thumb_label.mousePressEvent = lambda e, i=i: (
                create_click_handler(i)() if e.button() == Qt.LeftButton 
                else show_context_menu(e, image_paths, i)
            )
            thumb_label.setCursor(Qt.PointingHandCursor)
            
            # 异步加载缩略图
            def load_thumbnail(label, img_path):
                if label and not sip.isdeleted(label):
                    try:
                        reader = QImageReader(img_path)
                        reader.setAutoTransform(True)
                        reader.setScaledSize(QSize(layout_info['card_width'], layout_info['card_width']))
                        
                        if reader.canRead():
                            image = reader.read()
                            if not image.isNull():
                                pixmap = QPixmap.fromImage(image)
                                if not sip.isdeleted(label):
                                    label.setPixmap(pixmap)
                    except Exception as e:
                        print(f"加载缩略图失败 {img_path}: {str(e)}")
            
            # 使用定时器延迟加载
            QTimer.singleShot(50 * i, lambda l=thumb_label, p=image_path: load_thumbnail(l, p))
            
            # 添加到容器
            layout = QVBoxLayout(thumb_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(thumb_label)
            
            # 添加到网格
            grid_layout.addWidget(thumb_container, row, col)
            
            # 更新行列位置
            col += 1
            if col >= layout_info['cols']:
                col = 0
                row += 1
        
        # 设置滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        # scroll_area.setWidget(grid_widget)
        
        # 清除原有内容并添加新的滚动区域
        self.clear_layout(self.preview_layout)
        self.preview_layout.addWidget(scroll_area)
    def show_photo(self, image_paths, current_index):
        """显示单张照片查看界面"""
        # 如果已经存在photo_view_page，先移除它
        if hasattr(self, 'photo_view_page') and self.photo_view_page is not None:
            self.stacked_layout.removeWidget(self.photo_view_page)
            self.photo_view_page.deleteLater()
        
        # 创建新的照片查看页面
        self.photo_view_page = QWidget()
        self.current_index = current_index
        main_layout = QVBoxLayout(self.photo_view_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建工具栏容器
        toolbar_widget = QWidget()
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(10, 10, 10, 10)
        
        # 返回按钮
        back_btn = KeyButton("返回")
        back_btn.clicked.connect(lambda: (
            self.stacked_layout.setCurrentWidget(self.preview_page),
            self.stacked_layout.removeWidget(self.photo_view_page),
            self.restore_events(),
            self.refresh_btn.setVisible(True)
        ))
        
        # 添加照片计数
        count_label = QLabel(f"{current_index + 1} / {len(image_paths)}")
        count_label.setStyleSheet("color: white;")
        
        # 将组件添加到工具栏
        toolbar.addWidget(back_btn)
        toolbar.addStretch()
        toolbar.addWidget(count_label)
        toolbar.addStretch()
        
        # 将工具栏容器添加到主布局
        main_layout.addWidget(toolbar_widget)
        
        # 创建照片显示区域
        photo_container = QWidget()
        photo_layout = QHBoxLayout(photo_container)
        photo_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setAlignment(Qt.AlignCenter)  # 设置居中对齐
        photo_layout.addWidget(scroll_area)
        
        # 创建容器widget来包含照片标签
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignCenter)  # 设置居中对齐
        
        # 照片标签
        photo_label = QLabel()
        photo_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(photo_label)
        
        # 将容器设置为滚动区域的widget
        scroll_area.setWidget(container)
        self.refresh_btn.setVisible(False)
        class DragManager:
            def __init__(self):
                self.dragging = False
                self.last_pos = None
                self.zoom_level = 1.0  # 添加缩放级别属性
                
            def set_zoom(self, zoom):
                """设置缩放级别"""
                self.zoom_level = max(0.1, min(5.0, zoom))  # 限制缩放范围在0.1到5.0之间
        
        drag_manager = DragManager()
        
        def mousePressEvent(event):
            if event.button() == Qt.LeftButton:
                drag_manager.dragging = True
                drag_manager.last_pos = event.pos()
                photo_label.setCursor(Qt.ClosedHandCursor)
        
        def mouseReleaseEvent(event):
            if event.button() == Qt.LeftButton:
                drag_manager.dragging = False
                photo_label.setCursor(Qt.ArrowCursor)
                # 停止拖拽，不需要惯性
        
        def mouseMoveEvent(event):
            if drag_manager.dragging and drag_manager.last_pos:
                # 计算移动距离
                delta = event.pos() - drag_manager.last_pos
                
                # 更新滚动条位置
                h_value = scroll_area.horizontalScrollBar().value() - delta.x()
                v_value = scroll_area.verticalScrollBar().value() - delta.y()
                
                # 应用边界限制
                h_value = max(0, min(h_value, scroll_area.horizontalScrollBar().maximum()))
                v_value = max(0, min(v_value, scroll_area.verticalScrollBar().maximum()))
                
                # 设置新的滚动位置
                scroll_area.horizontalScrollBar().setValue(h_value)
                scroll_area.verticalScrollBar().setValue(v_value)
                
                # 更新上一次位置
                drag_manager.last_pos = event.pos()
        
        # 绑定鼠标事件
        photo_label.mousePressEvent = mousePressEvent
        photo_label.mouseReleaseEvent = mouseReleaseEvent
        photo_label.mouseMoveEvent = mouseMoveEvent
        photo_label.setMouseTracking(True)
        
        # 添加滚轮事件
        def wheelEvent(event):
            if event.modifiers() & Qt.ControlModifier:
                # 获取鼠标位置
                mouse_pos = event.pos()
                
                # 获取当前滚动条位置
                old_h = scroll_area.horizontalScrollBar().value()
                old_v = scroll_area.verticalScrollBar().value()
                
                # 计算鼠标位置相对于视口的比例
                viewport = scroll_area.viewport()
                rel_x = (mouse_pos.x() + old_h) / photo_label.width()
                rel_y = (mouse_pos.y() + old_v) / photo_label.height()
                
                # 计算新的缩放级别
                old_zoom = drag_manager.zoom_level
                if event.angleDelta().y() > 0:
                    new_zoom = old_zoom * 1.2  # 放大
                else:
                    new_zoom = old_zoom / 1.2  # 缩小
                
                # 设置新的缩放级别
                drag_manager.set_zoom(new_zoom)
                
                # 更新图片
                update_photo()
                
                # 计算新的滚动位置以保持鼠标指向的点
                new_h = int(photo_label.width() * rel_x - viewport.width() * (mouse_pos.x() / viewport.width()))
                new_v = int(photo_label.height() * rel_y - viewport.height() * (mouse_pos.y() / viewport.height()))
                
                # 设置新的滚动位置
                scroll_area.horizontalScrollBar().setValue(new_h)
                scroll_area.verticalScrollBar().setValue(new_v)
                
                event.accept()
        
        # 绑定滚轮事件
        photo_label.wheelEvent = wheelEvent
        scroll_area.wheelEvent = wheelEvent  # 也绑定到滚动区域
        
        def update_photo():
            """更新照片大小"""
            try:
                reader = QImageReader(image_paths[current_index])
                reader.setAutoTransform(True)
                
                if reader.canRead():
                    image = reader.read()
                    if not image.isNull():
                        # 获取可用空间
                        available_width = self.width() - 100
                        available_height = self.height() - 150
                        
                        # 计算缩放比例以适应窗口
                        width_ratio = available_width / image.width()
                        height_ratio = available_height / image.height()
                        scale_ratio = min(width_ratio, height_ratio)
                        
                        # 应用用户缩放
                        scale_ratio *= drag_manager.zoom_level
                        
                        # 计算新尺寸
                        new_width = int(image.width() * scale_ratio)
                        new_height = int(image.height() * scale_ratio)
                        
                        # 缩放图片
                        scaled_image = image.scaled(
                            new_width, new_height,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        
                        # 设置图片
                        pixmap = QPixmap.fromImage(scaled_image)
                        photo_label.setPixmap(pixmap)
                        
                        # 调整容器大小以适应图片
                        container.setFixedSize(pixmap.size())
                        
                        # 更新计数
                        count_label.setText(f"{current_index + 1} / {len(image_paths)}")
                        
            except Exception as e:
                print(f"加载图片失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 保存update_photo函数的引用
        self.update_photo = update_photo
        
        # 添加窗口大小改变事件处理
        def resizeEvent(event):
            if hasattr(self, 'update_photo'):
                self.update_photo()
        
        # 设置resize事件处理器
        self.photo_view_page.resizeEvent = resizeEvent
        
        # 添加前进后退按钮
        prev_btn = KeyButton("←")
        next_btn = KeyButton("→")
        
        for btn in (prev_btn, next_btn):
            btn.setFixedSize(50, 100)
            btn.hide()  # 初始状态隐藏按钮

        
        # 添加翻页功能
        def show_prev():
            nonlocal current_index
            current_index = (current_index - 1) % len(image_paths)
            update_photo()
        
        def show_next():
            nonlocal current_index
            current_index = (current_index + 1) % len(image_paths)
            update_photo()
        
        # 绑定按钮点击事件
        prev_btn.clicked.connect(show_prev)
        next_btn.clicked.connect(show_next)

        # 添加键盘事件
        def keyPressEvent(event):
            # 在照片查看页面时才处理这些按键
            # print("当前页面",self.stacked_layout.currentWidget())
            if self.stacked_layout.currentWidget() == self.photo_view_page:
                # print("在照片查看页面")
                if event.key() in (Qt.Key_Left, Qt.Key_Up):
                    # print("左键")
                    show_prev()
                    event.accept()  # 标记事件已处理
                    return  # 直接返回，不再传递事件
                elif event.key() in (Qt.Key_Right, Qt.Key_Down):
                    show_next()
                    event.accept()  # 标记事件已处理
                    return  # 直接返回，不再传递事件
                elif event.key() == Qt.Key_Escape:
                    self.stacked_layout.setCurrentWidget(self.preview_page)
                    event.accept()  # 标记事件已处理
                    return  # 直接返回，不再传递事件
            
            # 其他情况调用原始的事件处理
            if hasattr(self, 'original_keyPressEvent'):
                self.original_keyPressEvent(event)
        
        # 保存并设置事件处理器
        self.original_keyPressEvent = self.keyPressEvent
        self.keyPressEvent = keyPressEvent
        # 添加鼠标移动事件处理
        def mouseMoveEvent(event):
            # 获取按钮区域（调整区域宽度）
            print("mouseMoveEvent")
            prev_area = QRect(0, 0, 150, photo_container.height())
            next_area = QRect(photo_container.width() - 150, 0, 150, photo_container.height())
            
            # 检查鼠标位置
            pos = photo_container.mapFromGlobal(self.mapToGlobal(event.pos()))
            
            # 显示/隐藏按钮
            if prev_area.contains(pos):
                print("prev_area.contains(pos)")
                prev_btn.show()
            else:
                prev_btn.hide()
            
            if next_area.contains(pos):
                next_btn.show()
            else:
                next_btn.hide()
            
            # 调用原始的鼠标移动事件
            QWidget.mouseMoveEvent(photo_container, event)
        
        # 添加鼠标离开事件处理
        def leaveEvent(event):
            prev_btn.hide()
            next_btn.hide()
            QWidget.leaveEvent(photo_container, event)
        
        # 设置事件处理
        photo_container.mouseMoveEvent = mouseMoveEvent
        photo_container.leaveEvent = leaveEvent
        
        # 启用鼠标追踪
        photo_container.setMouseTracking(True)
        photo_label.setMouseTracking(True)  # 也为照片标签启用鼠标追踪
        
        # 将组件添加到布局
        photo_layout.addWidget(prev_btn)
        photo_layout.addWidget(scroll_area)  # 使用 scroll_area 而不是 photo_label
        photo_layout.addWidget(next_btn)
        
        main_layout.addWidget(photo_container, 1)  # 添加拉伸因子
        
        # 将页面添加到 stacked widget 并切换
        self.stacked_layout.addWidget(self.photo_view_page)
        self.stacked_layout.setCurrentWidget(self.photo_view_page)
        
        # 立即更新照片
        update_photo()
        
        # 设置焦点
        self.setFocus()

    def toggle_maximize(self):
        """切换最大化/还原窗口"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()


    def showMaximized(self):
        """重写最大化方法"""
        super().showMaximized()
        # 手动触发更新照片
        if hasattr(self, 'update_photo'):
            self.update_photo()

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.LeftButton and event.y() <= self.title_bar.height():
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            if self.isMaximized():
                self.showNormal()
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """处理鼠标双击事件"""
        if event.button() == Qt.LeftButton and event.y() <= self.title_bar.height():
            self.toggle_maximize()

    def restore_events(self):
        """在返回预览页面时恢复原始的键盘事件处理"""
        if hasattr(self, 'original_keyPressEvent'):
            self.keyPressEvent = self.original_keyPressEvent
        if hasattr(self, 'original_resize_event'):
            self.resizeEvent = self.original_resize_event

    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'last_folder': '',  # 上次打开的文件夹
                    'window_state': {
                        'width': 1200,
                        'height': 800,
                        'x': 100,
                        'y': 100,
                        'maximized': False
                    },
                    'face_recognition': {
                        'threshold': 0.5,
                        'backup_db': False,
                        'db_path': 'known_faces_op.pkl',
                        'update_db': True,
                        'clean_db': {
                            'output_path': 'known_faces_cleaned.pkl',
                            'method': 'dbscan',
                            'eps': 0.3,
                            'min_samples': 2,
                            'backup': True
                        }
                    }
                }
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
            self.config = {'last_folder': ''}

    def save_config(self):
        """保存配置文件"""
        try:
            # 更新窗口状态
            self.config['window_state'] = {
                'width': self.width(),
                'height': self.height(),
                'x': self.x(),
                'y': self.y(),
                'maximized': self.isMaximized()
            }
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置失败: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭时保存配置"""
        self.save_config()
        event.accept()

    def clear_layout(self, layout):
        """清除布局中的所有组件"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def handle_unknown_face(self, image_path):
        """处理未识别的人脸"""
        reply = QMessageBox.question(
            self,
            "未识别的人脸",
            f"在图片 {os.path.basename(image_path)} 中检测到未识别的人脸，是否要注册？"
        )
        
        if reply == QMessageBox.Yes:
            self.register_face(image_path)
        
        # 无论用户是否选择注册，都恢复线程运行
        self.process_worker.resume()

    def add_switch_buttons(self, layout, is_first_page=True):
        """添加切换按钮"""
        # 创建按钮容器
        button_container = QWidget()
        button_container.setFixedHeight(50)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建按钮
        left_btn = KeyButton("◀")
        right_btn = KeyButton("▶")
        
        # 设置按钮样式
        for btn in (left_btn, right_btn):
            btn.setFixedSize(30, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(45, 45, 45, 0.7);
                    color: #cccccc;
                    border: none;
                    border-radius: 15px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: rgba(60, 60, 60, 0.8);
                }
            """)
        
        # 根据页面显示不同的按钮
        if is_first_page:
            right_btn.setVisible(True)
            left_btn.setVisible(False)
        else:
            right_btn.setVisible(False)
            left_btn.setVisible(True)
        
        # 添加按钮到布局
        button_layout.addStretch()
        button_layout.addWidget(left_btn)
        button_layout.addWidget(right_btn)
        button_layout.addStretch()
        
        # 连接信号
        left_btn.clicked.connect(lambda: self.switch_cards_page(True))
        right_btn.clicked.connect(lambda: self.switch_cards_page(False))
        
        # 将按钮容器添加到主布局
        layout.addWidget(button_container)

    def switch_cards_page(self, to_first_page):
        """切换卡片页面"""
        # 创建动画
        self.fade_anim = QPropertyAnimation(self.stacked_layout.currentWidget(), b"windowOpacity")
        self.fade_anim.setDuration(150)  # 设置较短的动画时间
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        
        def switch_complete():
            if to_first_page:
                # 切换到第一个页面（主页面）
                self.stacked_layout.setCurrentWidget(self.cards_page)
                self.left_btn.setEnabled(False)
                self.right_btn.setEnabled(True)
                self.page_title.setText("已识别人脸")  # 更新标题
            else:
                # 切换到第二个页面（未识别/无人脸页面）
                self.stacked_layout.setCurrentWidget(self.cards_page_2)
                self.left_btn.setEnabled(True)
                self.right_btn.setEnabled(False)
                self.page_title.setText("风格界面")  # 更新标题
            
            # 创建淡入动画
            fade_in = QPropertyAnimation(self.stacked_layout.currentWidget(), b"windowOpacity")
            fade_in.setDuration(150)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.start()
            
            # 预加载和更新内容
            QTimer.singleShot(0, self.update_person_grid)
        
        self.fade_anim.finished.connect(switch_complete)
        self.fade_anim.start()
class NSFWWorker(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, folder,nsfw_classifier):
        super().__init__()
        self.folder = folder
        self.nsfw_classifier = nsfw_classifier
    def run(self):
        results = self.nsfw_classifier.scan_directory(self.folder)
        self.finished.emit(results)

# 添加人脸聚类线程
class FaceClusteringWorker(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, folder, min_samples=3, eps=0.3):
        super().__init__()
        self.folder = folder
        self.min_samples = min_samples
        self.eps = eps
        self.clusterer = FaceClusterer(min_samples=min_samples, eps=eps)
    def run(self):
        # 等待系统空闲
        while True:
            if psutil.cpu_percent(interval=1) < 50:
                break
        clusters = self.clusterer.get_clusters(self.folder)
        self.finished.emit(clusters)

#添加去除重复文件线程
class RemoveDuplicatesWorker(QThread):
    finished = pyqtSignal(list)
    def __init__(self, folder,similarity_threshold):
        super().__init__()
        self.folder = folder
        self.duplicates_remover = DuplicateRemover(similarity_threshold=similarity_threshold)
    def run(self):
        duplicates = self.duplicates_remover.scan_directory(self.folder)
        if duplicates:
            self.finished.emit(duplicates)
        else:
            self.finished.emit([])
# 添加工作线程类
class PhotoProcessWorker(QThread):
    """照片处理工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    
    def __init__(self, folder, face_organizer):
        super().__init__()
        self.folder = folder
        self.face_organizer = face_organizer
    
    def run(self):
        results = {
            'face_db': {},
        }
        
        root = Path(self.folder)
        total_files = sum(1 for _ in root.rglob("*.jpg"))
        processed = 0
        
        # 初始化特殊类别
        results['face_db']["未识别"] = []
        results['face_db']["无人脸"] = []
        
        # 遍历文件夹
        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                continue
            
            # 获取目录名
            person_name = person_dir.name
            
            # 处理特殊目录
            if person_name == "未识别":
                # 将"未识别"目录的照片添加到未识别类别
                for photo_path in person_dir.glob("*.jpg"):
                    try:
                        results['face_db']["未识别"].append(str(photo_path))
                        processed += 1
                        self.progress.emit(int(processed * 100 / total_files))
                    except Exception as e:
                        print(f"处理照片失败 {photo_path}: {str(e)}")
                continue
                
            elif person_name == "无人脸":
                # 将"无人脸"目录的照片添加到无人脸类别
                for photo_path in person_dir.glob("*.jpg"):
                    try:
                        results['face_db']["无人脸"].append(str(photo_path))
                        processed += 1
                        self.progress.emit(int(processed * 100 / total_files))
                    except Exception as e:
                        print(f"处理照片失败 {photo_path}: {str(e)}")
                continue
            
            # 处理普通人物目录
            results['face_db'][person_name] = []
            for photo_path in person_dir.glob("*.jpg"):
                try:
                    results['face_db'][person_name].append(str(photo_path))
                    processed += 1
                    self.progress.emit(int(processed * 100 / total_files))
                except Exception as e:
                    print(f"处理照片失败 {photo_path}: {str(e)}")
                    continue
        
        # 处理根目录下的照片（使用根目录名作为分类）
        root_name = root.name
        if root_name not in results['face_db']:
            results['face_db'][root_name] = []
            
        for photo_path in root.glob("*.jpg"):
            try:
                results['face_db'][root_name].append(str(photo_path))
                processed += 1
                self.progress.emit(int(processed * 100 / total_files))
            except Exception as e:
                print(f"处理照片失败 {photo_path}: {str(e)}")
                continue
        
        # 移除空类别
        empty_categories = [k for k, v in results['face_db'].items() if not v]
        for category in empty_categories:
            del results['face_db'][category]
        # 只更新界面显示，不保存到数据库
        self.finished.emit(results)
class ProcessWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    register_request = pyqtSignal(str)
    # total_register_person = pyqtSignal(dict)        
    def __init__(self, files, face_organizer, output_dir,mode):
        super().__init__()
        self.files = files
        self.face_organizer = face_organizer
        self.output_dir = output_dir
        self.total_files = len(files)
        self.processed = 0
        self.mode = mode
        self.wait_condition = threading.Event()  # 添加条件变量
        self.wait_condition.set()  # 初始状态为运行
                
    def pause(self):
        """暂停线程"""
        self.wait_condition.clear()
                
    def resume(self):
        """恢复线程"""
        self.wait_condition.set()
                
    def process_file(self, file_path):
        # 检测人脸
        matched_name = self.face_organizer.compare_face(normalize_path(file_path))
        if matched_name:
            # 创建对应人名的文件夹
            copy_photo(matched_name,file_path,self.output_dir)
        else:
                # 发出注册请求信号并等待
                self.pause()  # 暂停线程
                self.register_request.emit(file_path)
                self.wait_condition.wait()  # 等待恢复信号
                
                self.processed += 1
                self.progress.emit(int(self.processed * 100 / self.total_files))
        return matched_name
    def run(self):
        self.total_register_person ={} 
        if self.mode == "folder":
            try:
                for root, _, files in os.walk(self.files):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(root, file)
                            matched_name = self.process_file(full_path)
                            if matched_name:
                                if matched_name not in self.total_register_person:
                                    self.total_register_person[matched_name] = 1
                                else:
                                    self.total_register_person[matched_name] += 1
            except Exception as e:
                self.error.emit(str(e))
            

        elif self.mode == "file":
            matched_name =  self.process_file(self.files)
            if matched_name:
                if matched_name not in self.total_register_person:
                    self.total_register_person[matched_name] = 1
                else:
                    self.total_register_person[matched_name] += 1
        self.finished.emit(self.total_register_person)
        self.total_register_person ={}
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DisableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = PhotoManager()
    # 设置应用图标
    app_icon = QIcon("../assets/image.ico")
    window.setWindowIcon(app_icon)
    QApplication.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec_()) 
