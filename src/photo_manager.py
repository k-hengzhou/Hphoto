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
import qtawesome as qta
import time
import cv2
from .utils import (copy_photo,cv2_to_qpixmap,get_config,normalize_path)
from .clean_face_db import clean_face_database
from .face_clustering import FaceClusterer
import psutil
from .nsfw_classifier import NSFWClassifier,default_nsfw_class
from .styles import *
import msgpack
import numpy as np
from collections import Counter
import random
from PyQt5.QtCore import QEventLoop
# 在创建QApplication之前设置高DPI缩放
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
person_info_base ={
            "name":None,
            "photo_path":None,
            "embedding":np.array([]),
            "gender":0,
            "age":0,
            "add_time":0,
            "open_time":0,
            "nsfw_score":0,
            "nsfw_class":"",
            "duplicate_index":None,
            "star":False,
            "tag":[],
            "love_score":0,
            "is_nsfw":None,
            "face_cluster_index":None
        }
class VSCodeTooltip(QWidget):
    def __init__(self, title, shortcut=None, description=None, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TranslucentBackground)
                # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(2)
        
        # 添加标题
        title_label = QLabel(title)
        title_label.setStyleSheet(label_style)
        title_label.setStyleSheet("font-size: 24px;background-color: #252526;")
        layout.addWidget(title_label)
        
        # 如果有快捷键，添加快捷键
        if shortcut:
            shortcut_label = QLabel(shortcut)
            shortcut_label.setStyleSheet("color: #888888;")
            layout.addWidget(shortcut_label)
            
        # 如果有描述，添加描述
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888888;")
            layout.addWidget(desc_label)
            
        self.setStyleSheet(TOOLTIP_STYLE)

class QMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(MESSAGE_BOX_STYLE)
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
    def __init__(self, text=None, tooltip=None, parent=None):
        super().__init__(parent)
        self.setFixedSize(45, 45)
        self.tooltip_widget = None
        if text:
            self.setText(text)
        if tooltip:
            self.tooltip_widget = VSCodeTooltip(tooltip)
            self.tooltip_widget.hide()
        self.setStyleSheet(BUTTON_STYLE)
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
        try:
            if obj == self and self.tooltip_widget:
                if event.type() == QEvent.Enter:
                    self.showTooltip()
                elif event.type() == QEvent.Leave:
                    if self.tooltip_widget and self.tooltip_widget.isVisible():
                        self.tooltip_widget.hide()
                        self.tooltip_widget.close()  # 确保完全关闭
            return super().eventFilter(obj, event)
        except RuntimeError:
            self.removeEventFilter(self)
            return False
            

class PhotoManager(QMainWindow):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("照片管理器")
        self.setGeometry(100, 100, 1500, 900)
        self.thumbnail_timers = []
        # 加载配置
        self.config_file = "config/config.json"
        self.load_config()
        self.style_name =[["全部","收藏","未识别","无人脸"],["重复文件"]]
        self.person_info_base =person_info_base.copy()
        self.style_name.append(["未知人物"])
        self.style_name.append(default_nsfw_class)
        # self.style_name.append(["未知人物"])
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
        # left_buttons = QHBoxLayout()
        # left_buttons.setSpacing(15)
        left_button = QPushButton()
        left_button.setIcon(qta.icon('fa5s.bars', color='#c8c8c8'))
        left_button.setIconSize(QSize(24, 24))
        left_button.setStyleSheet(SWITCH_BUTTON_STYLE)


        # 创建一个下拉菜单
        self.function_menu = QMenu()
        self.function_menu.setStyleSheet(MENU_STYLE)
        self.open_btn = self.function_menu.addAction("打开照片库")
        self.new_photo_db_btn = self.function_menu.addAction("新建照片库")
        self.open_btn.setIcon(qta.icon('fa5s.images', color='#c8c8c8'))
        self.new_photo_db_btn.setIcon(qta.icon('fa5s.plus', color='#c8c8c8'))
        self.save_btn = self.function_menu.addAction("保存照片库")
        self.save_btn.setIcon(qta.icon('fa5s.save', color='#c8c8c8'))
        self.register_btn = self.function_menu.addAction("人脸注册")
        self.register_btn.setIcon(qta.icon('fa5s.user-plus', color='#c8c8c8'))
        self.folder_btn =self.function_menu.addAction("添加文件夹")
        self.folder_btn.setIcon(qta.icon('fa5s.folder', color='#c8c8c8'))
        self.file_btn=self.function_menu.addAction("添加文件")
        self.file_btn.setIcon(qta.icon('fa5s.file-image', color='#c8c8c8'))
        self.save_file_btn = self.function_menu.addAction("保存照片文件")
        # 生成文件夹的icon
        self.save_file_btn.setIcon(qta.icon('fa5s.download', color='#c8c8c8'))
        self.setting_btn = self.function_menu.addAction("设置")
        self.setting_btn.setIcon(qta.icon('fa5s.cog', color='#c8c8c8'))
        left_button.setMenu(self.function_menu)

        
        
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
            btn.setFixedSize(48, 48)  # 增大尺寸
            btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        
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
            btn.setFixedSize(70, 70)  # 保持原有尺寸
            right_buttons.addWidget(btn)
        
        # 将三个部分添加到标题栏布局
        
        title_layout.addWidget(left_button)
        title_layout.addStretch()
        title_layout.addStretch()
        title_layout.addLayout(center_buttons)
        title_layout.addStretch()
        title_layout.addLayout(right_buttons)
        
        # 连接按钮信号
        self.open_btn.triggered.connect(self.open_photo_db_file)
        self.new_photo_db_btn.triggered.connect(self.new_photo_db)
        self.save_btn.triggered.connect(self.save_photo_db)
        self.folder_btn.triggered.connect(self.add_folder)
        self.file_btn.triggered.connect(self.add_file)
        self.register_btn.triggered.connect(lambda: self.register_face())
        self.setting_btn.triggered.connect(self.setting)
        self.save_file_btn.triggered.connect(self.save_file)
        
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
        
        self.photo_db = []
        self.photo_db_len = 0
        self.is_first_page = True
        self.is_open_nsfw = False
        self.is_select_page = False
        self.selected_images = []
        # 初始化人脸识别器
        try:
            # 获取配置中的人脸库路径
            faces_db_path = get_config(self.config,'face_recognition','db_path','known_faces_op.pkl')    
            threshold = get_config(self.config,'face_recognition','threshold',0.5)
            update_db = get_config(self.config,'face_recognition','update_db',True)
            backup_db = get_config(self.config,'face_recognition','backup_db',False)
            ingest_model_path = get_config(self.config,'face_recognition','insightface_model_path','model')
            ingest_model_provider = get_config(self.config,'face_recognition','insightface_model_provider','CPUExecutionProvider')
            face_confidence = get_config(self.config,'face_recognition','face_confidence',0.5)
            self.face_organizer = FaceOrganizer(
                model_path=ingest_model_path,
                providers=[ingest_model_provider],        # CPU GPU
                confidence=face_confidence,             # 人脸置信度
                faces_db_path=faces_db_path,            # 数据库路径
                threshold=threshold,                   # 匹配阈值
                update_db=update_db,                  # 允许更新数据库
                backup_db=backup_db                   # 启用数据库备份
            )
            # 等待五分后在启动人脸识别器
            # self.timer = QTimer()
            # self.timer.timeout.connect(lambda:self.create_thread())
            # self.timer.start(300000)
            # self.face_db={}

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
        self.scroll_area.setStyleSheet(MAIN_WINDOW_STYLE)
        self.cards_layout.addWidget(self.scroll_area)
        
        # 创建第二个页面和滚动区域
        self.cards_page_2 = QWidget()
        self.cards_layout_2 = QVBoxLayout(self.cards_page_2)
        
        self.scroll_area_2 = QScrollArea()
        self.scroll_area_2.setWidgetResizable(True)
        self.scroll_area_2.setStyleSheet(MAIN_WINDOW_STYLE)
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
        self.nsfw_worker = None
        self.face_clustering_worker = None
        self.duplicate_worker = None
        # 设置窗口样式
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        
        # 添加防抖定时器，减少延迟时间
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.process_photo_db)
        # self.process_photo_db_timer = QTimer()
        # self.process_photo_db_timer.timeout.connect(self.process_photo_db)
        # #每秒执行一次
        # self.process_photo_db_timer.start(5000)
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
        self.unsafe_threshold = get_config(self.config,'nsfw_classifier','unsafe_threshold',0.6) 
        self.model_path = get_config(self.config,'nsfw_classifier','model_path','D:/project/小程序/model/640m.onnx')
        self.providers = get_config(self.config,'nsfw_classifier','providers',['CPUExecutionProvider'])
        self.nsfw_class = get_config(self.config,'nsfw_classifier','nsfw_class',default_nsfw_class)
        self.nsfw_classifier = NSFWClassifier(unsafe_threshold=float(self.unsafe_threshold),
                                    model_path=self.model_path,
                                    providers=self.providers,
                                    nsfw_class=self.nsfw_class)
        self.face_clusterer = FaceClusterer(eps=get_config(self.config,'face_clustering','eps',0.6),
                                            min_samples=get_config(self.config,'face_clustering','min_samples',1))
        self.duplicates_remover = DuplicateRemover(similarity_threshold=get_config(self.config,'duplicate_photo','eps',0.9))
        self.photo_db_path = self.config['last_photo_db']
        self.photo_db = []
        if os.path.exists(self.photo_db_path):
            try:
                self.open_photo_db()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开照片数据库失败: {str(e)}")
                    # 初始化NSFW分类器
        self.thread_timer = QTimer()
        self.thread_timer.timeout.connect(self.create_thread)
        # 每分钟执行一次
        self.thread_timer.start(60*1000)
        
        
        # 保存原始的键盘事件处理函数
        # self.original_keyPressEvent = self.keyPressEvent
        # # 重写键盘事件处理函数
        # self.keyPressEvent = self.new_keyPressEvent
    def open_photo_db_file(self):
        """打开照片数据库文件"""
        file_name,_ = QFileDialog.getOpenFileName(self, "打开照片数据库", "photo.msgpack","所有文件 (*.*);;数据库文件 (*.msgpack)")
        if file_name:
            self.photo_db_path = file_name
            self.config['last_photo_db'] = file_name
            self.open_photo_db()
    def save_file(self):
        """保存照片文件"""
        if not os.path.exists("photo"):
            os.makedirs("photo")
        for item in self.photo_db:
            copy_photo(item['name'],item['photo_path'],"photos")

    def open_photo_db(self):
        """打开照片数据库"""
        with open(self.photo_db_path, 'rb') as f:
            self.photo_db = msgpack.load(f)
            for item in self.photo_db:
                item['photo_path'] = normalize_path(item['photo_path'])
                item['embedding'] = np.array(item['embedding']) 
                if 'gender' in item:
                    item['gender'] = np.int64(item['gender'])
        self.process_photo_db()
    def save_photo_db(self):
        """保存照片数据库"""
        if self.photo_db is None or len(self.photo_db) == 0:
            return
        for item in self.photo_db:
            
            item['photo_path'] = normalize_path(item['photo_path'])
            item['embedding'] = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
            if 'gender' in item:
                item['gender'] = int(item['gender'])
        with open(self.photo_db_path, 'wb') as f:
            msgpack.pack(self.photo_db, f)
    def handle_face_clustering(self,clusters):
        face_cluster_list =[item["face_cluster_index"] for item in 
                            self.photo_db if item["face_cluster_index"] is not None]
        for i in range(len(clusters)):
            j=0
            while j in face_cluster_list:
                j+=1
            for item in clusters[i]:
                photo_db_index = self.get_image_path_index(self.photo_db,item)
                self.photo_db[photo_db_index]['face_cluster_index'] = j
            face_cluster_list.append(j)
        for item in self.photo_db:
            if item['face_cluster_index'] is None:
                item['face_cluster_index'] = -1

    def handle_duplicates(self,photo_db):
        for i in range(len(photo_db)):
            if 'duplicate_index' not in photo_db[i] :
                continue
            elif photo_db[i]['duplicate_index'] is None:
                continue
            else:
                self.photo_db[i]['duplicate_index'] = photo_db[i]['duplicate_index']
        
        
    def clean_face_db(self,db_path,output_path,method,eps,min_samples,backup):
        """清理人脸数据库"""
        msg = clean_face_database(db_path=db_path,output_path=output_path,method=method,eps=float(eps),min_samples=int(min_samples)   ,backup=backup)
        QMessageBox.information(self, "清理结果", msg)
    def add_face_clustering(self,min_samples,eps,restart=False):
        """添加人脸聚类"""
        if self.face_clustering_worker is not None:
            if self.face_clustering_worker.isRunning():
                if restart:
                    self.face_clustering_worker.terminate()
                else:
                    return
        if restart:
            for item in self.photo_db:
                item["face_cluster_index"] = None
        self.face_clustering_worker = FaceClusteringWorker(self.photo_db,self.face_clusterer,self.face_organizer)
        self.face_clustering_worker.finished.connect(self.handle_face_clustering)
        self.face_clustering_worker.start()
        # msg = add_face_clustering(min_samples=int(min_samples),eps=float(eps))
        # QMessageBox.information(self, "人脸聚类结果", msg)
    def nsfw_classify(self,restart=False):
        """NSFW分类"""
        if self.nsfw_worker is not None:
            if self.nsfw_worker.isRunning():
                if restart:
                    self.nsfw_worker.terminate()
                else:
                    return
        if restart:
            for item in self.photo_db:
                item["is_nsfw"] = None
                item["nsfw_score"] = 0
                item["nsfw_class"] = ""
        self.nsfw_worker = NSFWWorker(self.photo_db,self.nsfw_classifier,self.unsafe_threshold)
        self.nsfw_worker.finished.connect(self.handle_nsfw)
        self.nsfw_worker.start()
    def handle_nsfw(self,results):
        """处理NSFW分类结果"""
        for i in range(len(results)):
            self.photo_db[i]["nsfw_score"] = results[i]["nsfw_score"]
            self.photo_db[i]["nsfw_class"] = results[i]["nsfw_class"]
            self.photo_db[i]["is_nsfw"] = results[i]["is_nsfw"]
        # self.process_photo_db()
    def setting_person_info(self,image_path_list):
        """设置人物信息"""
        image_index=self.get_image_path_index(self.photo_db,image_path_list[0])
        gender_list =["男","女"]
        dialog = QDialog(self)
        dialog.setWindowTitle("设置人物信息")
        dialog.setFixedSize(500, 400)
        dialog.setStyleSheet(SETTINGS_DIALOG_STYLE)
        layout = QVBoxLayout(dialog)
        layout.setAlignment(Qt.AlignTop)
        name_layout = QHBoxLayout()
        name_label = QLabel("姓名:")
        name_input = QLineEdit()
        name_input.setText(self.photo_db[image_index]["name"])
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        age_layout = QHBoxLayout()
        age_label = QLabel("年龄:")
        age_input = QLineEdit()
        age_input.setText(str(self.photo_db[image_index]["age"]))
        age_layout.addWidget(age_label)
        age_layout.addWidget(age_input)
        layout.addLayout(age_layout)
        gender_layout = QHBoxLayout()
        gender_label = QLabel("性别:")
        gender_input = QComboBox()
        gender_input.setStyleSheet(QComboBox_style)
        gender_input.addItems(gender_list)
        gender_input.setCurrentText(gender_list[self.photo_db[image_index]["gender"]])
        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(gender_input)
        layout.addLayout(gender_layout)
        is_nsfw_layout = QHBoxLayout()
        is_nsfw_label = QLabel("是否NSFW:")
        is_nsfw_input = QCheckBox()
        is_nsfw_input.setChecked(self.photo_db[image_index]["is_nsfw"])
        nsfw_score_label = QLabel("NSFW分数:")
        nsfw_score_input = QLineEdit()
        nsfw_score_input.setText(f"{self.photo_db[image_index]['nsfw_score']:.2f}")
        nsfw_score_input.setFixedSize(100, 40)
        duplicate_index_label = QCheckBox("重复照片")
        duplicate_index_label.setChecked(self.photo_db[image_index]["duplicate_index"] is not None and self.photo_db[image_index]["duplicate_index"] != -1)
        layout.addWidget(duplicate_index_label)
        face_cluster_index_label = QCheckBox("人脸聚类")
        face_cluster_index_label.setChecked(self.photo_db[image_index]["face_cluster_index"] is not None and self.photo_db[image_index]["face_cluster_index"] != -1)
        layout_1 = QHBoxLayout()
        layout_1.addWidget(face_cluster_index_label)
        layout_1.addWidget(duplicate_index_label)
        layout.addLayout(layout_1)
        is_nsfw_layout.addWidget(is_nsfw_label)
        is_nsfw_layout.addWidget(is_nsfw_input)
        is_nsfw_layout.addStretch()
        is_nsfw_layout.addWidget(nsfw_score_label)
        is_nsfw_layout.addWidget(nsfw_score_input)
        layout.addLayout(is_nsfw_layout)
        score_layout = QHBoxLayout()
        score_label = QLabel("得分:")
        score_input = QLineEdit()
        score_input.setText(f"{self.photo_db[image_index]['love_score']:.2f}")
        # score_input.setFixedSize(100, 40)
        star_combo = QCheckBox("收藏")
        star_combo.setChecked(self.photo_db[image_index]["star"])
        score_layout.addWidget(score_label)
        score_layout.addWidget(score_input)
        score_layout.addWidget(star_combo)
        layout.addLayout(score_layout)
        # 保存按钮
        save_btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(lambda: save_person_info())
        save_btn_layout.addWidget(save_btn)
        def save_person_info():
            for image_path in image_path_list:
                image_index=self.get_image_path_index(self.photo_db,image_path)
                person_info =self.photo_db[image_index]
                person_info["name"] = name_input.text()
                person_info["age"] = int(age_input.text())
                person_info["gender"] = gender_list.index(gender_input.currentText())
                person_info["is_nsfw"] = is_nsfw_input.isChecked()
                person_info["star"] = star_combo.isChecked()
                person_info["love_score"] = float(score_input.text())
                if duplicate_index_label.isChecked():
                    person_info["duplicate_index"] = None
                else:
                    person_info["duplicate_index"] = -1
                if face_cluster_index_label.isChecked():
                    person_info["face_cluster_index"] = None
                else:
                    person_info["face_cluster_index"] = -1
            dialog.close()
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.close)
        save_btn_layout.addWidget(cancel_btn)
        layout.addLayout(save_btn_layout)
        self.selected_images = []
        dialog.exec_()
    
    def setting(self):
        """设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("设置")
        dialog.setFixedSize(500, 1200)
        dialog.setStyleSheet(SETTINGS_DIALOG_STYLE)
        
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
            h_line.setStyleSheet("background-color: #3d3d3d;front_size=24px; color: #ffffff;padding: 0;margin: 0;")
            line_label = QLabel(text)
            line_label.setStyleSheet("color: #ffffff; border: none; front_size=24px;background-color: transparent;padding: 0;margin: 0;")
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
        # 模型路径
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("模型路径:")
        model_path_input = QLineEdit()
        model_path_input.setText(get_config(self.config,'face_recognition','insightface_model_path','model'))
        model_path_browse = QPushButton("浏览")
        model_path_browse.clicked.connect(lambda: browse_db())
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(model_path_input)
        model_path_layout.addWidget(model_path_browse)
        register_layout.addLayout(model_path_layout)
        model_provider_layout = QHBoxLayout()
        model_provider_label = QLabel("模型提供者:")
        model_provider_input = QComboBox()
        model_provider_input.setStyleSheet(QComboBox_style)
        model_provider_input.addItems(["CPUExecutionProvider", "CUDAExecutionProvider"])
        model_provider_input.setCurrentText(get_config(self.config,'face_recognition','insightface_model_provider','CPUExecutionProvider'))
        model_provider_layout.addWidget(model_provider_label)
        model_provider_layout.addWidget(model_provider_input)
        #人脸置信度
        face_confidence_layout = QHBoxLayout()
        face_confidence_label = QLabel("人脸置信度:")
        face_confidence_input = QLineEdit()
        face_confidence_input.setText(str(get_config(self.config,'face_recognition','face_confidence',0.5)))
        face_confidence_layout.addWidget(face_confidence_label)
        face_confidence_layout.addWidget(face_confidence_input)
        register_layout.addLayout(model_provider_layout)
        register_layout.addLayout(face_confidence_layout)
        layout.addLayout(register_layout)
        
        

        clean_layout = QVBoxLayout()
        # 清理人脸库
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
        method_input.setStyleSheet(QComboBox_style)
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
        add_face_clustering_btn.clicked.connect(lambda: self.add_face_clustering(face_clustering_min_samples_input.text(),face_clustering_eps_input.text(),restart=True))
        add_face_clustering_layout.addLayout(face_clustering_min_samples_layout)
        add_face_clustering_layout.addWidget(add_face_clustering_btn)
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
        nsfw_providers_input.setStyleSheet(QComboBox_style)
        nsfw_providers_input.addItems(["CPUExecutionProvider", "CUDAExecutionProvider"])
        nsfw_providers_input.setCurrentText(get_config(self.config,'nsfw_classifier','providers','CPUExecutionProvider'))
        nsfw_providers_layout.addWidget(nsfw_providers_label)
        nsfw_providers_layout.addWidget(nsfw_providers_input)
        add_nsfw_layout.addLayout(nsfw_providers_layout)
        add_nsfw_layout.addLayout(nsfw_model_path_layout)
        nsfw_button = QPushButton("NSFW 重新分类")
        nsfw_button.clicked.connect(lambda: self.nsfw_classify(restart=True))
        self.unsafe_threshold = nsfw_threshold_input.text()
        add_nsfw_layout.addWidget(nsfw_button)


        layout.addLayout(add_face_clustering_layout)
        layout.addLayout(add_nsfw_layout)
        duplicate_layout = QVBoxLayout()
        add_line(duplicate_layout,"重复照片")
        duplicate_eps_layout = QHBoxLayout()
        duplicate_eps_label = QLabel("eps:")
        duplicate_eps_input = QLineEdit()
        duplicate_eps_input.setText(str(get_config(self.config,'duplicate_photo','eps',0.9)))
        duplicate_eps_layout.addWidget(duplicate_eps_label)
        duplicate_eps_layout.addWidget(duplicate_eps_input)
        duplicate_button = QPushButton("重复照片")
        duplicate_button.clicked.connect(lambda: self.duplicate_photo(duplicate_eps_input.text(),restart=True))    
        duplicate_eps_layout.addWidget(duplicate_button)
        duplicate_layout.addLayout(duplicate_eps_layout)
        layout.addLayout(duplicate_layout)
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
                "所有文件 (*.*)"
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
                    'ingest_model_path': model_path_input.text(),
                    'ingest_model_provider': model_provider_input.currentText(),
                    'face_confidence': float(face_confidence_input.text()) if face_confidence_input.text() else 0.5
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
                self.config['duplicate_photo'] = {
                    'eps': float(duplicate_eps_input.text()) if duplicate_eps_input.text() else 0.3
                }
                if "nsfw_class" not in self.config['nsfw_classifier']:
                    self.config['nsfw_classifier']['nsfw_class'] = default_nsfw_class
                # 保存配置
                self.save_config()
                self.face_organizer = FaceOrganizer(
                    threshold=threshold,
                    backup_db=backup_checkbox.isChecked(),
                    faces_db_path=db_input.text(),
                    model_path=normalize_path(model_path_input.text()),
                    # ingest_model_provider=model_provider_input.currentText(),
                    confidence=float(face_confidence_input.text()) if face_confidence_input.text() else 0.5,
                    update_db=True,
                    providers=[model_provider_input.currentText()]
                )
                self.nsfw_classifier = NSFWClassifier(
                    model_path=normalize_path(nsfw_model_path_input.text()),
                    providers=[nsfw_providers_input.currentText()],
                    unsafe_threshold=float(nsfw_threshold_input.text()) if nsfw_threshold_input.text() else 0.6
                )
                self.unsafe_threshold = float(nsfw_threshold_input.text()) if nsfw_threshold_input.text() else 0.6
                min_samples = int(face_clustering_min_samples_input.text()) if face_clustering_min_samples_input.text() else 2
                eps = float(face_clustering_eps_input.text()) if face_clustering_eps_input.text() else 0.3
                self.add_face_clustering(min_samples,eps)
                # 更新人脸识别器设置
                # self.face_organizer.threshold = threshold
                # self.face_organizer.backup_db = backup_checkbox.isChecked()
                # self.face_organizer.faces_db_path = db_input.text()
                
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
            files=folder_path,
            face_organizer=self.face_organizer,
            mode="folder",
            nsfw_classifier=self.nsfw_classifier
        )
        
        # 连接信号
        self.process_worker.progress.connect(self.progress_bar.setValue)
        self.process_worker.finished.connect(self.on_process_complete)
        self.process_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self.process_worker.register_request.connect(self.handle_unknown_face)  # 处理未识别人脸
        
        # 启动线程
        self.process_worker.start()
        
        # 禁用相关按钮
        # self.open_btn.setEnabled(False)
        # self.folder_btn.setEnabled(False)
        # self.register_btn.setEnabled(False)

    def add_file(self):
        """添加文件"""
        # 创建文件对话框
        file_path, _ = QFileDialog.getOpenFileName(self,
            "选择文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png)")
        
        if file_path == "":
            print("取消添加文件")
            return
        # 显示进度条
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        
        # 创建工作线程实例
        self.process_worker = ProcessWorker(
            files=file_path,
            face_organizer=self.face_organizer,
            mode="file",
            nsfw_classifier=self.nsfw_classifier
        )
        
        # 连接信号
        self.process_worker.progress.connect(self.progress_bar.setValue)
        self.process_worker.finished.connect(self.on_process_complete)
        self.process_worker.error.connect(lambda msg: QMessageBox.warning(self, "错误", msg))
        self.process_worker.register_request.connect(self.handle_unknown_face)  # 处理未识别人脸
        
        # 启动线程
        self.process_worker.start()
        
        # 禁用相关按钮
        # self.open_btn.setEnabled(False)
        # self.folder_btn.setEnabled(False)
        # self.register_btn.setEnabled(False)

    def on_process_complete(self,total_register_person,total_register_person_info):
        """处理完成后的回调"""
        # 恢复按钮状态
        self.open_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.register_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.hide()
        path_list = [item["photo_path"] for item in self.photo_db]
        for item in total_register_person_info:
            if item is not None:
                if normalize_path(item["photo_path"]) not in path_list:
                    self.photo_db.append(item)
                else:
                    total_register_person[item["name"]] -= 1
        
       
        self.process_photo_db()
        
        # 显示完成消息,在一定时间后自动关闭
        str_msg = "添加完成\n"
        for person,count in total_register_person.items():
            str_msg += f"{person}：共 {count} 张照片\n"
        QMessageBox.information(self, "完成", str_msg)

    def resizeEvent(self, event):
        """窗口大小变化时重新排列缩略图"""
        super().resizeEvent(event)
        print("resizeEvent")
      
        # if hasattr(setlf, 'update_preview_grid'):
        if self.stacked_layout.currentWidget() == self.preview_page:
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
    
    def calculate_layout(self,cols=None):
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
        person_name = list(self.face_db.keys())
        is_first_page = (self.stacked_layout.currentWidget() == self.cards_page)
        if is_first_page:
            layout_info = self.calculate_layout()
        else:
            layout_info = self.calculate_layout(cols=5)
        
        # 获取对应的网格容器
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(layout_info['spacing'])
        grid_layout.setContentsMargins(20, 20, 20, 20)
        # 找出重复文件
        
        # 预先筛选需要显示的人物
        if is_first_page:
            people_to_show = []
            for person_name, photos in self.face_db.items():
                if not photos:
                    continue
                if self.is_person_name(person_name) :
                    people_to_show.append((person_name, len(photos)))
            people_to_show.sort(key=lambda x: x[1], reverse=True)
            row = col = 0
            for person_name, _ in people_to_show:
                person_card = self.create_person_card(person_name, layout_info['card_width'])
                grid_layout.addWidget(person_card, row, col)
                col += 1
                if col >= layout_info['cols']:
                    col = 0
                    row += 1
        else:
            row = col = 0
            for style_type in self.style_name:
                for style_name in style_type:
                    if style_name in self.face_db:
                        person_card = self.create_person_card(style_name, layout_info['card_width'], is_other=True)
                        grid_layout.addWidget(person_card, row, col)
                        col += 1    
                col = 0
                row += 1
            row += 1
            person_name_list = list(self.face_db.keys())
            # print(person_name_list)
            for person_name in person_name_list:
                if "未知人物" in person_name:
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
                    scaled_pixmap = pixmap.scaled(width, width, 
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
                background:transparent;
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
                background: transparent;
                border: none;
            }
        """)
        
        layout.addWidget(avatar_container)
        
        # 设置卡片整体样式
        card.setStyleSheet("background: transparent; border: none;")
        
        # 添加悬停效果
        def enterEvent(event):
            avatar_container.setStyleSheet("""
                QWidget {
                    background: rgba(255, 255, 255, 0.1);
                    border: none;
                }
            """)
        
        def leaveEvent(event):
            avatar_container.setStyleSheet("""
                QWidget {
                    background: transparent;
                    border: none;
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
                    self.show_preview(person_name=person_name, image_paths=image_paths)
        
        card.mouseReleaseEvent = mouseReleaseEvent
        
        # 修改鼠标样式
        card.setCursor(Qt.PointingHandCursor)
        
        return card

    def new_photo_db(self):
        """新建照片数据库"""
        self.photo_db = []
        dialog = QFileDialog(self)
        dialog.setWindowTitle("新建照片数据库")
        dialog.setAcceptMode(QFileDialog.AcceptSave)  # 设置为保存模式
        dialog.setFileMode(QFileDialog.AnyFile)       # 允许选择任何文件
        dialog.setNameFilter("数据库文件 (*.msgpack);;所有文件 (*.*)")
        dialog.setDefaultSuffix("msgpack")           # 设置默认后缀
        
        if dialog.exec_() == QFileDialog.Accepted:
            file_name = dialog.selectedFiles()[0]     # 获取选择的文件路径
            if file_name:
                self.photo_db_path = file_name
                self.config['last_photo_db'] = self.photo_db_path
                self.photo_db = []
                self.process_photo_db()
                
    def create_thread(self):
        """创建线程"""
        print("create_thread")
        self.nsfw_classify()
        self.add_face_clustering(self.config['face_clustering']['min_samples'],self.config['face_clustering']['eps'])
        self.duplicate_photo(self.config['duplicate_photo']['eps'])
        self.process_photo_db()
    def duplicate_photo(self,eps=0.9,restart=False):
        """重复照片"""
        if self.duplicate_worker is not None:
            if self.duplicate_worker.isRunning():
                if restart:
                    self.duplicate_worker.terminate()
                else:
                    return
        if restart:
            for item in self.photo_db:
                item["duplicate_index"] = None
                # print("duplicate_index_counter",duplicate_index_counter)
                # print("duplicate_index_counter",duplicate_index_counter)
        duplicate_index=[item["duplicate_index"] for item in self.photo_db if item["duplicate_index"] is not None]
        duplicate_index_counter=Counter(duplicate_index)
                # print("duplicate_index_counter",duplicate_index_counter)
        for item in self.photo_db:
            if item["duplicate_index"] is not None:
                if duplicate_index_counter[item["duplicate_index"]] <= 1:
                    item["duplicate_index"] = -1
        no_duplicate_photo = [item for item in self.photo_db if item["duplicate_index"] is None]
        # if len(no_duplicate_photo) >100 :
        #     duplicates_db = no_duplicate_photo[:100]
        # else:
        to_duplicates_db = no_duplicate_photo
        self.duplicate_worker = RemoveDuplicatesWorker(to_duplicates_db,self.photo_db,self.duplicates_remover)
        self.duplicate_worker.finished.connect(self.handle_duplicates)
        self.duplicate_worker.start()
    def process_photo_db(self):
        """打开照片数据库"""
        if self.photo_db is None:
            return
        loop = QEventLoop()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        #判断self.worker 是否在运行
        if self.worker is not None:
            if self.worker.isRunning():
                return
        self.worker = PhotoProcessWorker(self.photo_db, self.face_organizer)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_complete)
        self.worker.finished.connect(loop.quit)
        self.worker.start()
        #阻塞
        loop.exec_()
        # self.create_thread()
        

        
        self.update_person_grid()
        # self.open_btn.setEnabled(False)
        # self.register_btn.setEnabled(False)

    def update_progress(self, value):
        """更新进度条"""
        # self.progress_bar.setValue(value)
        pass
    def process_complete(self, results):
        """处理完成后更新界面"""
        self.face_db=results['face_db']
        # self.update_person_grid()
        
        # 重新启用按钮
        # self.open_btn.setEnabled(True)
        # self.register_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.hide()
    def delete_photo(self, image_path,is_delete_local=True):
        """删除照片"""
        if is_delete_local:
            tip="确定要删除这些照片吗？"
        else:
            tip="确定要移出照片库吗？"
        reply = QMessageBox.question(
            self,
            "确认删除",
            tip
        )
    
        if reply == QMessageBox.Yes:
            try:
                # 删除文件
                if isinstance(image_path,str):
                    if os.path.exists(image_path):
                        if is_delete_local:
                            os.remove(image_path)
                    # if image_path in self.current_preview_images:
                    #     self.current_preview_images.remove(image_path)
                    path_list = [item["photo_path"] for item in self.photo_db]
                    if normalize_path(image_path) in path_list:
                        self.photo_db.remove(self.photo_db[path_list.index(image_path)])
                elif isinstance(image_path,list):
                    for item in image_path:
                        if os.path.exists(item):
                            if is_delete_local:
                                os.remove(item)
                        if item in self.current_preview_images:
                            self.current_preview_images.remove(item)
                        path_list = [item["photo_path"] for item in self.photo_db]
                        if normalize_path(item) in path_list:
                            self.photo_db.remove(self.photo_db[path_list.index(item)])
                # 更新界面
                self.selected_images=[]
                self.is_select_page=False
                duplicate_index=[item["duplicate_index"] for item in self.photo_db if item["duplicate_index"] is not None]
                duplicate_index_counter=Counter(duplicate_index)
                # print("duplicate_index_counter",duplicate_index_counter)
                for item in self.photo_db:
                    if item["duplicate_index"] is not None:
                        if duplicate_index_counter[item["duplicate_index"]] <= 1:
                            
                            item["duplicate_index"] = None
                            self.current_preview_images.remove(item["photo_path"])
                        
                            
                
                # self.show_preview(person_name=self.title, image_paths=self.current_preview_images)
                QMessageBox.information(self, "成功", "照片已删除")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")
            self.show_preview(person_name=self.title, image_paths=self.current_preview_images)
        else:
            self.show_preview(person_name=self.title, image_paths=self.current_preview_images)
    
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
        
        pixmap = QPixmap()
        image_path = normalize_path(image_path)
        # 提取人脸
        faces, img= self.face_organizer.detect_faces(image_path)
        if faces is None:
            person_info =self.person_info_base.copy()
            person_info["name"]="无人脸"
            person_info["photo_path"]=normalize_path(image_path)
            person_info["add_time"]=time.time()
            person_info["open_time"]=time.time()
            self.photo_db.append(person_info)
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
        img_size_x,img_size_y = img.shape[0],img.shape[1]
        if img_size_x > img_size_y:
            if img_size_x > 400:
                ratio = img_size_y/img_size_x
                img = cv2.resize(img, (int(400*ratio),400 ))
        elif img_size_y > img_size_x:
            if img_size_y > 400:
                ratio = img_size_x/img_size_y
                img = cv2.resize(img, (400,int(400*ratio)))
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
        name_input.setStyleSheet("background-color: #ffffff; color: black;font-size: 24px;")
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        button_layout = QHBoxLayout()
        confirm_btn = KeyButton("确认",tooltip="确认注册")
        cancel_btn = KeyButton("取消",tooltip="取消注册")
        no_face_btn = KeyButton("入库",tooltip="只入库，不注册人脸")
        button_layout.addWidget(no_face_btn)
        button_layout.addWidget(confirm_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        def on_no_face():
            person_name = name_input.text().strip()
            if person_name is None or person_name == "":
                person_name="未识别"
            person_info =self.person_info_base.copy()
            person_info["name"]=person_name
            person_info["age"]=faces[combo_box.currentIndex()].age
            person_info["gender"]=faces[combo_box.currentIndex()].gender
            person_info["photo_path"]=normalize_path(image_path)
            person_info["embedding"]=faces[combo_box.currentIndex()].embedding
            person_info["add_time"]=time.time()
            person_info["open_time"]=time.time()
            self.photo_db.append(person_info)
            preview_dialog.accept()
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
                person_info =self.person_info_base.copy()
                person_info["name"]=person_name
                person_info["age"]=faces[combo_box.currentIndex()].age
                person_info["gender"]=faces[combo_box.currentIndex()].gender
                person_info["photo_path"]=normalize_path(image_path)
                person_info["embedding"]=faces[combo_box.currentIndex()].embedding
                person_info["add_time"]=time.time()
                person_info["open_time"]=time.time()
                path_list = [item["photo_path"] for item in self.photo_db]
                if normalize_path(image_path) not in path_list:
                    self.photo_db.append(person_info)
                
                else:
                    # 返回image_path在self.photo_db中的索引
                    index = path_list.index(normalize_path(image_path))
                    self.photo_db[index]["name"] = person_name
                    self.photo_db[index]["gender"] = faces[combo_box.currentIndex()].gender
                    self.photo_db[index]["age"] = faces[combo_box.currentIndex()].age
                    self.photo_db[index]["embedding"] = faces[combo_box.currentIndex()].embedding
                    if "face_cluster_index" in self.photo_db[index]:
                        face_cluster_index = self.photo_db[index]["face_cluster_index"]
                        self.photo_db[index]["face_cluster_index"] = None
                        for item in self.photo_db:
                            if item["face_cluster_index"] == face_cluster_index:
                                self.photo_db[index]["name"] = person_name
                                item["face_cluster_index"] = None
                
                QMessageBox.information(
                    preview_dialog,
                    "成功",
                    f"已成功注册 {person_name}"
                )
                self.process_photo_db()
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
        no_face_btn.clicked.connect(on_no_face)
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
    def is_person_name(self,person_name):
        """判断是否是人物名"""
        flat_list = [item for sublist in self.style_name for item in sublist]
        if person_name not in flat_list and person_name.split("_")[0] not in flat_list:
            return True
        else:
            return False
    def  sort_photo(self,title,sort_type):
        """排序"""
        if sort_type == "按入库正序":
            self.person_name_db.sort(key=lambda x: x["add_time"],reverse=False)
        elif sort_type == "按入库倒序":
            self.person_name_db.sort(key=lambda x: x["add_time"],reverse=True)
        elif sort_type == "按最近正序":
            self.person_name_db.sort(key=lambda x: x["open_time"],reverse=True)
        elif sort_type == "按最近倒序":
            self.person_name_db.sort(key=lambda x: x["open_time"],reverse=False)
        elif sort_type == "按得分正序":
            self.person_name_db.sort(key=lambda x: x["love_score"],reverse=False)
        elif sort_type == "按得分倒序":
            self.person_name_db.sort(key=lambda x: x["love_score"],reverse=True)
        elif sort_type == "按NSFW正序":
            self.person_name_db.sort(key=lambda x: x["nsfw_score"],reverse=True)
        elif sort_type == "按NSFW倒序":
            self.person_name_db.sort(key=lambda x: x["nsfw_score"],reverse=False)
        elif sort_type == "随机排序":
            random.shuffle(self.person_name_db)
        self.show_preview(title,person_name_db=self.person_name_db)
    def get_image_path_index(self,photo_db,image_path):
        """获取图片路径在self.person_name_db中的索引"""
        path_list = [item["photo_path"] for item in photo_db]
        return path_list.index(image_path)
    def switch_nsfw(self,is_show_preview=True):
        """切换NSFW开关"""
        self.is_open_nsfw = not self.is_open_nsfw
        if self.is_open_nsfw:
            # 过滤掉非NSFW的照片
            nsfw_list=[item for item in self.person_name_db if item["is_nsfw"] == True]
            nsfw_list.sort(key=lambda x: x["nsfw_score"],reverse=True)
            
            self.show_preview(person_name=self.title, person_name_db=nsfw_list)
        else:
            self.show_preview(person_name=self.title, person_name_db=self.person_name_db)
    def select_all(self,select_btn,heart_btn,delete_btn,sort_photo,switch_btn,setting_btn,database_slash,person_name, image_paths,person_name_db):
        """选择所有照片"""
        self.is_select_page = not self.is_select_page
        if  self.is_select_page:
            sort_photo.setEnabled(False)
            switch_btn.setEnabled(False)
            database_slash.setEnabled(True)
            database_slash.setVisible(True)
            self.selected_images=[]
            select_btn.setIcon(qta.icon("fa5s.tasks",color="#c8c8c8"))
            heart_btn.setVisible(True)
            heart_btn.setEnabled(True)
            delete_btn.setVisible(True)
            delete_btn.setEnabled(True)
            select_btn.setIcon(qta.icon("fa5s.check",color="#c8c8c8"))
            setting_btn.setVisible(True)
            setting_btn.setEnabled(True)
        else:
            self.show_preview(person_name=person_name, image_paths=image_paths,person_name_db=person_name_db)
    def love_photos(self,selected_images,person_name, image_paths=None,person_name_db=None):
        """爱心按钮"""
        self.is_select_page=False
        for image_path in selected_images:
            image_index=self.get_image_path_index(self.photo_db,image_path)
            self.photo_db[image_index]["star"]=True
        self.selected_images=[]
        # self.show_preview(person_name=person_name, image_paths=image_paths,person_name_db=person_name_db)
    def cleanup_timers(self):
        for timer in self.thumbnail_timers:
            timer.stop()
        self.thumbnail_timers.clear()
    def show_preview(self, person_name, image_paths=None,person_name_db=None):
        """显示照片预览"""
        # 保存当前人名，用于刷新
        self.title = person_name  # 添加这行
        if hasattr(self, 'thumbnail_timers'):
            self.cleanup_timers()
        # 清除原有内容
        self.clear_layout(self.preview_layout)
        # sielf.image_paths = image_paths
        # 创建顶部工具栏
        toolbar = QHBoxLayout()
        self.current_images = image_paths
        self.current_person = person_name

        # 返回按钮
        back_btn = KeyButton("返回")
        # 正确的返  回cards_page或者cards_page_2
        if self.is_first_page:
            back_btn.clicked.connect(lambda: (self.stacked_layout.setCurrentWidget(self.cards_page),
                                     self.process_photo_db()))
        else:
            back_btn.clicked.connect(lambda: (self.stacked_layout.setCurrentWidget(self.cards_page_2),
                                     self.process_photo_db()))
        toolbar.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(back_btn)
        toolbar.addStretch()
        if person_name_db is None:
            path_list = [item["photo_path"] for item in self.photo_db]
            person_name_db=[self.photo_db[path_list.index(item)] for item in image_paths if item in path_list]
            self.person_name_db = person_name_db
        
        # 添加标题
        if person_name is not None:
            title_label = QLabel(person_name)
            title_label.setStyleSheet(label_style)
            title_label.setAlignment(Qt.AlignCenter)
            # title_label.setFixedSize(len(person_name)*10+40,40)
            toolbar.addWidget(title_label)
        toolbar.addStretch()
        delete_btn=KeyButton(tooltip="删除本地照片")
        delete_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        delete_btn.setFixedSize(40,40)
        delete_btn.setIcon(qta.icon("fa5s.trash",color="#c8c8c8"))
        database_slash=KeyButton("",tooltip="移出数据库")
        database_slash.setIcon(qta.icon("fa5s.trash-alt",color="#c8c8c8"))
        # database_slash.setToolTip("删除数据库")
        database_slash.setStyleSheet(SWITCH_BUTTON_STYLE)
        database_slash.setFixedSize(40,40)
        database_slash.clicked.connect(lambda: self.delete_photo(self.selected_images,is_delete_local=False))
        toolbar.addWidget(delete_btn)
        toolbar.addWidget(database_slash)
        delete_btn.clicked.connect(lambda: (self.delete_photo(self.selected_images)))

        heart_btn=QPushButton()
        heart_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        heart_btn.setFixedSize(40,40)
        heart_btn.setIcon(qta.icon("fa5s.heart",color="#c8c8c8"))
        heart_btn.clicked.connect(lambda: self.love_photos(self.selected_images,person_name, image_paths,person_name_db))
        toolbar.addWidget(heart_btn)
        # heart_btn.clicked.connect(lambda: self.love_photo(image_paths))
        setting_btn=QPushButton()
        setting_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        setting_btn.setFixedSize(40,40)
        setting_btn.setIcon(qta.icon("fa5s.cog",color="#c8c8c8"))
        setting_btn.setIconSize(QSize(32, 32))
        toolbar.addWidget(setting_btn)
        setting_btn.clicked.connect(lambda: self.setting_person_info(self.selected_images))
        select_btn=QPushButton()
        select_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        select_btn.setFixedSize(40,40)
        toolbar.addWidget(select_btn)
        # select_btn.clicked.connect(lambda _,select_btn=select_btn,heart_btn=heart_btn,delete_btn=delete_btn,sort_photo=sort_photo,switch_btn=switch_btn,database_slash=database_slash: self.select_all(select_btn,heart_btn,delete_btn,sort_photo,switch_btn,database_slash,person_name, image_paths,person_name_db))
        sort_photo=QPushButton()
        sort_photo.setStyleSheet(SWITCH_BUTTON_STYLE)
        sort_photo.setFixedSize(40,40)
        sort_photo.setIcon(qta.icon("fa5s.sort-amount-down",color="#c8c8c8")) 
        sort_photo.setIconSize(QSize(32, 32))
        sort_type =QMenu()
        sort_type.setStyleSheet("background-color: #1e1e1e; color: white;")
        sort_add_time=sort_type.addAction("按入库正序")
        sort_add_time.triggered.connect(lambda: self.sort_photo(person_name,"按入库正序"))
        sort_add_time_reverse=sort_type.addAction("按入库倒序")
        sort_add_time_reverse.triggered.connect(lambda: self.sort_photo(person_name,"按入库倒序"))
        sort_open=sort_type.addAction("按最近正序")
        sort_open.triggered.connect(lambda: self.sort_photo(person_name,"按最近正序"))
        sort_open_reverse=sort_type.addAction("按最近倒序")
        sort_open_reverse.triggered.connect(lambda: self.sort_photo(person_name,"按最近倒序"))
        sort_score =sort_type.addAction("按得分正序")
        sort_score.triggered.connect(lambda: self.sort_photo(person_name,"按得分正序"))
        sort_score_reverse=sort_type.addAction("按得分倒序")
        sort_score_reverse.triggered.connect(lambda: self.sort_photo(person_name,"按得分倒序"))
        sort_nsfw=sort_type.addAction("按NSFW正序")
        sort_nsfw.triggered.connect(lambda: self.sort_photo(person_name,"按NSFW正序"))
        sort_nsfw_reverse=sort_type.addAction("按NSFW倒序")
        sort_nsfw_reverse.triggered.connect(lambda: self.sort_photo(person_name,"按NSFW倒序"))
        sort_random=sort_type.addAction("随机排序")
        sort_random.triggered.connect(lambda: self.sort_photo(person_name,"随机排序"))
        # sort_photo右键点击事件
        
        sort_photo.setMenu(sort_type)
        toolbar.addWidget(sort_photo)
        
        # 添加开关
        switch_btn = KeyButton("",tooltip="NSFW switch")
        switch_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        switch_btn.setFixedSize(40,40)

        select_btn.clicked.connect(lambda _,select_btn=select_btn,heart_btn=heart_btn,delete_btn=delete_btn,sort_photo=sort_photo,switch_btn=switch_btn,database_slash=database_slash,setting_btn=setting_btn: self.select_all(select_btn,heart_btn,delete_btn,sort_photo,switch_btn,database_slash,setting_btn ,person_name, image_paths,person_name_db))
        if not self.is_select_page:
            heart_btn.setEnabled(False)
            heart_btn.setVisible(False)
            delete_btn.setEnabled(False)
            delete_btn.setVisible(False)
            database_slash.setEnabled(False)    
            database_slash.setVisible(False)
            sort_photo.setEnabled(True)
            switch_btn.setEnabled(True)
            setting_btn.setEnabled(False)
            setting_btn.setVisible(False)
            select_btn.setIcon(qta.icon("fa5s.tasks",color="#c8c8c8"))
        else:
            delete_btn.setVisible(True)
            delete_btn.setEnabled(True)
            database_slash.setVisible(True)
            database_slash.setEnabled(True)
            heart_btn.setVisible(True)
            heart_btn.setEnabled(True)
            switch_btn.setEnabled(False)
            sort_photo.setEnabled(False)
            select_btn.setIcon(qta.icon("fa5s.check",color="#c8c8c8"))
            setting_btn.setEnabled(True)
            setting_btn.setVisible(True)
        if self.is_open_nsfw:
            switch_btn.setIcon(qta.icon("fa5s.toggle-on",color="#c8c8c8")) 
            nsfw_list=[item for item in self.person_name_db if item["is_nsfw"] == True]
            nsfw_list.sort(key=lambda x: x["nsfw_score"],reverse=True)
            person_name_db=nsfw_list
        else:
            switch_btn.setIcon(qta.icon("fa5s.toggle-off",color="#c8c8c8")) 
        switch_btn.setIconSize(QSize(32, 32))
        # 开关点击后切换形状
        switch_btn.clicked.connect(self.switch_nsfw)
        toolbar.addWidget(switch_btn)
       
        
        self.preview_layout.addLayout(toolbar)
        
        # 创建网格容器
        grid_widget = QWidget()
        grid_widget.setStyleSheet(MAIN_WINDOW_STYLE)  # 使用相同的深色背景
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        if "重复文件" in person_name:
            person_name_db.sort(key=lambda x: x["duplicate_index"],reverse=False)
            image_paths=[item["photo_path"] for item in person_name_db]
            duplicate_index=[item["duplicate_index"] for item in person_name_db]
            cows=Counter(duplicate_index)
            # print("cows",cows)
        else:
            image_paths=[item["photo_path"] for item in person_name_db]

        # 保存图片路径用于重排
        self.current_preview_images = image_paths
        
        # 计算缩放后的尺寸和列数
        base_size = 200
        scale = 1.2
        thumbnail_size = int(base_size * scale)
        scroll_width = self.width()  # 使用完整宽度
        columns = max(1, scroll_width // thumbnail_size)
        self.image_paths = image_paths
        person_info_label=QLabel()
        # 添加缩略图
        row=col=0
        j=0
        duplicate_index=[item["duplicate_index"] for item in person_name_db]
        for i, path in enumerate(image_paths):

            # 创建缩略图容器
            thumb_container = QWidget()
            thumb_container.setFixedSize(thumbnail_size, thumbnail_size)
            thumb_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            # 创建缩略图标签
            thumb_label = QLabel()
            thumb_label.setFixedSize(thumbnail_size, thumbnail_size)
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_label.setText("加载中...")
            thumb_container.setStyleSheet(container_style)
            thumb_label.setStyleSheet(label_style)
            #添加一个鼠标长时间悬停事件
            # thumb_label.setMouseTracking(True)
            # thumb_label.enterEvent = lambda event: thumb_label.setText("悬停")
            # thumb_label.leaveEvent = lambda event: thumb_label.setText("加载中...")
            def create_hover_handler(label,i):
                def enterEvent(event):
                    # label.setText("悬停")
                    # label.setToolTip("悬停")
                    #获取鼠标位置
                    # mouse_pos = (event.globalPos())
                    #获取鼠标位置在self.person_name_db中的索引
                    #获取label 位置
                    label_pos = label.mapToGlobal(QPoint(0, 0))
                    # mouse_pos = label.mapFromGlobal(event.globalPos())
                    person_index=self.get_image_path_index(self.photo_db,image_paths[i])
                    str = f"姓名：{self.photo_db[person_index]['name']} 年龄：{self.photo_db[person_index]['age']}\n "
                    # str += f"性别：{self.photo_db[person_index]['gender']}\n"
                    str += f"is_star： {self.photo_db[person_index]['star']} "
                    str += f"score： {self.photo_db[person_index]['love_score']}\n"
                    if self.photo_db[person_index]['nsfw_class'] != "" and self.photo_db[person_index]['nsfw_class'] is not None:
                        str += f"nsfw_class： {self.photo_db[person_index]['nsfw_class']}\n "
                        str += f"NSFW： {self.photo_db[person_index]['nsfw_score']}\n"
                    # 获取相对根目录路径
                    root_path = os.path.dirname(self.photo_db_path)
                    relative_path = os.path.relpath(self.photo_db[person_index]['photo_path'], root_path)
                    #去除开头的../
                    relative_path = relative_path.replace("..\\", "")
                    str += f" {relative_path}"


                    person_info_label.setText(str)
                    # person_info_label.setFixedSize(200,100)
                    person_info_label.setStyleSheet("background-color: #1e1e1e; color: white;")
                    person_info_label.setAlignment(Qt.AlignCenter)
                    #移除标题栏
                    person_info_label.setWindowFlags(Qt.FramelessWindowHint)
                    # person_info_label.setFixedSize(100,40)
                    person_info_label.move(label_pos.x(),label_pos.y())
                    person_info_label.show()

                def leaveEvent(event):
                    person_info_label.hide()
                #     label.setText("加载中...")
    
                    # 将事件处理器绑定到标签
                thumb_label.enterEvent = enterEvent
                thumb_label.leaveEvent = leaveEvent
            create_hover_handler(thumb_label,i)   
            
            # 添加点击事件
            def create_click_handler(index,label):
                #删除thumb_label中的check_btn
                if self.is_select_page:
                    # self.selected_images.append(image_paths[index])
                    check_btn =QCheckBox("check_btn")
                    check_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
                    if image_paths[index] in self.selected_images:
                        self.selected_images.remove(image_paths[index])  
                        label.setStyleSheet("border: none;")
                    else:
                        self.selected_images.append(image_paths[index])
                        label.setStyleSheet("border: 2px solid red;")
                else:
                    image_index=self.get_image_path_index(self.photo_db,image_paths[index])
                    self.photo_db[image_index]["open_time"]=time.time()
                    return lambda: self.show_photo(image_paths, index)
                
            def show_context_menu(event, image_paths, index):
                if event.button() == Qt.RightButton:
                    menu = QMenu()
                    menu.setStyleSheet("background-color: #1e1e1e; color: white;")
                    # 加人脸注册选项
                    register_action = menu.addAction("人脸注册")
                    register_action.triggered.connect(lambda: self.register_face(image_paths[index]))
                    setting_person_info=menu.addAction("设置人物信息")
                    setting_person_info.triggered.connect(lambda: self.setting_person_info([image_paths[index]]))
                    # 添加删除选项
                    delete_action = menu.addAction("删除照片")
                    delete_action.triggered.connect(lambda: self.delete_photo(image_paths[index]))
                    database_slash_action = menu.addAction("移出数据库")
                    database_slash_action.triggered.connect(lambda: self.delete_photo(image_paths[index],is_delete_local=False))
                    open_action = menu.addAction("打开文件夹")
                    open_action.triggered.connect(lambda: QProcess.startDetached('explorer.exe', ['/select,', image_paths[index]]))
                    # 在鼠标位置显示菜单
                    # setting_person_info=menu.addAction("设置人物信息")
                    # setting_person_info.triggered.connect(lambda: self.setting_person_info(image_paths[index]))
                    menu.exec_(event.globalPos())
            
            # 绑定右键菜单事件
            def handle_mouse_press(event,i,label):
                if event.button() == Qt.LeftButton:
                    if self.is_select_page:
                       create_click_handler(i,label)
                    else:
                        handler=create_click_handler(i,label)
                        handler()
                if event.button() == Qt.RightButton:
                    show_context_menu(event, image_paths, i)
            thumb_label.mousePressEvent = lambda e, i=i,label=thumb_label: handle_mouse_press(e, i,label)
            
            
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
                                    label.setStyleSheet("border: none;")
                    except Exception as e:
                        print(f"加载缩略图失败 {img_path}: {str(e)}")
            
            # 使用更大的时间间隔
            # QTimer.singleShot(50 * i, lambda l=thumb_label, p=path: load_thumbnail(l, p))
            # def create_delayed_loader(label, path, delay):
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda l=thumb_label, p=path: load_thumbnail(l, p))
            timer.start(100 * i)
            # self.thumbnail_timers = []

            # 使用时:
            # timer = create_delayed_loader(thumb_label, path, 50 * i)
            self.thumbnail_timers.append(timer)
            # 添加到容器
            layout = QVBoxLayout(thumb_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(thumb_label)
            
            # 添加到网格
            grid_layout.addWidget(thumb_container, row, col, Qt.AlignTop)
            col = col+1
            if "重复文件" in person_name:
                if col >= cows[duplicate_index[i]]:
                    row = row + 1
                    col = 0
                    j += 1
            if col > columns:
                row = row + 1
                col = 0
        # 设置滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        scroll_area.setWidget(grid_widget)
        self.preview_layout.addWidget(scroll_area)
        self.preview_page.resizeEvent = self.resizeEvent
        # 切换到预览页面
        self.stacked_layout.setCurrentWidget(self.preview_page)
    def love_photo(self,image_index,love_btn):
        """爱心按钮"""
        if self.photo_db[image_index]["star"]:
            # print(self.photo_db[image_index]["star"])
            self.photo_db[image_index]["star"]=False
            self.photo_db[image_index]["love_score"]-=50
            love_icon = qta.icon("fa5s.heart",option=[{"color":"#1e1e1e","border":"2px solid #c8c8c8"}])
            love_btn.setIcon(love_icon)
        else:
            # print(self.photo_db[image_index]["star"])
            self.photo_db[image_index]["star"]=True
            self.photo_db[image_index]["love_score"]+=50
            love_icon = qta.icon("fa5s.heart",color="#c8c8c8",border=True,border_color="#c8c8c8")
            love_btn.setIcon(love_icon)
        # self.show_photo(image_paths,current_index)
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
        toolbar.setAlignment(Qt.AlignCenter)
        toolbar.setContentsMargins(10, 10, 10, 10)
        
        # 返回按钮
        back_btn = KeyButton("返回")
        back_btn.clicked.connect(lambda: (
            self.stacked_layout.setCurrentWidget(self.preview_page),
            self.stacked_layout.removeWidget(self.photo_view_page),
            self.restore_events()
        ))
        
        # 添加照片计数
        count_label = QLabel(f"{current_index + 1} / {len(image_paths)}")
        count_label.setStyleSheet("border:none;color:white;font-size:24px;background-color:transparent;padding:0;margin:0;")
        
        # 将组件添加到工具栏
        toolbar.addWidget(back_btn)
        toolbar.addStretch()
        toolbar.addWidget(count_label)
        # 让返回按钮和计数标签居中
        toolbar.addStretch()
        # 爱心按钮
        love_btn = QPushButton()
        love_btn.setFixedSize(40,40)
        # love_btn.setIcon(qta.icon("fa5s.heart",color="#c8c8c8"))
        love_btn.setIconSize(QSize(32, 32))
        love_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        image_index=self.get_image_path_index(self.photo_db,image_paths[current_index])
        self.photo_db[image_index]["love_score"]+=1
        if self.photo_db[image_index]["star"]:
            love_icon = qta.icon("fa5s.heart",color="#c8c8c8",border=True,border_color="#c8c8c8")
            love_btn.setIcon(love_icon)
        else:
            love_icon = qta.icon("fa5s.heart",option=[{"color":"#1e1e1e","border":"2px solid #c8c8c8"}])
            love_btn.setIcon(love_icon)
            # love_btn.setStyleSheet("border: 2px solid #ffffff;")
        love_btn.clicked.connect(lambda _,love_btn=love_btn:self.love_photo(image_index,love_btn))
        setting_person_info_btn = QPushButton()
        # setting_person_info_btn.setFixedSize(100, 40)
        setting_person_info_btn.setStyleSheet(SWITCH_BUTTON_STYLE)
        setting_person_info_btn.setIcon(qta.icon("fa5s.user-edit",color="#c8c8c8"))
        setting_person_info_btn.clicked.connect(lambda: self.setting_person_info([image_paths[current_index]]))
        toolbar.addWidget(setting_person_info_btn)
        toolbar.addWidget(love_btn)


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
            if self.stacked_layout.currentWidget() == self.photo_view_page:
                if event.key() in (Qt.Key_Left, Qt.Key_Up):
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

            prev_area = QRect(0, 0, 150, photo_container.height())
            next_area = QRect(photo_container.width() - 150, 0, 150, photo_container.height())
            
            # 检查鼠标位置
            pos = photo_container.mapFromGlobal(self.mapToGlobal(event.pos()))
            
            # 显示/隐藏按钮
            if prev_area.contains(pos):
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
        self.save_photo_db()
        if hasattr(self, 'thumbnail_timers'):
            self.cleanup_timers()
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
        if hasattr(self, 'resize_timer'):
            self.resize_timer.stop()
            self.resize_timer.deleteLater()
            del self.resize_timer

        
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
        # reply = QMessageBox.question(
        #     self,
        #     "未识别的人脸",
        #     f"在图片 {os.path.basename(image_path)} 中检测到未识别的人脸，是否要注册？"
        # )
        
        
        # if reply == QMessageBox.Yes:
        self.register_face(image_path)
        # else:
        #     self.photo_db.append(self.person_info_base.copy())
        #     self.photo_db[-1]["name"]="未识别"
        #     self.photo_db[-1]["photo_path"]=image_path
        #     self.photo_db[-1]["embedding"]=np.array([])
        #     self.photo_db[-1]["add_time"]=time.time()
        #     self.photo_db[-1]["open_time"]=time.time()

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
                self.is_first_page = True
                self.stacked_layout.setCurrentWidget(self.cards_page)
                self.left_btn.setEnabled(False)
                self.right_btn.setEnabled(True)
                self.page_title.setText("人物界面")  # 更新标题
            else:
                # 切换到第二个页面（未识别/无人脸页面）
                self.stacked_layout.setCurrentWidget(self.cards_page_2)
                self.is_first_page = False
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
            QTimer.singleShot(0, self.process_photo_db)
        
        self.fade_anim.finished.connect(switch_complete)
        self.fade_anim.start()
    def __del__(self):
        self.cleanup_timers()
        if hasattr(self, 'face_clustering_worker'):
            self.face_clustering_worker.stop()
            self.face_clustering_worker.deleteLater()
            del self.face_clustering_worker
        if hasattr(self, 'remove_duplicates_worker'):
            self.remove_duplicates_worker.stop()
            self.remove_duplicates_worker.deleteLater()
            del self.remove_duplicates_worker
        if hasattr(self, 'nsfw_worker'):
            self.nsfw_worker.stop()
            self.nsfw_worker.deleteLater()
            del self.nsfw_worker
        if hasattr(self, 'photo_process_worker'):
            self.photo_process_worker.stop()
            self.photo_process_worker.deleteLater()
            del self.photo_process_worker
        if hasattr(self, 'process_worker'):
            self.process_worker.stop()
            self.process_worker.deleteLater()
            del self.process_worker
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
        if hasattr(self, 'resize_timer'):
            self.resize_timer.stop()
            self.resize_timer.deleteLater()
            del self.resize_timer
class NSFWWorker(QThread):
    finished = pyqtSignal(list)
    def __init__(self, photo_db,nsfw_classifier,unsafe_threshold):
        super().__init__()
        self.photo_db = photo_db
        self.nsfw_classifier = nsfw_classifier
        self.unsafe_threshold = unsafe_threshold
    def run(self):
        no_nsfw_photo =[item for item in self.photo_db if item["is_nsfw"] is None]
        if len(no_nsfw_photo) == 0:
            self.finished.emit(self.photo_db)
            return
        for item in self.photo_db:
            if item["is_nsfw"] is None :
                # print("开始检测 nsfw")
                is_nsfw,score,class_name = self.nsfw_classifier.classify_image(item["photo_path"])
                if is_nsfw:
                    item["is_nsfw"] = True
                    item["nsfw_score"] = score
                    item["nsfw_class"] = class_name
                else:
                    item["is_nsfw"] = False
                    item["nsfw_score"] = 0
                    item["nsfw_class"] = "safe"
        self.finished.emit(self.photo_db)

# 添加人脸聚类线程
class FaceClusteringWorker(QThread):
    finished = pyqtSignal(list)
    def __init__(self, photo_db, face_clusterer,face_organizer):
        super().__init__()
        self.photo_db = photo_db
        self.face_clusterer = face_clusterer
        self.face_organizer = face_organizer
    def run(self):
        embeddings = []
        all_paths = []
        no_cluster_face =[item for item in self.photo_db if item['face_cluster_index'] is None]
        if len(no_cluster_face) == 0:
            self.finished.emit([])
            return
        clustered_face = [item for item in self.photo_db if item["face_cluster_index"] is not None and item["face_cluster_index"] != -1]
        # print("开始人脸聚类")
        for item in self.photo_db:
            if item["embedding"] == np.array([]):
                continue
            if len(item["embedding"]) != 512:
                continue
            if item in clustered_face:
                continue
            if item['duplicate_index'] == -1:
                continue
            else:
                for clustered_item in clustered_face:
                    if self.face_organizer.compare_face_from_embedding(item["embedding"],clustered_item["embedding"]):
                        item["face_cluster_index"] = clustered_item["face_cluster_index"]
                        break
                
        for item in self.photo_db:
            if item['embedding'] != np.array([])  and item['name'] == "未识别":
                # print(item['name'])
                if item["face_cluster_index"] is None or item["face_cluster_index"] == -1:
                    embeddings.append(item['embedding'])
                    all_paths.append(item['photo_path'])
        if len(embeddings) > 0:
            clusters = self.face_clusterer.cluster_faces(embeddings,all_paths)
            # print(len(clusters))
            self.finished.emit(clusters)
        else:
            self.finished.emit([])
        # print("人脸聚类完成")
#添加去除重复文件线程
class RemoveDuplicatesWorker(QThread):
    finished = pyqtSignal(list)
    def __init__(self, to_duplicate_photo,photo_db,duplicates_remover):
        super().__init__()
        self.to_duplicate_photo = to_duplicate_photo
        self.photo_db = photo_db
        self.duplicates_remover = duplicates_remover
        # self.duplicates_remover = DuplicateRemover(similarity_threshold=similarity_threshold)
    def run(self):
        # print("开始去重")
        # no_duplicates_photo =[item for item in self.photo_db if item["duplicate_index"] is None]
        if len(self.to_duplicate_photo) == 0:
            self.finished.emit(self.photo_db)
            return
        duplicates_index=[item['duplicate_index'] for item in self.photo_db if 'duplicate_index' in item and item['duplicate_index'] is not None]
        self.duplicates_remover.build_annoy_index(self.photo_db)
        for i in range(len(self.to_duplicate_photo)):
            # print("开始去重 ",i)
            similar_photo = self.duplicates_remover.find_similar_by_annoy_deep(self.to_duplicate_photo[i]['photo_path'])
            image_index_2 = self.photo_db.index(self.to_duplicate_photo[i])
            if len(similar_photo) > 1:
                # print("len(similar_photo)",len(similar_photo))
                # self.to_duplicate_photo[i]['duplicate_index'] = -1
            #    ? image_index_2 = self.photo_db.index(self.to_duplicate_photo[i])
                for item in similar_photo:
                    image_path_list = [item['photo_path'] for item in self.photo_db]
                    image_index_1 = image_path_list.index(item)
                    # image_index_2 = image_path_list.index(self.to_duplicate_photo[i]['photo_path'])
                    if image_index_1 != -1 and image_index_2 != -1:
                        if image_index_1 == image_index_2:
                            continue
                        if self.photo_db[image_index_1]['duplicate_index'] is not None and self.photo_db[image_index_1]['duplicate_index'] != -1:
                            self.photo_db[image_index_2]['duplicate_index'] = self.photo_db[image_index_1]['duplicate_index']
                            break
                        else:
                            index = 0
                            while True:
                                if index not in duplicates_index:
                                    self.photo_db[image_index_1]['duplicate_index'] = index
                                    self.photo_db[image_index_2]['duplicate_index'] = index
                                    duplicates_index.append(index)
                                    break
                                index+=1
                            break
            else:  
                # print("去重完成 -1",image_index_2)
                self.photo_db[image_index_2]['duplicate_index'] =  -1
            # print("去重完成",i)
        # print("去重完成")    
        self.finished.emit(self.photo_db)
        return
# 添加工作线程类
class PhotoProcessWorker(QThread):
    """照片处理工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    
    def __init__(self, photo_db, face_organizer):
        super().__init__()
        self.photo_db = photo_db
        self.face_organizer = face_organizer
        self._is_running = True
    def run(self):
        results = {
            'face_db': {},
        }
        
        # root = Path(self.photo_db)
        # total_files = sum(1 for _ in root.rglob("*.jpg"))
        total_files = len(self.photo_db)
        processed = 0
        
        # 初始化特殊类别
        results['face_db']["未识别"] = []
        results['face_db']["无人脸"] = []
        
        # 遍历文件夹
        for item in self.photo_db:
            person_name = item['name']
            photo_path = item['photo_path'] 
            if not os.path.exists(photo_path):
                self.photo_db.remove(item)
                continue
            # 处理特殊目录
            if person_name == "未识别":
                # 将"未识别"目录的照片添加到未识别类别
                try:
                    results['face_db']["未识别"].append(str(photo_path))
                    processed += 1
                    self.progress.emit(int(processed * 100 / total_files))
                except Exception as e:
                    print(f"处理照片失败 {photo_path}: {str(e)}")
                    continue
            elif person_name == "无人脸":
                # 将"无人脸"目录的照片添加到无人脸类别
                try:
                    results['face_db']["无人脸"].append(str(photo_path))
                    processed += 1
                    self.progress.emit(int(processed * 100 / total_files))
                except Exception as e:
                        print(f"处理照片失败 {photo_path}: {str(e)}")
                
            
            # 处理普通人物目录
            else:
                if person_name not in results['face_db']:
                    results['face_db'][person_name] = []
                try:
                    results['face_db'][person_name].append(str(photo_path))
                    processed += 1
                    self.progress.emit(int(processed * 100 / total_files))
                except Exception as e:
                    print(f"处理照片失败 {photo_path}: {str(e)}")
                    continue
            if 'is_nsfw' in item and item["is_nsfw"] == True:
                if item["nsfw_class"] not in results['face_db']:
                        results['face_db'][item["nsfw_class"]] = []
                results['face_db'][item["nsfw_class"]].append(item["photo_path"])
            if item["face_cluster_index"] is not None and not item['face_cluster_index'] == -1:
                if "未知人物_{}".format(item["face_cluster_index"]) not in results['face_db']:
                    results['face_db']["未知人物_{}".format(item["face_cluster_index"])] = []
                results['face_db']["未知人物_{}".format(item["face_cluster_index"])].append(item["photo_path"])
            if item["duplicate_index"] is not None and item['duplicate_index'] != -1:
                if "重复文件" not in results['face_db']:
                    results['face_db']["重复文件"] = []
                results['face_db']["重复文件"].append(item["photo_path"])
            if 'star' in item and item["star"] == True:
                if "收藏" not in results['face_db']:
                    results['face_db']["收藏"] = []
                results['face_db']["收藏"].append(item["photo_path"])
            if '全部' not in results['face_db']:
                results['face_db']['全部'] = []
            results['face_db']['全部'].append(item["photo_path"])
        # 移除空类别
        empty_categories = [k for k, v in results['face_db'].items() if not v]
        for category in empty_categories:
            del results['face_db'][category]
        # 只更新界面显示，不保存到数据库
        self.finished.emit(results)

class ProcessWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict,list)
    error = pyqtSignal(str)
    register_request = pyqtSignal(str)
    # total_register_person = pyqtSignal(dict)        
    def __init__(self, files, face_organizer,nsfw_classifier,mode):
        super().__init__()
        self.files = files
        self.face_organizer = face_organizer
        self.nsfw_classifier = nsfw_classifier
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
        matched_name,matched_embedding,age,gender = self.face_organizer.compare_face(normalize_path(file_path))
        person_info = None
        print("process_file",file_path)
        if matched_name:
            # 创建对应人名的文件夹
            # copy_photo(matched_name,file_path,self.output_dir)
            person_info = person_info_base.copy()
            person_info["name"]=matched_name
            person_info["age"]=age
            person_info["gender"]=gender
            person_info["embedding"]=matched_embedding
            person_info["photo_path"]=normalize_path(file_path)
            person_info["add_time"]=time.time()
            person_info["open_time"]=time.time()
           
        else:
                # 发出注册请求信号并等待
                self.pause()  # 暂停线程
                self.register_request.emit(file_path)
                self.wait_condition.wait()  # 等待恢复信号
                
                self.processed += 1
                self.progress.emit(int(self.processed * 100 / self.total_files))
        return person_info
    def run(self):
        self.total_register_person ={} 
        self.total_register_person_info =[]
        if self.mode == "folder":
            try:
                for root, _, files in os.walk(self.files):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            full_path = os.path.join(root, file)
                            person_info = self.process_file(full_path)
                            if person_info is not None:
                                self.total_register_person_info.append(person_info)
                                if person_info["name"] not in self.total_register_person:
                                    self.total_register_person[person_info["name"]] = 1
                                else:
                                    self.total_register_person[person_info["name"]] += 1
            except Exception as e:
                print("error",e)
                self.error.emit(str(e))
            

        elif self.mode == "file":
            person_info =  self.process_file(self.files)
            if person_info["name"] is not None:
                if person_info["name"] not in self.total_register_person:
                    self.total_register_person[person_info["name"]] = 1
                else:
                    self.total_register_person[person_info["name"]] += 1
                self.total_register_person_info.append(person_info)
        self.finished.emit(self.total_register_person,self.total_register_person_info)
        self.total_register_person ={}

if __name__ == '__main__':
    # 在创建QApplication之前设置高DPI缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    window = PhotoManager()
    window.show()
    sys.exit(app.exec_()) 


