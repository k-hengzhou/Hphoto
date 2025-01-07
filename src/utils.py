import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QScrollArea, QLabel, 
                            QFileDialog, QGridLayout, QMessageBox, QProgressBar, 
                            QDialog, QStackedWidget, QSizePolicy, QLineEdit, QMenu, QComboBox)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, 
                         QEasingCurve, QSize, QRect, QObject, QPoint, QEvent)
from PyQt5.QtGui import QPixmap, QPainter, QImage, QImageReader
from pathlib import Path
import shutil
import cv2
def normalize_path(path):
    """统一处理文件路径
    
    Args:
        path: 原始路径
        
    Returns:
        处理后的路径
    """
    # 统一使用正斜杠
    path = path.replace('\\', '/')
    
    # 处理中文路径
    path = os.path.abspath(os.path.normpath(path))
    
    return path

def check_path(path, create_if_missing=False):
    """检查路径是否有效
    
    Args:
        path: 要检查的路径
        create_if_missing: 如果路径不存在是否创建
    
    Returns:
        处理后的路径
    
    Raises:
        FileNotFoundError: 路径不存在且未设置创建
    """
    path = normalize_path(path)
    
    if not os.path.exists(path):
        if create_if_missing:
            os.makedirs(path, exist_ok=True)
        else:
            raise FileNotFoundError(f"路径不存在: {path}")
    
    return path 

def ensure_directory(path):
    """确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        创建的目录路径
    """
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        raise OSError(f"创建目录失败 {path}: {str(e)}")

def check_input_directory(path):
    """检查输入目录是否存在且包含图片
    
    Args:
        path: 输入目录路径
        
    Returns:
        bool: 目录是否有效
    """
    if not os.path.exists(path):
        print(f"错误：输入目录不存在: {path}")
        return False
    
    # 检查是否包含图片
    has_images = False
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                has_images = True
                break
        if has_images:
            break
    
    if not has_images:
        print(f"警告：目录中未找到图片文件: {path}")
        return False
    
    return True 

def copy_photo(person_name,image_path,last_folder):
    """复制照片"""
    image_dir = os.path.abspath(image_path)
    last_folder_dir = os.path.abspath(last_folder)
    relative_path = os.path.relpath(image_dir, last_folder_dir)
    try:
        if not relative_path.startswith(".."):
            print("已存在同名文件夹")
            return True
        else:
            if person_name == os.path.basename(last_folder):
                person_dir = last_folder
            else:
                person_dir = os.path.join(last_folder, person_name)
            print(person_dir)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir, exist_ok=True)
                # 复制照片

            dest_path = os.path.join(person_dir, os.path.basename(image_path))
            shutil.copy(image_path, dest_path)
    except Exception as e:
        print(f"复制照片失败: {e}")
        return False
    return True
def cv2_to_qpixmap(cv_img):
    # 1. 转换颜色空间（OpenCV 是 BGR，Qt 是 RGB）
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # 2. 转换为 QImage
    height, width, channel = rgb_image.shape
    bytes_per_line = channel * width
    q_image = QImage(
        rgb_image.data, 
        width, 
        height, 
        bytes_per_line, 
        QImage.Format_RGB888
    )
    
    # 3. 转换为 QPixmap
    return QPixmap.fromImage(q_image)
def get_config(config,name1,name2,default=None):
    if name1 not in config:
        return default
    elif name2 not in config[name1]:
        return default
    else:
        return config[name1][name2]
