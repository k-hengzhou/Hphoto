"""Qt组件样式表配置"""

# 按钮样式
BUTTON_STYLE = """
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
"""

# 左侧按钮样式
LEFT_BUTTON_STYLE = """
    QPushButton {
        background-color: rgba(32, 32, 32, 0.85);
        border: none;
    }
    QPushButton::indicator {
        image:none;
        width:0px;
        height:0px;
        background:none;
        border:none;
    }
    QPushButton::menu-indicator {
        image: none;
        width: 0;
        height: 0;
        padding: 0;
        margin: 0;
        border: none;
        background: none;
        subcontrol-position: none;
        subcontrol-origin: none;
    }
"""

# 菜单样式
MENU_STYLE = """
    QMenu {
        background-color: rgba(32, 32, 32, 0.85);
        color: #c8c8c8;
        border: none;
        font-family: "Segoe UI", Arial;
        font-size: 24px;
    }
    QMenu::item {
        color: #c8c8c8;
    }
    QMenu::item:selected {
        background-color: rgba(98, 114, 164, 0.9);
    }
    QMenu::item:hover {
        background-color: rgba(128, 128, 128, 0.3);
    }
    QMenu::item:disabled {
        background-color: rgba(45, 45, 45, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    QMenu::indicator,
    QMenu::icon,
    QMenu::separator,
    QMenu::tearoff {
        image: none;
        width: 0px;
        height: 0px;
        padding: 0px;
        margin: 0px;
        border: none;
        background: none;
    }
    QMenu::right-arrow {
        image: none;
        width: 0px;
        height: 0px;
    }
"""

# 页面标题样式
PAGE_TITLE_STYLE = """
    QLabel {
        color: #c8c8c8;
        font-size: 16px;
        font-weight: bold;
        background-color: rgba(45, 45, 45, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 5px 25px;
        min-width: 150px;
    }
"""

# 切换按钮样式
SWITCH_BUTTON_STYLE = """
    QPushButton {
        background-color: rgba(45, 45, 45, 0.4);
        border: none;
        font-size: 24px;
    }
    QPushButton:hover {
        background-color: rgba(60, 60, 60, 0.6);
        border: none;
    }
    QPushButton:disabled {
        background-color: rgba(45, 45, 45, 0.2);
        border: none;
    }
    QPushButton::menu-indicator {
        image: none;
        width: 0;
        height: 0;
        padding: 0;
        margin: 0;
        border: none;
        background: none;
        subcontrol-position: none;
        subcontrol-origin: none;
    }
    QPushButton::indicator {
        image:none;
        width:0px;
        height:0px;
        background:none;
        border:none;
    }
"""

# 消息框样式
MESSAGE_BOX_STYLE = """
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
        """

# 工具提示样式
TOOLTIP_STYLE = """
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
"""

# 下拉框样式
COMBOBOX_STYLE = """
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
"""

# 主窗口样式
MAIN_WINDOW_STYLE = """
    QMainWindow, QWidget {
        background-color: rgba(32, 32, 32, 0.85);
        border: none;
        border-radius: 0px;
        padding: 0px;
        margin: 0px;
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
        border:none;
    }
    QScrollBar::handle:vertical:hover {
        background: rgba(128, 128, 128, 0.7);
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
        height: 0;
    }
"""

# 设置对话框样式
SETTINGS_DIALOG_STYLE = """
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
""" 
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
label_style = """
    QLabel {
            border: none;
            margin: 0;
            padding: 0;
            background: transparent;
    }
    """
QComboBox_style = """
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
        """
