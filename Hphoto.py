from src import PhotoManager
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtCore import Qt

def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DisableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = PhotoManager()
    # 设置应用图标
    app_icon = QIcon("assets/image.ico")
    window.setWindowIcon(app_icon)
    QApplication.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec_()) 



if __name__ == "__main__":
    main()
