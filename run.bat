@echo off
title HPhoto
echo 正在启动 HPhoto...

:: 最小化命令行窗口
powershell -window minimized -command ""

:: 直接使用环境中的 Python
start "" /b C:\Users\khz\anaconda3\envs\main\python.exe Hphoto.py

if errorlevel 1 (
    echo 程序运行出错！
    pause
)

exit 