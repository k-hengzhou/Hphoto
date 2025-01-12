@echo off
title HPhoto
echo 正在启动 HPhoto...

:: 直接使用环境中的 Python
C:\Users\khz\anaconda3\envs\main\python.exe Hphoto.py

if errorlevel 1 (
    echo 程序运行出错！
    pause
)

pause 